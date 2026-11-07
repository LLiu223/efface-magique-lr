"""Export a fine-tuned LaMa checkpoint to TorchScript with big-lama.pt's contract, plus meta.json.

    python -m tools.finetune.export CKPT OUT_DIR --lama-dir LAMA --manifest CURATED/manifest.json

CKPT is a checkpoint from train.py (WORK/<name>/models/last.ckpt; its config.yaml is read from WORK/<name>/).
model.pt takes (image float [1,3,H,W] in 0..1, mask [1,1,H,W] with >0 = hole) and returns the inpainted
image [1,3,H,W] in 0..1, like big-lama.pt. The file is reloaded with torch.jit.load and run once before
OUT_DIR is created; on any failure nothing is left behind. Never writes to big-lama.pt.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Must match companion/models/tuned.py META_KEYS.
META_KEYS = ("name", "date", "base_ckpt", "manifest_hash", "steps", "lr", "metrics")


def default_big_lama_pt() -> Path:
    """Where simple-lama-inpainting caches big-lama.pt (torch hub dir), without importing torch."""
    home = os.environ.get("TORCH_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "torch")
    return Path(home) / "hub" / "checkpoints" / "big-lama.pt"


def refuses_base(out_dir: Path, base_pt: Path) -> bool:
    """True if OUT_DIR, or OUT_DIR/model.pt, is (or links to) the base big-lama.pt."""
    base = Path(base_pt).resolve()
    for p in (Path(out_dir), Path(out_dir) / "model.pt"):
        if p.resolve() == base or (p.exists() and base.exists() and os.path.samefile(p, base)):
            return True
    return False


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_meta(name, base_ckpt, manifest_hash, steps, lr, metrics=None, date=None) -> dict:
    meta = {"name": name, "date": date or datetime.datetime.now().isoformat(timespec="seconds"),
            "base_ckpt": str(base_ckpt), "manifest_hash": manifest_hash, "steps": int(steps), "lr": float(lr),
            "metrics": metrics or {}}
    validate_meta(meta)
    return meta


def validate_meta(meta) -> None:
    if not isinstance(meta, dict):
        raise ValueError("meta.json must be a JSON object")
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"meta.json missing {', '.join(missing)}")
    types = {"name": str, "date": str, "base_ckpt": str, "manifest_hash": str, "steps": int, "lr": (int, float),
             "metrics": dict}
    for k, t in types.items():
        if not isinstance(meta[k], t) or isinstance(meta[k], bool):
            raise ValueError(f"meta.json {k!r} has the wrong type")


def export_generator(generator, out_dir: Path, meta: dict, base_pt: Path) -> None:
    """Trace generator into the big-lama.pt contract, verify, then move into out_dir atomically."""
    import torch

    out_dir = Path(out_dir)
    if refuses_base(out_dir, base_pt):
        raise SystemExit(f"Refusing: {out_dir} resolves to the base model {base_pt}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Refusing: OUT_DIR {out_dir} is not empty")
    validate_meta(meta)

    class Wrapper(torch.nn.Module):
        def __init__(self, gen):
            super().__init__()
            self.generator = gen

        def forward(self, image, mask):
            mask = (mask > 0).to(image.dtype)  # simple-lama passes an integer mask
            masked = image * (1 - mask)
            out = self.generator(torch.cat([masked, mask], dim=1))
            return mask * out + (1 - mask) * image

    wrapper = Wrapper(generator.eval().cpu()).eval()
    img, mask = torch.rand(1, 3, 256, 256), torch.zeros(1, 1, 256, 256)
    mask[:, :, 96:160, 96:160] = 1
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".export-", dir=out_dir.parent))
    try:
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (img, mask), check_trace=False)
        traced.save(str(tmp / "model.pt"))
        reloaded = torch.jit.load(str(tmp / "model.pt"), map_location="cpu")
        with torch.no_grad():
            res = reloaded(torch.rand(1, 3, 128, 192), (torch.rand(1, 1, 128, 192) > 0.7).long())
        if tuple(res.shape) != (1, 3, 128, 192) or not torch.isfinite(res).all():
            raise RuntimeError(f"verification forward pass returned {tuple(res.shape)} / non-finite values")
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if out_dir.exists():
            out_dir.rmdir()  # empty (checked above)
        os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def load_generator(ckpt: Path, lama_dir: Path):
    """Upstream generator weights from a train.py checkpoint (needs the LaMa checkout on sys.path)."""
    sys.path.insert(0, str(Path(lama_dir).resolve()))
    from omegaconf import OmegaConf
    from saicinpainting.training.trainers import load_checkpoint
    cfg = OmegaConf.load(Path(ckpt).resolve().parent.parent / "config.yaml")
    cfg.training_model.predict_only = True
    cfg.visualizer.kind = "noop"
    model = load_checkpoint(cfg, str(ckpt), strict=False, map_location="cpu")
    return model.generator, cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out_dir", type=Path, help="new or empty folder, e.g. models/finetuned/<name>")
    ap.add_argument("--lama-dir", type=Path, required=True, help="advimman/lama checkout at the pinned commit")
    ap.add_argument("--manifest", type=Path, required=True, help="curate.py manifest.json used for training")
    ap.add_argument("--name", help="model name (default: OUT_DIR folder name)")
    ap.add_argument("--base-ckpt", default="big-lama/models/best.ckpt", help="recorded in meta.json")
    ap.add_argument("--base-pt", type=Path, default=default_big_lama_pt(), help="base big-lama.pt that must never be overwritten")
    a = ap.parse_args(argv)
    if refuses_base(a.out_dir, a.base_pt):
        raise SystemExit(f"Refusing: {a.out_dir} resolves to the base model {a.base_pt}")

    import torch
    state = torch.load(a.ckpt, map_location="cpu")
    lr = next((g["lr"] for s in state.get("optimizer_states", [])[:1] for g in s["param_groups"][:1]), 0.0)
    generator, _ = load_generator(a.ckpt, a.lama_dir)
    meta = build_meta(a.name or a.out_dir.name, a.base_ckpt, sha256_file(a.manifest), state.get("global_step", 0), lr)
    export_generator(generator, a.out_dir, meta, a.base_pt)
    print(f"exported {a.out_dir / 'model.pt'}; run tools.finetune.eval before copying it into models/finetuned/")


if __name__ == "__main__":
    main()
