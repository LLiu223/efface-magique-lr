"""Fine-tune big-lama on a curated set with the pinned upstream LaMa trainer (advimman/lama).

    python -m tools.finetune.train --lama-dir LAMA --big-lama BIG_LAMA_DIR --data CURATED --work WORK --dry-run

Prepares WORK/<name>/data/{train,val,visual_test} from CURATED (curate.py output), writes an init
checkpoint from big-lama's best.ckpt with step 0 and the low LR, then runs upstream bin/train.py with
Hydra overrides. Re-running the same command resumes from WORK/<name>/models/last.ckpt.
--dry-run validates paths and prints the exact command without importing torch or writing anything.
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

MIN_VRAM_GB = 12
TRAIN_MAX_SIDE = 1024  # train images are downscaled to this; LaMa trains on 256 px random crops anyway
VAL_SIZE = 512         # upstream val loader batches images, so they must share one size


def run_dir(a) -> Path:
    return Path(a.work).resolve() / a.name


def resume_ckpt(a) -> Path:
    last = run_dir(a) / "models" / "last.ckpt"
    return last if last.is_file() else run_dir(a) / "init.ckpt"


def train_count(a) -> int:
    manifest = json.loads((Path(a.data) / "manifest.json").read_text(encoding="utf-8"))
    return manifest["counts"]["train"]


def build_command(a, n_train: int):
    """(argv, env additions) for upstream bin/train.py."""
    rd = run_dir(a)
    batches = max(1, math.ceil(n_train / a.batch_size))
    # upstream ModelCheckpoint (period=1, save_last) saves after every validation, so this is "checkpoint every N steps";
    # PL 1.2.9 rejects an interval longer than one epoch, hence the clamp
    every = min(a.ckpt_every, batches)
    overrides = [
        "location=places_example",
        f"location.data_root_dir={(rd / 'data').as_posix()}",
        f"location.out_root_dir={(rd / 'experiments').as_posix()}",
        f"location.tb_dir={(rd / 'tb_logs').as_posix()}",
        f"location.pretrained_models={Path(a.big_lama).resolve().as_posix()}",
        f"hydra.run.dir={rd.as_posix()}",
        f"run_title={a.name}",
        f"data.batch_size={a.batch_size}",
        f"data.train.out_size={a.crop}",
        f"optimizers.generator.lr={a.lr:g}",
        f"optimizers.discriminator.lr={a.lr / 10:g}",
        "trainer.kwargs.gpus=1",
        "trainer.kwargs.accelerator=null",
        f"trainer.kwargs.max_epochs={a.epochs}",
        f"trainer.kwargs.limit_train_batches={batches}",
        f"trainer.kwargs.val_check_interval={every}",
        f"+trainer.kwargs.resume_from_checkpoint={resume_ckpt(a).as_posix()}",
    ]
    argv = [sys.executable, "bin/train.py", "-cn", "big-lama", *overrides]
    lama = Path(a.lama_dir).resolve()
    env = {"PYTHONPATH": str(lama), "TORCH_HOME": str(lama), "USER": os.environ.get("USER", "efface"),
           "HYDRA_FULL_ERROR": "1"}
    return argv, env


def validate(a):
    errors = []
    if not (Path(a.lama_dir) / "bin" / "train.py").is_file():
        errors.append(f"--lama-dir {a.lama_dir}: bin/train.py not found (clone advimman/lama at the pinned commit)")
    if not (Path(a.big_lama) / "models" / "best.ckpt").is_file():
        errors.append(f"--big-lama {a.big_lama}: models/best.ckpt not found (unzip big-lama.zip)")
    if not (Path(a.data) / "manifest.json").is_file():
        errors.append(f"--data {a.data}: manifest.json not found (run tools.finetune.curate first)")
    if not all(c.isalnum() or c in "_-" for c in a.name):
        errors.append(f"--name {a.name!r}: use only [A-Za-z0-9_-] (it becomes the models/finetuned folder)")
    return errors


def check_gpu(min_gb: float):
    """None if a CUDA GPU with at least min_gb is present, else a message (imports torch)."""
    import torch
    hint = "Try a smaller --batch-size (e.g. 2 or 1) and/or --crop 192, then lower --min-vram-gb to match."
    if not torch.cuda.is_available():
        return f"No CUDA GPU found. Fine-tuning needs an NVIDIA GPU with about {MIN_VRAM_GB} GB VRAM. {hint}"
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if total < min_gb:
        return f"GPU has {total:.1f} GB VRAM; at least {min_gb} GB is needed for these settings. {hint}"
    return None


def prepare_data(a):
    """Build upstream's layout: train/*.jpg, val/ and visual_test/ with <stem>.png + <stem>_mask000.png."""
    from PIL import Image, ImageOps
    from tools.finetune.masks import generate_mask, mask_filename, write_mask
    data = Path(a.data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    root = run_dir(a) / "data"
    for sub in ("train", "val", "visual_test"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    val_i = 0
    for r in sorted((r for r in manifest["images"] if r["status"] == "kept"), key=lambda r: r["out"]):
        stem = r["sha256"][:16]  # upstream pairs masks by splitting on "_mask", so keep user names out
        with Image.open(data / r["out"]) as im:
            im = im.convert("RGB")
            if r["split"] == "train":
                im.thumbnail((TRAIN_MAX_SIDE, TRAIN_MAX_SIDE), Image.Resampling.LANCZOS)
                im.save(root / "train" / f"{stem}.jpg", quality=95)
            elif r["split"] == "val":
                im = ImageOps.fit(im, (VAL_SIZE, VAL_SIZE), Image.Resampling.LANCZOS)
                mask = generate_mask(VAL_SIZE, VAL_SIZE, a.seed + val_i, "normal")
                val_i += 1
                for sub in ("val", "visual_test"):
                    im.save(root / sub / f"{stem}.png")
                    write_mask(root / sub / mask_filename(stem), mask)


def prepare_init_ckpt(a):
    """init.ckpt = big-lama best.ckpt restarted at step 0 with the fine-tune LR (resume would restore the old LR)."""
    import torch
    out = run_dir(a) / "init.ckpt"
    if out.exists():
        return
    ckpt = torch.load(Path(a.big_lama) / "models" / "best.ckpt", map_location="cpu")
    ckpt["epoch"], ckpt["global_step"] = 0, 0
    ckpt.pop("lr_schedulers", None)
    ckpt.pop("callbacks", None)
    lrs = (a.lr, a.lr / 10)  # generator, discriminator (upstream configure_optimizers order)
    if "optimizer_states" in ckpt:
        for state, lr in zip(ckpt["optimizer_states"], lrs):
            for g in state["param_groups"]:
                g["lr"] = lr
    else:  # weights-only checkpoint: fresh Adam state over the parameters (buffers excluded)
        buffers = ("running_mean", "running_var", "num_batches_tracked")
        counts = [sum(1 for k in ckpt["state_dict"] if k.startswith(p) and not k.endswith(buffers))
                  for p in ("generator.", "discriminator.")]
        ckpt["optimizer_states"] = [
            {"state": {}, "param_groups": [{"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0,
                                            "amsgrad": False, "params": list(range(n))}]}
            for n, lr in zip(counts, lrs)]
    torch.save(ckpt, out)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--lama-dir", required=True, help="advimman/lama checkout at the pinned commit")
    ap.add_argument("--big-lama", required=True, help="unzipped big-lama folder (config.yaml, models/best.ckpt)")
    ap.add_argument("--data", required=True, help="curate.py OUT_DIR")
    ap.add_argument("--work", required=True, help="working folder for data, checkpoints and logs")
    ap.add_argument("--name", default="mine", help="run name; also the models/finetuned/<name> folder")
    ap.add_argument("--lr", type=float, default=1e-5, help="generator LR (discriminator gets LR/10)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256, help="training crop size (px)")
    ap.add_argument("--ckpt-every", type=int, default=500, help="validate and checkpoint every N steps")
    ap.add_argument("--min-vram-gb", type=float, default=MIN_VRAM_GB)
    ap.add_argument("--seed", type=int, default=0, help="val mask seed")
    ap.add_argument("--dry-run", action="store_true", help="validate and print the command; no torch, no writes")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    errors = validate(a)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    n_train = 0 if errors and not (Path(a.data) / "manifest.json").is_file() else train_count(a)
    argv_, env = build_command(a, n_train)
    print("cd " + shlex.quote(str(Path(a.lama_dir).resolve())))
    print(" ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + " ".join(shlex.quote(x) for x in argv_))
    if errors:
        return 2
    if a.dry_run:
        print("dry run: nothing written")
        return 0
    problem = check_gpu(a.min_vram_gb)
    if problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 3
    prepare_data(a)
    prepare_init_ckpt(a)
    return subprocess.run(argv_, cwd=a.lama_dir, env={**os.environ, **env}).returncode


if __name__ == "__main__":
    sys.exit(main())
