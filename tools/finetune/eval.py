"""Compare a fine-tuned model against big-lama on held-out photos and print ADOPT or KEEP BASELINE.

    python -m tools.finetune.eval --baseline big-lama.pt --candidate model.pt --data CURATED/test --out REPORT_DIR

Each image (sorted by name) gets the fixed mask generate_mask(h, w, seed + i, regime). PSNR, SSIM and LPIPS
are computed inside the mask and on the whole image. Writes report.md, report.json and contact_sheet.jpg.
Verdict rule (thresholds are flags): ADOPT only if mean in-mask LPIPS improves by at least --min-lpips-gain
(relative), mean in-mask PSNR drops by no more than --max-psnr-drop dB, and the candidate has the lower
in-mask LPIPS on more than half of the images. Otherwise KEEP BASELINE.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.finetune.masks import generate_mask

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MIN_LPIPS_GAIN = 0.02  # 2 % relative
MAX_PSNR_DROP = 0.2    # dB


def _region(mask, shape):
    return np.ones(shape[:2], bool) if mask is None else (np.asarray(mask) > 0)


def psnr(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    """PSNR in dB of two uint8 HxWx3 images, over mask>0 pixels (all if mask is None)."""
    sel = _region(mask, a.shape)
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2)[sel].mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM (Wang et al., 11x11 Gaussian sigma 1.5) averaged over channels."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    va, vb, cov = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.mean(axis=2) if s.ndim == 3 else s


def ssim(a, b, mask=None) -> float:
    return float(ssim_map(a, b)[_region(mask, a.shape)].mean())


def verdict(summary: dict, n_images: int, min_lpips_gain=MIN_LPIPS_GAIN, max_psnr_drop=MAX_PSNR_DROP):
    """('ADOPT' | 'KEEP BASELINE', reasons). summary holds mean baseline/candidate metrics and wins."""
    b, c = summary["baseline"], summary["candidate"]
    gain = (b["lpips_mask"] - c["lpips_mask"]) / b["lpips_mask"] if b["lpips_mask"] > 0 else 0.0
    drop = b["psnr_mask"] - c["psnr_mask"]
    checks = [
        (gain >= min_lpips_gain, f"in-mask LPIPS gain {gain:+.1%} (need >= {min_lpips_gain:.0%})"),
        (drop <= max_psnr_drop, f"in-mask PSNR change {-drop:+.2f} dB (may drop at most {max_psnr_drop} dB)"),
        (summary["wins"] * 2 > n_images, f"candidate wins {summary['wins']}/{n_images} (need a majority)"),
    ]
    return ("ADOPT" if all(ok for ok, _ in checks) else "KEEP BASELINE"), [("PASS " if ok else "FAIL ") + t for ok, t in checks]


class Lpips:
    """LPIPS (AlexNet) without network access: refuses to run unless setup already cached the weights."""

    def __init__(self, device):
        import torch
        import lpips
        cached = Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
        if not cached.is_file():
            raise SystemExit(f"LPIPS backbone weights missing ({cached}). Run the one-time setup in "
                             "tools/finetune/README.md; eval never downloads anything.")
        self.torch, self.device = torch, device
        self.net = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()

    def __call__(self, a, b, mask=None) -> float:
        t = lambda x: self.torch.from_numpy(x).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(self.device)  # noqa: E731
        with self.torch.no_grad():
            d = self.net(t(a), t(b))[0, 0].cpu().numpy()
        return float(d[_region(mask, d.shape)].mean())


def load_model(path, device):
    import torch
    m = torch.jit.load(str(path), map_location=device)
    m.eval()
    return m


def run_model(model, rgb: np.ndarray, mask: np.ndarray, device) -> np.ndarray:
    """Same input prep as the app (simple-lama's prepare_img_and_mask): 0..1, symmetric pad to /8, mask 0/1."""
    import torch
    h, w = rgb.shape[:2]
    pad = ((0, -h % 8), (0, -w % 8))
    img = np.pad(rgb, pad + ((0, 0),), mode="symmetric").transpose(2, 0, 1).astype(np.float32) / 255
    msk = (np.pad(mask, pad, mode="symmetric") > 0)[None].astype(np.int64)
    img_t, mask_t = torch.from_numpy(img)[None].to(device), torch.from_numpy(msk)[None].to(device)
    with torch.no_grad():
        out = model(img_t, mask_t)[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(out * 255, 0, 255).astype(np.uint8)[:h, :w]


def contact_sheet(rows, path, height=256):
    """rows: list of (original with mask overlay, baseline, candidate) uint8 RGB images."""
    from PIL import Image
    tiles = []
    for row in rows:
        scaled = [cv2.resize(x, (max(1, round(x.shape[1] * height / x.shape[0])), height), interpolation=cv2.INTER_AREA)
                  for x in row]
        tiles.append(np.concatenate(scaled, axis=1))
    width = max(t.shape[1] for t in tiles)
    sheet = np.concatenate([np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0))) for t in tiles], axis=0)
    Image.fromarray(sheet).save(path, quality=90)


def read_rgb(path, max_side):
    from tools.finetune.curate import read_rgb8
    got = read_rgb8(path)
    if got is None:
        return None
    rgb = got[0]
    s = max_side / max(rgb.shape[:2])
    return cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb


METRICS = ("psnr", "ssim", "lpips")


def evaluate(a):
    import torch
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU available: rerun with --device cpu (slower, same results).")
    files = sorted(p for p in Path(a.data).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No images in {a.data}")
    base, cand, lp = load_model(a.baseline, device), load_model(a.candidate, device), Lpips(device)
    per_image, rows = [], []
    for i, p in enumerate(files):
        rgb = read_rgb(p, a.max_side)
        if rgb is None:
            print(f"skip unreadable {p.name}")
            continue
        mask = generate_mask(rgb.shape[0], rgb.shape[1], a.seed + i, a.regime)
        rec = {"file": p.name, "mask_seed": a.seed + i, "coverage": round(float(mask.mean()) / 255, 4)}
        outs = {"baseline": run_model(base, rgb, mask, device), "candidate": run_model(cand, rgb, mask, device)}
        for who, out in outs.items():
            for fn, key in ((psnr, "psnr"), (ssim, "ssim"), (lp, "lpips")):
                rec[f"{who}_{key}_mask"] = fn(rgb, out, mask)
                rec[f"{who}_{key}_full"] = fn(rgb, out)
        rec["candidate_wins"] = rec["candidate_lpips_mask"] < rec["baseline_lpips_mask"]
        per_image.append(rec)
        if len(rows) < a.contact:
            overlay = rgb.copy()
            overlay[mask > 0] = (0.5 * overlay[mask > 0] + [127, 0, 0]).astype(np.uint8)
            rows.append((overlay, outs["baseline"], outs["candidate"]))
        print(f"{p.name}: LPIPS in-mask base {rec['baseline_lpips_mask']:.4f} cand {rec['candidate_lpips_mask']:.4f}")
    return per_image, rows


def summarize(per_image):
    def mean(k):
        vals = [r[k] for r in per_image if np.isfinite(r[k])]
        return float(np.mean(vals)) if vals else float("inf")
    return {
        who: {f"{m}_{reg}": mean(f"{who}_{m}_{reg}") for m in METRICS for reg in ("mask", "full")}
        for who in ("baseline", "candidate")
    } | {"wins": sum(r["candidate_wins"] for r in per_image)}


def write_reports(out: Path, a, per_image, summary, decision, reasons):
    out.mkdir(parents=True, exist_ok=True)
    report = {"baseline": str(a.baseline), "candidate": str(a.candidate), "data": str(a.data), "seed": a.seed,
              "regime": a.regime, "rule": {"min_lpips_gain": a.min_lpips_gain, "max_psnr_drop": a.max_psnr_drop},
              "summary": summary, "verdict": decision, "reasons": reasons, "images": per_image}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# Fine-tune eval: {Path(a.candidate).name} vs {Path(a.baseline).name}", "",
             f"{len(per_image)} held-out images, masks seed {a.seed} ({a.regime}).", "",
             "| metric | baseline | candidate |", "| --- | --- | --- |"]
    for m in METRICS:
        for reg in ("mask", "full"):
            k = f"{m}_{reg}"
            lines.append(f"| {m.upper()} {'in-mask' if reg == 'mask' else 'whole'} | "
                         f"{summary['baseline'][k]:.4f} | {summary['candidate'][k]:.4f} |")
    lines += ["", f"Candidate wins (lower in-mask LPIPS): {summary['wins']}/{len(per_image)}", "",
              "Contact sheet: `contact_sheet.jpg` (original + mask / baseline / candidate).", ""]
    lines += [f"- {r}" for r in reasons] + ["", f"**Verdict: {decision}**", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--baseline", required=True, help="big-lama.pt")
    ap.add_argument("--candidate", required=True, help="exported model.pt")
    ap.add_argument("--data", required=True, help="held-out photos (curate OUT_DIR/test)")
    ap.add_argument("--out", default="eval_report", help="report folder")
    ap.add_argument("--device", default="cuda", help="cuda or cpu (cpu works, slowly)")
    ap.add_argument("--seed", type=int, default=1000, help="mask i uses seed + i")
    ap.add_argument("--regime", choices=("normal", "small"), default="normal")
    ap.add_argument("--max-side", type=int, default=1024, help="downscale images to this long side")
    ap.add_argument("--contact", type=int, default=6, help="rows in the contact sheet")
    ap.add_argument("--min-lpips-gain", type=float, default=MIN_LPIPS_GAIN)
    ap.add_argument("--max-psnr-drop", type=float, default=MAX_PSNR_DROP)
    a = ap.parse_args(argv)
    per_image, rows = evaluate(a)
    if not per_image:
        raise SystemExit("No readable images evaluated")
    summary = summarize(per_image)
    decision, reasons = verdict(summary, len(per_image), a.min_lpips_gain, a.max_psnr_drop)
    out = Path(a.out)
    write_reports(out, a, per_image, summary, decision, reasons)
    if rows:
        contact_sheet(rows, out / "contact_sheet.jpg")
    print("\n".join(reasons))
    print(decision)


if __name__ == "__main__":
    main()
