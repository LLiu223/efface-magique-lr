"""Curate a folder of photos for fine-tuning: drop bad images, split 80/10/10, write manifest.json.

    python -m tools.finetune.curate INPUT_DIR OUT_DIR [--dry-run]

Reads JPEG/PNG/TIFF. Drops (with a reason) unreadable, tiny, blurry, clipped and near-duplicate images,
in that order; near-duplicates are removed before the split so copies never span train and test.
Writes OUT_DIR/{train,val,test}/ (copies, or symlinks with --link symlink) and OUT_DIR/manifest.json.
16-bit TIFFs are converted to 8-bit sRGB PNG in OUT_DIR only. Originals are only ever read.
"""

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MIN_RECOMMENDED = 200
SPLITS = ("train", "val", "test")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_srgb(rgb8: np.ndarray, path) -> np.ndarray:
    """Convert via the embedded ICC profile when there is one (else assume sRGB already)."""
    try:
        from PIL import Image, ImageCms
        with Image.open(path) as im:
            icc = im.info.get("icc_profile")
        if not icc:
            return rgb8
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        out = ImageCms.profileToProfile(Image.fromarray(rgb8), src, ImageCms.createProfile("sRGB"), outputMode="RGB")
        return np.asarray(out)
    except Exception:  # Pillow cannot open some 16-bit TIFFs; keep the pixels as read
        return rgb8


def read_rgb8(path):
    """(rgb uint8 HxWx3, was_16bit) or None if unreadable."""
    try:
        arr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    is16 = arr.dtype == np.uint16
    if is16:
        arr = ((arr.astype(np.uint32) + 128) // 257).astype(np.uint8)
    elif arr.dtype != np.uint8:
        return None
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return (_to_srgb(rgb, path) if is16 else rgb), is16


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian on a copy scaled to 1024 px long side (comparable across sizes)."""
    s = 1024 / max(gray.shape)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clip_shares(gray: np.ndarray):
    return float((gray >= 250).mean()), float((gray <= 5).mean())


def phash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def split_names(names, seed, fractions=(0.8, 0.1, 0.1)):
    """Deterministic split by image: sorted, shuffled with seed, cut into train/val/test."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    n = len(order)
    n_val, n_test = round(n * fractions[1]), round(n * fractions[2])
    n_train = n - n_val - n_test
    return {"train": order[:n_train], "val": order[n_train:n_train + n_val], "test": order[n_train + n_val:]}


def _overlaps(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def score_image(path: Path, args):
    """Return (record, reason or None). record holds the scores."""
    rec = {"sha256": sha256_file(path), "scores": {}}
    got = read_rgb8(path)
    if got is None:
        return rec, "unreadable"
    rgb, is16 = got
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hi, lo = clip_shares(gray)
    rec["scores"] = {"width": w, "height": h, "blur": round(blur_score(gray), 2),
                     "highlight_share": round(hi, 4), "shadow_share": round(lo, 4), "phash": f"{phash(gray):016x}"}
    rec["bit_depth"] = 16 if is16 else 8
    if min(h, w) < args.min_side:
        return rec, f"tiny: min side {min(h, w)} < {args.min_side}"
    if rec["scores"]["blur"] < args.blur:
        return rec, f"blurry: Laplacian variance {rec['scores']['blur']} < {args.blur}"
    if hi > args.max_highlight:
        return rec, f"clipped highlights: {hi:.3f} > {args.max_highlight}"
    if lo > args.max_shadow:
        return rec, f"clipped shadows: {lo:.3f} > {args.max_shadow}"
    return rec, None


def curate(args) -> dict:
    inp, out = Path(args.input_dir), Path(args.out_dir)
    if not inp.is_dir():
        raise SystemExit(f"INPUT_DIR {inp} is not a directory")
    if _overlaps(inp, out):
        raise SystemExit(f"Refusing: OUT_DIR {out} overlaps INPUT_DIR {inp}")
    if not args.dry_run and out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing: OUT_DIR {out} is not empty")

    files = sorted(p for p in inp.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    images, kept_hashes = [], []  # kept_hashes: (phash int, file) of kept images so far
    for p in files:
        rel = p.relative_to(inp).as_posix()
        rec, reason = score_image(p, args)
        if reason is None:
            ph = int(rec["scores"]["phash"], 16)
            dup = next((f for h, f in kept_hashes if hamming(h, ph) <= args.dup_distance), None)
            if dup is not None:
                reason = f"near-duplicate of {dup} (pHash distance <= {args.dup_distance})"
            else:
                kept_hashes.append((ph, rel))
        images.append({"file": rel, **rec, "status": "dropped" if reason else "kept", "reason": reason or "ok"})

    kept = [r for r in images if r["status"] == "kept"]
    split = split_names([r["file"] for r in kept], args.seed)
    where = {f: s for s, fs in split.items() for f in fs}
    for r in kept:
        r["split"] = where[r["file"]]
        ext = ".png" if r.get("bit_depth") == 16 else Path(r["file"]).suffix.lower()
        r["out"] = f"{r['split']}/{r['sha256'][:12]}_{Path(r['file']).stem}{ext}"

    manifest = {
        "version": 1, "input_dir": str(inp.resolve()), "seed": args.seed,
        "thresholds": {k: getattr(args, k) for k in ("min_side", "blur", "max_highlight", "max_shadow", "dup_distance")},
        "counts": {"scanned": len(images), "kept": len(kept), **{s: len(v) for s, v in split.items()}},
        "images": images,
    }

    for r in images:
        print(f"{r['status']:8} {r['file']}: {r['reason']}")
    print(f"kept {len(kept)}/{len(images)}: " + ", ".join(f"{s} {len(split[s])}" for s in SPLITS))
    if len(kept) < MIN_RECOMMENDED:
        print(f"WARNING: only {len(kept)} images kept; fewer than {MIN_RECOMMENDED} will likely overfit.", file=sys.stderr)
    if args.dry_run:
        print("dry run: nothing written")
        return manifest

    for s in SPLITS:
        (out / s).mkdir(parents=True, exist_ok=True)
    for r in kept:
        src, dst = inp / r["file"], out / r["out"]
        if r.get("bit_depth") == 16:
            cv2.imencode(".png", cv2.cvtColor(read_rgb8(src)[0], cv2.COLOR_RGB2BGR))[1].tofile(str(dst))
        elif args.link == "symlink":
            os.symlink(src.resolve(), dst)
        else:
            shutil.copyfile(src, dst)  # copyfile: reads src only, never touches its metadata
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--min-side", type=int, default=512, help="drop if the shorter side is below this (px)")
    ap.add_argument("--blur", type=float, default=50.0, help="drop if Laplacian variance (at 1024 px) is below this")
    ap.add_argument("--max-highlight", type=float, default=0.10, help="drop if this share of pixels is >= 250")
    ap.add_argument("--max-shadow", type=float, default=0.25, help="drop if this share of pixels is <= 5")
    ap.add_argument("--dup-distance", type=int, default=6, help="near-duplicate if pHash Hamming distance <= this")
    ap.add_argument("--seed", type=int, default=0, help="split seed")
    ap.add_argument("--link", choices=("copy", "symlink"), default="copy", help="how kept 8-bit files are written")
    ap.add_argument("--dry-run", action="store_true", help="print the report, write nothing")
    return ap


def main(argv=None):
    curate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
