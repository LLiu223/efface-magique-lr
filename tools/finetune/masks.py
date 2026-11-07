"""Seeded inpainting masks shared by train (val masks) and eval.

    python -m tools.finetune.masks OUT_DIR --size 512x512 --count 20 --seed 0 --regime normal

generate_mask(h, w, seed, regime) is pure numpy: same arguments, byte-identical mask.
Output format is the LaMa trainer's: single-channel 8-bit PNG, 255 = hole, 0 = keep,
saved next to the image as <stem>_mask000.png (upstream InpaintingDataset pairs them by name).
"""

import argparse
from pathlib import Path

import numpy as np

# Coverage ranges (hole share of the image) per regime.
REGIMES = {
    "normal": (0.05, 0.40),  # strokes, boxes, blobs: object/person removals
    "small": (0.002, 0.03),  # spots, dust, wires
}


def _disc(m, cy, cx, r):
    h, w = m.shape
    y0, y1, x0, x1 = max(0, int(cy - r)), min(h, int(cy + r) + 1), max(0, int(cx - r)), min(w, int(cx + r) + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m[y0:y1, x0:x1] |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _stroke(m, rng, scale, width):
    h, w = m.shape
    y, x = rng.uniform(0, h), rng.uniform(0, w)
    angle = rng.uniform(0, 2 * np.pi)
    for _ in range(int(rng.integers(1, 6))):
        angle += rng.uniform(-1.2, 1.2)
        length = rng.uniform(0.1, 0.4) * scale
        steps = max(1, int(length / max(1.0, width / 2)))
        for _ in range(steps):
            y = float(np.clip(y + np.sin(angle) * length / steps, 0, h - 1))
            x = float(np.clip(x + np.cos(angle) * length / steps, 0, w - 1))
            _disc(m, y, x, width / 2)


def _box(m, rng, scale, lo, hi):
    h, w = m.shape
    bh, bw = int(rng.uniform(lo, hi) * scale) + 1, int(rng.uniform(lo, hi) * scale) + 1
    y, x = int(rng.integers(0, max(1, h - bh))), int(rng.integers(0, max(1, w - bw)))
    m[y:y + bh, x:x + bw] = True


def _blob(m, rng, scale, lo, hi):
    h, w = m.shape
    cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    ry, rx = rng.uniform(lo, hi) * scale + 1, rng.uniform(lo, hi) * scale + 1
    yy, xx = np.ogrid[0:h, 0:w]
    m |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def generate_mask(h: int, w: int, seed: int, regime: str = "normal") -> np.ndarray:
    """HxW uint8 mask (255 = hole) whose coverage lies in REGIMES[regime]."""
    lo, hi = REGIMES[regime]
    rng = np.random.default_rng(seed)
    target = rng.uniform(lo, hi)
    scale = float(min(h, w))
    m = np.zeros((h, w), dtype=bool)
    for _ in range(200):
        if m.mean() >= target:
            break
        trial = m.copy()
        kind = rng.integers(0, 3)
        if regime == "small":
            if kind == 0:
                _stroke(trial, rng, scale * 0.5, max(1.0, rng.uniform(0.003, 0.01) * scale))  # wire
            elif kind == 1:
                _box(trial, rng, scale, 0.01, 0.06)
            else:
                _blob(trial, rng, scale, 0.005, 0.04)  # spot
        elif kind == 0:
            _stroke(trial, rng, scale, rng.uniform(0.03, 0.12) * scale)
        elif kind == 1:
            _box(trial, rng, scale, 0.1, 0.4)
        else:
            _blob(trial, rng, scale, 0.05, 0.25)
        if trial.mean() <= hi:  # reject shapes that would overshoot the regime's maximum
            m = trial
    if m.mean() < lo:  # rare: every shape overshot; fall back to a centred box of exactly lo
        side_h, side_w = max(1, int(round(h * np.sqrt(lo)))), max(1, int(round(w * np.sqrt(lo))))
        y0, x0 = (h - side_h) // 2, (w - side_w) // 2
        m[y0:y0 + side_h, x0:x0 + side_w] = True
    return m.astype(np.uint8) * 255


def mask_filename(stem: str, index: int = 0) -> str:
    return f"{stem}_mask{index:03d}.png"


def write_mask(path, mask: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(mask, mode="L").save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--size", default="512x512", help="WxH")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="mask i uses seed + i")
    ap.add_argument("--regime", choices=sorted(REGIMES), default="normal")
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(a.count):
        m = generate_mask(h, w, a.seed + i, a.regime)
        write_mask(a.out_dir / mask_filename(f"{i:05d}"), m)
        print(f"{i:05d}: coverage {m.mean() / 255:.3f}")


if __name__ == "__main__":
    main()
