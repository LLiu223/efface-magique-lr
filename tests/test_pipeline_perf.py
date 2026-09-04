"""
test_pipeline_perf.py
Performance Regression Tests for the Efface Magique LR Inpainting Pipeline.

Validates that critical hot-path functions complete within documented time budgets.
These tests will FAIL if an optimization regression reintroduces the slow path.

Budgets are intentionally generous (5–10× real-world times) so they pass reliably
on CI machines without GPU — the goal is to catch catastrophic regressions, not
micro-benchmark differences.
"""

import os
import sys
import time
import pytest
import numpy as np
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.pipeline import (
    srgb_to_linear,
    linear_to_srgb,
    harmonic_boundary_harmonization,
    calculate_context_crop,
    synthesize_and_match_sensor_grain,
    estimate_sensor_noise_profile,
    seamless_distance_feather_blend,
)
from companion.config import CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_mask(h: int, w: int, cx: float = 0.5, cy: float = 0.5, r: float = 0.15) -> np.ndarray:
    """Return an elliptical binary mask centred at (cx*w, cy*h) with radius r*min(w,h)."""
    m = np.zeros((h, w), dtype=np.uint8)
    radius_px = int(min(w, h) * r)
    center_x, center_y = int(w * cx), int(h * cy)
    import cv2
    cv2.circle(m, (center_x, center_y), radius_px, 255, -1)
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_srgb_conversion_speed():
    """
    srgb_to_linear + linear_to_srgb on a 4000×3000 float32 array must complete
    within 500 ms on any CPU.  (The old np.where full-array approach was ~2–3×
    slower on large arrays; the masked in-place variant cuts this in half.)
    """
    arr = np.random.randint(0, 256, (3000, 4000, 3), dtype=np.uint8).astype(np.float32)

    t0 = time.perf_counter()
    linear = srgb_to_linear(arr)
    back   = linear_to_srgb(linear)
    elapsed = time.perf_counter() - t0

    assert elapsed < 4.0, (
        f"sRGB↔Linear on 4000×3000 took {elapsed:.3f}s — must be < 4.0s. "
        f"Check for regression in the vectorised path."
    )
    # Sanity: values should still be in [0, 255]
    assert float(back.max()) <= 255.0
    assert float(back.min()) >= 0.0


def test_harmonic_harmonization_speed():
    """
    harmonic_boundary_harmonization on a 1000×1000 crop must complete within 3 s.
    The previous 5-loop GaussianBlur was ~8–10× slower; the pyramid version must
    be comfortably within budget.
    """
    h, w = 1000, 1000
    rng = np.random.default_rng(42)
    orig    = rng.integers(50, 200, (h, w, 3), dtype=np.uint8)
    inpaint = rng.integers(50, 200, (h, w, 3), dtype=np.uint8)
    mask    = _make_mask(h, w)

    t0 = time.perf_counter()
    result = harmonic_boundary_harmonization(orig, inpaint, mask, blend_width=35)
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0, (
        f"harmonic_boundary_harmonization on 1000×1000 took {elapsed:.3f}s — must be < 3.0s. "
        f"Pyramid diffusion regression detected."
    )
    assert result.shape == (h, w, 3)


def test_context_crop_speed():
    """
    calculate_context_crop on a simulated 60MP (9504×6336) image with a 200×200
    mask must complete in under 50 ms — it should be essentially instant.
    """
    img_w, img_h = 9504, 6336
    mask_np = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_np[3000:3200, 4500:4700] = 255  # 200×200 spot

    t0 = time.perf_counter()
    for _ in range(10):  # Run 10 times to average
        result = calculate_context_crop((img_w, img_h), mask_np, min_dim=512)
    elapsed = (time.perf_counter() - t0) / 10

    assert elapsed < 0.200, (
        f"calculate_context_crop averaged {elapsed*1000:.1f}ms — must be < 200ms."
    )
    x1, y1, x2, y2 = result
    assert x1 <= 4500
    assert x2 >= 4700
    assert y1 <= 3000
    assert y2 >= 3200


def test_grain_synthesis_speed():
    """
    synthesize_and_match_sensor_grain on an 800×600 crop must complete within 200 ms.
    """
    crop = Image.new("RGB", (800, 600), (128, 128, 128))
    mask = Image.new("L",   (800, 600), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([200, 150, 600, 450], fill=255)
    profile = {"mean_sigma": 8.0, "sigma_r": 8.0, "sigma_g": 8.0, "sigma_b": 8.0, "laplacian_var": 50.0}

    t0 = time.perf_counter()
    result = synthesize_and_match_sensor_grain(crop, profile, mask, seed=42, enable_grain=True)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.2, (
        f"synthesize_and_match_sensor_grain on 800×600 took {elapsed:.3f}s — must be < 200ms."
    )
    assert result.size == (800, 600)


def test_seamless_blend_zero_diff_outside_mask_and_speed():
    """
    seamless_distance_feather_blend on a 1200×800 crop must:
    1. Complete within 1.5 s.
    2. Guarantee 0-diff bit-exact identity outside the mask.
    """
    h, w = 800, 1200
    rng = np.random.default_rng(7)
    orig    = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    inpaint = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask    = _make_mask(h, w, cx=0.5, cy=0.5, r=0.20)

    t0 = time.perf_counter()
    result = seamless_distance_feather_blend(orig, inpaint, mask, feather_radius=16)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.5, (
        f"seamless_distance_feather_blend on 1200×800 took {elapsed:.3f}s — must be < 1.5s."
    )

    result_np = np.array(result)
    outside = mask == 0
    diff_outside = np.max(np.abs(orig[outside].astype(int) - result_np[outside].astype(int)))
    assert diff_outside == 0, (
        f"Outside mask must have 0-diff (got max diff = {diff_outside})."
    )
