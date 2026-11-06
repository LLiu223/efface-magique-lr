"""AC6(d): the default "lama" model gives bit-identical output to pre-registry main.

The golden file was recorded on main (before the model registry) with the same fake
SimpleLama, image, mask and seeds as below. The 128px case forces the LANCZOS
down/up-scale path; on main it was produced by patching CPU_MAX_INFERENCE_DIM to 128.
"""

import os
import sys
import types

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from companion.model_engine import InpaintingEngine, EngineMode

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_013_lama.npz")
SEED = 4242


class GoldenFakeSimpleLama:
    """Deterministic SimpleLama stand-in; pads to a multiple of 8 like the real one."""

    def __init__(self, device=None):
        self.device = device

    def __call__(self, image, mask):
        image = image.convert("RGB")
        mask = mask.convert("L")
        filled = ImageOps.invert(image).filter(ImageFilter.GaussianBlur(3))
        out = Image.composite(filled, image, mask)
        w, h = out.size
        pw, ph = (w + 7) // 8 * 8, (h + 7) // 8 * 8
        padded = Image.new("RGB", (pw, ph), (0, 0, 0))
        padded.paste(out, (0, 0))
        return padded


def install_fake_lama(monkeypatch):
    fake_module = types.ModuleType("simple_lama_inpainting")
    fake_module.SimpleLama = GoldenFakeSimpleLama
    monkeypatch.setitem(sys.modules, "simple_lama_inpainting", fake_module)


def golden_inputs():
    w, h = 256, 192
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(30, 230, w, dtype=np.uint8)
    arr[:, :, 1] = np.linspace(200, 50, h, dtype=np.uint8)[:, None]
    arr[:, :, 2] = 140
    img = Image.fromarray(arr)
    mask = Image.new("L", (w, h), 0)
    mask.paste(255, (90, 60, 160, 120))
    return img, mask


def run_cases():
    """Return {case_key: uint8 array} for every golden case (caller patches max dim)."""
    img, mask = golden_inputs()
    out = {}
    for mode, n in ((EngineMode.FAST, 1), (EngineMode.FIREFLY, 3)):
        engine = InpaintingEngine(device=torch.device("cpu"), mode=mode)
        results = engine.generate_variations(img, mask, num_variations=n, base_seed=SEED)
        for i, r in enumerate(results):
            out[f"{mode.value}_{i}"] = np.asarray(r)
    return out


def _check(cases, prefix):
    golden = np.load(GOLDEN_PATH)
    for key, arr in cases.items():
        assert np.array_equal(arr, golden[f"{prefix}_{key}"]), f"{prefix}_{key} differs from golden"


def test_default_lama_matches_golden(monkeypatch):
    install_fake_lama(monkeypatch)
    _check(run_cases(), "full")


def test_default_lama_matches_golden_with_resize(monkeypatch):
    from companion.models.lama import LamaModel
    install_fake_lama(monkeypatch)
    monkeypatch.setattr(LamaModel, "max_dim_cpu", 128)
    _check(run_cases(), "dim128")
