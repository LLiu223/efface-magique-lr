"""Old-vs-new byte-identity checks for the job 009 hotspot fixes."""
from types import SimpleNamespace

import os
import sys

import cv2
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import companion.model_engine as model_engine  # noqa: E402
from companion.model_engine import EngineMode, InpaintingEngine  # noqa: E402
from companion.pipeline import dilate_mask_for_contact_shadows  # noqa: E402


def _old_dilate(mask, radius=12):
    """dilate_mask_for_contact_shadows before job 009 (full-frame cv2.dilate)."""
    if isinstance(mask, Image.Image):
        mask_arr = np.array(mask.convert("L"))
    else:
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
    binary = (mask_arr > 10).astype(np.uint8) * 255
    if np.count_nonzero(binary) == 0:
        return mask_arr
    r = max(4, min(24, int(radius)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
    return cv2.dilate(binary, kernel)


def _masks():
    rng = np.random.default_rng(9)
    h, w = 240, 320
    yield np.zeros((h, w), np.uint8)                                   # empty
    yield (rng.random((h, w)) > 0.995).astype(np.uint8) * 255          # sparse dots everywhere
    edge = np.zeros((h, w), np.uint8)
    edge[0:5, 0:7] = 200                                                # corner
    edge[h - 3:, w // 2:w // 2 + 4] = 11                                # bottom edge, just over threshold
    edge[100:110, w - 2:] = 255                                         # right edge
    yield edge
    low = np.zeros((h, w), np.uint8)
    low[50:60, 50:60] = 10                                              # below threshold only
    yield low
    small = np.zeros((h, w), np.uint8)
    small[120, 160] = 255                                               # single pixel
    yield small
    yield np.dstack([edge, edge, edge])                                 # 3-channel ndarray input


@pytest.mark.parametrize("radius", [1, 8, 12, 16, 24, 40])
def test_bbox_dilate_matches_full_frame(radius):
    for m in _masks():
        assert np.array_equal(dilate_mask_for_contact_shadows(m, radius), _old_dilate(m, radius))
        if m.ndim == 2:
            for mode in ("L", "RGB"):
                pil = Image.fromarray(m).convert(mode)
                new = dilate_mask_for_contact_shadows(pil, radius)
                old = _old_dilate(pil, radius)
                assert new.dtype == old.dtype and np.array_equal(new, old)


def _engine(mode):
    e = InpaintingEngine(device=torch.device("cpu"), mode=mode)
    e._is_loaded = True
    if mode == EngineMode.FAST:  # deterministic stand-in for LaMa
        e._model = SimpleNamespace(inpaint=lambda i, mk, seed, prompt: Image.fromarray(
            cv2.inpaint(np.asarray(i), np.asarray(mk), 3, cv2.INPAINT_TELEA)))
    return e


def _inputs():
    rng = np.random.default_rng(7)
    img = Image.fromarray(rng.integers(0, 256, (300, 450, 3), dtype=np.uint8))
    m = Image.new("L", (450, 300), 0)
    ImageDraw.Draw(m).ellipse((400, 5, 449, 60), fill=255)  # touches the right edge
    ImageDraw.Draw(m).rectangle((60, 150, 100, 180), fill=200)
    return img, m


@pytest.mark.parametrize("mode", [EngineMode.FIREFLY, EngineMode.FAST])
def test_generate_variations_unchanged_by_bbox_dilate(monkeypatch, mode):
    img, m = _inputs()
    new = _engine(mode).generate_variations(img, m, num_variations=2, base_seed=42)
    monkeypatch.setattr(model_engine, "dilate_mask_for_contact_shadows", _old_dilate)
    old = _engine(mode).generate_variations(img, m, num_variations=2, base_seed=42)
    for a, b in zip(new, old):
        assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("mode", [EngineMode.FIREFLY, EngineMode.FAST])
def test_rgb_base_not_copied_twice_same_output_no_aliasing(mode):
    img, m = _inputs()
    before = np.array(img)
    # RGB input takes the new no-convert path; RGBA (alpha 255) still takes the old convert("RGB") path.
    new = _engine(mode).generate_variations(img, m, num_variations=3, base_seed=42)
    old = _engine(mode).generate_variations(img.convert("RGBA"), m, num_variations=3, base_seed=42)
    for a, b in zip(new, old):
        assert a.mode == b.mode == "RGB"
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # The caller's image is untouched and every variation owns its buffer.
    assert np.array_equal(np.asarray(img), before)
    assert all(v is not img for v in new)
    new[0].paste((0, 0, 0), (0, 0, 450, 300))
    assert np.array_equal(np.asarray(img), before)
    assert not np.array_equal(np.asarray(new[1]), np.asarray(new[0]))
