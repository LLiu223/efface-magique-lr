"""16-bit TIFF round trip: lossless untouched pixels, uint16 output, ICC/DPI kept, atomic write."""

import os
import sys
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image
from PyQt6.QtWidgets import QMessageBox

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.app import MainWindow
from companion.utils import tiff16
from companion.utils.tiff16 import read_rgb16, write_tiff16

H, W = 150, 333  # not multiples of the strip height
ICC = bytes(range(256)) * 3 + b"odd"  # odd length exercises word alignment
PAD = 24
BOX = (100, 40, 180, 90)  # x0, y0, x1, y1 of the painted mask


def _read16(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)[:, :, ::-1]


def _noisy16():
    rng = np.random.default_rng(1)
    ramp = np.linspace(0, 60000, W, dtype=np.float32)[None, :, None]
    noise = rng.integers(0, 4000, (H, W, 3))
    return np.clip(ramp + noise, 0, 65535).astype(np.uint16)


def _smooth16():
    # 333 columns spanning 1000..61000: every value distinct, many per 8-bit level
    row = np.linspace(1000, 61000, W).astype(np.uint16)
    arr = np.empty((H, W, 3), np.uint16)
    arr[..., 0], arr[..., 1], arr[..., 2] = row, row[::-1], 30000 + np.arange(H, dtype=np.uint16)[:, None]
    return arr


def _erase_and_save(qtbot, arr16, tmp_path, name="in.tif"):
    src = str(tmp_path / name)
    write_tiff16(src, arr16, ICC, (300.0, 300.0))
    out = str(tmp_path / "out.tif")
    window = MainWindow(input_path=src, output_path=out)
    qtbot.addWidget(window)
    assert window.base_image16 is not None and np.array_equal(window.base_image16, arr16)

    # Stubbed variation: changes pixels only inside the mask bbox plus padding, like the blender
    mask = Image.new("L", (W, H), 0)
    mask.paste(255, BOX)
    x0, y0, x1, y1 = BOX
    var = np.asarray(window.base_image).copy()
    var[y0 - PAD:y1 + PAD, x0 - PAD:x1 + PAD] = (var[y0 - PAD:y1 + PAD, x0 - PAD:x1 + PAD] // 2) + 7
    window.last_used_mask = mask
    window._on_inpainting_finished([Image.fromarray(var)])
    assert window.save_image_to_disk() == out
    return window, out, var


def _outside(a):
    x0, y0, x1, y1 = BOX
    keep = np.ones(a.shape[:2], bool)
    keep[y0 - PAD:y1 + PAD, x0 - PAD:x1 + PAD] = False
    return a[keep]


def test_untouched_pixels_bit_identical_after_erase(qtbot, tmp_path):
    arr16 = _noisy16()
    window, out, var = _erase_and_save(qtbot, arr16, tmp_path)
    saved = _read16(out)
    assert saved.dtype == np.uint16 and saved.shape == arr16.shape
    assert np.array_equal(_outside(saved), _outside(arr16))
    # Inside, the documented rule holds: orig16 + (out8 - base8) * 257, clipped
    base8 = np.asarray(window.base_image).astype(np.int32)
    expected = np.clip(arr16.astype(np.int32) + (var.astype(np.int32) - base8) * 257, 0, 65535)
    assert np.array_equal(saved, expected)
    assert not np.array_equal(saved, arr16)


def test_icc_and_dpi_read_back_through_pillow(qtbot, tmp_path):
    _, out, _ = _erase_and_save(qtbot, _noisy16(), tmp_path)
    with Image.open(out) as im:
        assert im.info["icc_profile"] == ICC
        assert tuple(round(v) for v in im.info["dpi"]) == (300, 300)
        assert im.tag_v2[258] == (16, 16, 16)
        assert im.size == (W, H)
        im.load()


def test_smooth_gradient_keeps_unique_values(qtbot, tmp_path):
    arr16 = _smooth16()
    _, out, _ = _erase_and_save(qtbot, arr16, tmp_path)
    saved = _read16(out)
    before, after = _outside(arr16), _outside(saved)
    assert len(np.unique(before[:, 0])) > 256
    assert len(np.unique(after[:, 0])) == len(np.unique(before[:, 0]))


def test_8bit_input_still_saves_8bit(qtbot, tmp_path):
    src = str(tmp_path / "in8.tif")
    Image.fromarray((_noisy16() >> 8).astype(np.uint8)).save(src, icc_profile=ICC)
    out = str(tmp_path / "out8.tif")
    window = MainWindow(input_path=src, output_path=out)
    qtbot.addWidget(window)
    assert window.base_image16 is None
    assert window.save_image_to_disk() == out
    assert cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_UNCHANGED).dtype == np.uint8


def test_16bit_input_to_png_stays_8bit(qtbot, tmp_path):
    src = str(tmp_path / "in.tif")
    write_tiff16(src, _noisy16())
    out = str(tmp_path / "out.png")
    window = MainWindow(input_path=src, output_path=out)
    qtbot.addWidget(window)
    assert window.base_image16 is not None
    assert window.save_image_to_disk() == out
    assert cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_UNCHANGED).dtype == np.uint8


def test_writer_roundtrip_non_ascii_path_and_cv2_written_input(tmp_path):
    arr = np.random.default_rng(2).integers(0, 65536, (130, 67, 3)).astype(np.uint16)
    p = str(tmp_path / "été_写真.tif")
    write_tiff16(p, arr)
    assert np.array_equal(read_rgb16(p), arr)
    # cv2-written 16-bit TIFF (different writer) also decodes losslessly, RGBA alpha dropped
    q = str(tmp_path / "cv.tif")
    rgba = np.dstack([arr, np.full(arr.shape[:2], 9, np.uint16)])
    cv2.imwrite(q, rgba[:, :, [2, 1, 0, 3]])
    assert np.array_equal(read_rgb16(q), arr)


def test_failed_16bit_write_leaves_no_file(qtbot, tmp_path, monkeypatch):
    src = str(tmp_path / "in.tif")
    write_tiff16(src, _noisy16())
    out = str(tmp_path / "out.tif")
    window = MainWindow(input_path=src, output_path=out)
    qtbot.addWidget(window)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(tiff16.zlib, "compress", boom)
    assert window.save_image_to_disk() is None
    assert errors
    assert sorted(os.listdir(tmp_path)) == ["in.tif"]


def test_failed_overwrite_keeps_previous_file(tmp_path, monkeypatch):
    p = str(tmp_path / "x.tif")
    arr = _noisy16()
    write_tiff16(p, arr)
    monkeypatch.setattr(tiff16.zlib, "compress", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        write_tiff16(p, arr // 2)
    monkeypatch.setattr(tiff16.zlib, "compress", zlib.compress)
    assert np.array_equal(read_rgb16(p), arr)
    assert os.listdir(tmp_path) == ["x.tif"]
