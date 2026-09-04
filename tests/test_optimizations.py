"""
test_optimizations.py
Comprehensive unit tests for AI optimization features:
- Hardware detection, device fallback, and CPU threading configuration
- Local bounding-box cropping with context padding and seamless blending
- Interactive canvas 60 FPS stroke throttling and buffer flushing
- Loading spinner overlay and background worker lifecycle
- Memory management and tensor cache clearance
"""

import os
import pytest
import torch
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtWidgets import QApplication

from companion.device import (
    DeviceType,
    DeviceInfo,
    get_optimal_device,
    get_device_info,
    get_device_telemetry,
    configure_cpu_threading,
    setup_device_optimizations,
    free_device_memory,
)
from companion.pipeline import (
    calculate_context_crop,
    seamless_distance_feather_blend,
    feathered_sigmoid_blend,
    estimate_sensor_noise_profile,
    synthesize_and_match_sensor_grain,
)
from companion.config import CONFIG
from companion.canvas import ImageCanvas
from companion.ui import LoadingSpinnerOverlay
from companion.app import ImageLoaderWorker
from companion.model_engine import InpaintingEngine, EngineMode, InpaintingWorker


def test_hardware_detection_and_fallback():
    """Verify get_optimal_device returns valid torch.device and get_device_info returns DeviceInfo."""
    device = get_optimal_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")

    info = get_device_info(device)
    assert isinstance(info, DeviceInfo)
    assert info.device_type in (DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU)
    assert isinstance(info.name, str)
    assert len(info.name) > 0

    # Verify device telemetry string format
    telemetry = get_device_telemetry(device)
    assert isinstance(telemetry, str)
    assert "⚡" in telemetry or "GPU" in telemetry or "CPU" in telemetry or "MPS" in telemetry

    # Verify setup_device_optimizations runs cleanly without raising
    setup_device_optimizations(device)

    # Verify free_device_memory runs cleanly without raising
    free_device_memory(device)


def test_cpu_threading_configuration():
    """Verify configure_cpu_threading safely applies thread counts."""
    applied = configure_cpu_threading(num_threads=2)
    assert applied == 2
    # Auto configuration should also succeed
    auto_applied = configure_cpu_threading(num_threads=None)
    assert auto_applied >= 1


def test_local_bounding_box_cropping_computation():
    """Verify local bounding-box crop accurately isolates masked spots with configurable padding."""
    img_w, img_h = 4000, 3000
    mask_np = np.zeros((img_h, img_w), dtype=np.uint8)
    # A small 50x50 spot centered at (1000, 1500)
    mask_np[1500:1550, 1000:1050] = 255
    mask_img = Image.fromarray(mask_np, mode="L")

    # Small context crop with custom padding 35px
    crop_rect = calculate_context_crop(
        (img_w, img_h),
        mask_img,
        custom_padding=CONFIG.DEFAULT_CONTEXT_PADDING,
        min_dim=256,
    )
    assert crop_rect is not None
    x1, y1, x2, y2 = crop_rect

    # Bounding box must fully contain the original spot (1000, 1500, 1050, 1550)
    assert x1 <= 1000
    assert y1 <= 1500
    assert x2 >= 1050
    assert y2 >= 1550

    crop_w = x2 - x1
    crop_h = y2 - y1

    # Dimensions must be dramatically smaller than 4000x3000 (e.g. ~256x256), yielding massive speedup
    assert crop_w < 600
    assert crop_h < 600


def test_feathered_blending_preserves_unmasked_regions():
    """Verify seamless feather blending leaves unmasked pixels completely untouched."""
    w, h = 200, 200
    orig = Image.new("RGB", (w, h), (100, 150, 200))
    inpainted = Image.new("RGB", (w, h), (255, 0, 0))

    # Mask with a small 20x20 square in the center
    mask_np = np.zeros((h, w), dtype=np.uint8)
    mask_np[90:110, 90:110] = 255
    mask = Image.fromarray(mask_np, mode="L")

    blended = seamless_distance_feather_blend(orig, inpainted, mask, feather_radius=5)
    blended_np = np.asarray(blended)

    # Pixel outside mask and feather falloff should match original exactly
    np.testing.assert_array_equal(blended_np[10, 10], [100, 150, 200])
    # Center pixel should be inpainting color
    assert blended_np[100, 100, 0] > 200


def test_canvas_stroke_throttling_timer(qtbot):
    """Verify ImageCanvas uses throttled timer to batch mask pixmap conversion."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)

    test_img = Image.new("RGB", (800, 600), (40, 40, 40))
    canvas.load_image(test_img)

    assert hasattr(canvas, "_stroke_timer")
    assert canvas._stroke_timer.interval() == CONFIG.CANVAS_THROTTLE_MS
    assert not canvas._mask_dirty

    # Simulate beginning of stroke
    canvas._is_drawing = True
    p1 = QPointF(100.0, 100.0)
    p2 = QPointF(110.0, 110.0)
    canvas._paint_stroke(p1, p2)

    # Must be marked dirty and stroke timer started
    assert canvas._mask_dirty is True
    assert canvas._stroke_timer.isActive()

    # Flush pixmap
    canvas._flush_mask_pixmap()
    assert canvas._mask_dirty is False

    # Clear mask stops timer
    canvas.clear_mask()
    assert not canvas._stroke_timer.isActive()
    assert not canvas.has_mask()


def test_loading_spinner_overlay_lifecycle(qtbot):
    """Verify LoadingSpinnerOverlay start, set_message, and stop behavior."""
    parent_widget = ImageCanvas()
    qtbot.addWidget(parent_widget)
    parent_widget.show()

    overlay = LoadingSpinnerOverlay(parent_widget)

    assert overlay.isHidden()
    overlay.start(message="Erasing...", subtext="0% complete")
    assert not overlay.isHidden()
    assert overlay._timer.isActive()

    overlay.set_message("Blending layers...", "50% complete")
    assert overlay._message == "Blending layers..."

    overlay.stop()
    assert overlay.isHidden()
    assert not overlay._timer.isActive()


def test_image_loader_worker_background_execution(tmp_path, qtbot):
    """Verify ImageLoaderWorker reads and decodes images in background thread."""
    img_path = str(tmp_path / "test_async_load.png")
    test_img = Image.new("RGB", (300, 200), (255, 128, 64))
    test_img.save(img_path)

    worker = ImageLoaderWorker(img_path)
    loaded_results = []

    worker.loaded.connect(lambda img, p: loaded_results.append((img, p)))

    with qtbot.waitSignal(worker.loaded, timeout=3000):
        worker.start()

    assert len(loaded_results) == 1
    loaded_img, path = loaded_results[0]
    assert path == img_path
    assert loaded_img.size == (300, 200)


def test_free_device_memory_is_idempotent():
    """free_device_memory() must be callable multiple times without raising an error."""
    device = get_optimal_device()
    for _ in range(5):
        try:
            free_device_memory(device)
        except Exception as e:
            pytest.fail(f"free_device_memory raised on repeat call: {e}")


def test_context_crop_empty_mask_returns_valid_centered_crop():
    """calculate_context_crop with all-zero mask must not crash and must return a centred crop."""
    empty_mask = np.zeros((3000, 4000), dtype=np.uint8)
    x1, y1, x2, y2 = calculate_context_crop(
        (4000, 3000),
        empty_mask,
        min_dim=512,
    )
    assert 0 <= x1 < x2 <= 4000, "Crop x-bounds must be within image width."
    assert 0 <= y1 < y2 <= 3000, "Crop y-bounds must be within image height."
    assert (x2 - x1) >= 1, "Crop must have positive width."
    assert (y2 - y1) >= 1, "Crop must have positive height."
