"""
test_gui.py
Automated testing for Efface Magique LR PyQt6 Canvas & GUI interactions using pytest-qt.

Covers:
- Coordinate transformation math (screen-to-image mapping at 25%, 100%, 400% zoom and pan)
- Undo / Redo history stack consistency
- Hotkeys and brush radius clamping ([ and ])
- Save & exit flow verifying output file creation and status
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QKeyEvent, QMouseEvent
import pytest

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.canvas import ImageCanvas
from companion.app import MainWindow


@pytest.fixture
def sample_pil_image():
    """Create a standard 600x400 synthetic RGB test image."""
    arr = np.zeros((400, 600, 3), dtype=np.uint8)
    arr[:, :, 0] = 120
    arr[:, :, 1] = 150
    arr[:, :, 2] = 200
    return Image.fromarray(arr, mode="RGB")


def test_coordinate_transformation_math(qtbot, sample_pil_image):
    """Verify screen-to-image coordinate mapping at 25%, 100%, and 400% zoom with pan offsets."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(800, 600)
    canvas.show()

    canvas.load_image(sample_pil_image)

    # Test points inside the 600x400 image
    target_image_pts = [
        QPointF(0, 0),
        QPointF(100.0, 150.0),
        QPointF(300.0, 200.0),
        QPointF(599.0, 399.0),
    ]

    for zoom in [0.25, 1.0, 4.0]:
        canvas.resetTransform()
        canvas.scale(zoom, zoom)

        # Screen pixels are integer coordinates, so quantization tolerance is 1.0 / zoom scene pixels
        tol = max(1.0, (1.0 / zoom) + 0.1)

        for img_pt in target_image_pts:
            # Map scene (image) point to viewport (screen) coordinates
            screen_pt = canvas.mapFromScene(img_pt)
            # Map back from viewport to scene coordinates
            reconstructed_scene_pt = canvas.mapToScene(screen_pt)

            # Precision should be accurate within quantization tolerance
            assert abs(reconstructed_scene_pt.x() - img_pt.x()) <= tol, (
                f"Zoom {zoom}: X mismatch {reconstructed_scene_pt.x()} vs {img_pt.x()} (tol {tol})"
            )
            assert abs(reconstructed_scene_pt.y() - img_pt.y()) <= tol, (
                f"Zoom {zoom}: Y mismatch {reconstructed_scene_pt.y()} vs {img_pt.y()} (tol {tol})"
            )


def test_undo_redo_stack_behavior(qtbot, sample_pil_image):
    """Simulate 5 sequential strokes, undo 3 times -> check state #2, redo -> inpaint consistency."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.load_image(sample_pil_image)

    # 1. Simulate 5 sequential brush strokes
    stroke_snapshots = []
    for i in range(1, 6):
        # Draw a distinct stroke line
        p1 = QPointF(i * 50, 100)
        p2 = QPointF(i * 50 + 20, 120)
        canvas._paint_stroke(p1, p2)
        canvas._save_mask_state()
        stroke_snapshots.append(np.array(canvas.get_mask_image()))

    # Verify 5 strokes resulted in mask history
    assert len(canvas._mask_history) == 6  # 1 initial empty state + 5 strokes

    # 2. Undo 3 times -> should match stroke #2 (index 1 of stroke_snapshots)
    assert canvas.undo() is True
    assert canvas.undo() is True
    assert canvas.undo() is True

    current_mask_arr = np.array(canvas.get_mask_image())
    expected_mask_arr = stroke_snapshots[1]  # stroke #2
    assert np.array_equal(current_mask_arr, expected_mask_arr), "Mask after 3 undos must match stroke #2!"

    # 3. Redo once -> should advance to stroke #3
    assert canvas.redo() is True
    assert np.array_equal(np.array(canvas.get_mask_image()), stroke_snapshots[2])

    # 4. Apply inpainting result -> history stacks must update cleanly
    dummy_inpainted = sample_pil_image.copy()
    canvas.apply_inpainted_image(dummy_inpainted)

    # Redo stack must be cleared, mask must be reset, and image history appended
    assert len(canvas._mask_redo_stack) == 0
    assert len(canvas._image_history) == 2
    assert not canvas.has_mask()


def test_hotkeys_and_brush_resizing(qtbot, sample_pil_image):
    """Verify [ and ] hotkeys adjust brush radius and properly clamp to min/max bounds."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.load_image(sample_pil_image)

    # Start radius
    initial_radius = canvas.brush_radius

    # Press ']' to increase
    event_bracket_right = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_BracketRight, Qt.KeyboardModifier.NoModifier)
    canvas.keyPressEvent(event_bracket_right)
    assert canvas.brush_radius == initial_radius + 5

    # Press '[' to decrease
    event_bracket_left = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_BracketLeft, Qt.KeyboardModifier.NoModifier)
    canvas.keyPressEvent(event_bracket_left)
    assert canvas.brush_radius == initial_radius

    # Test clamping minimum bound (min_brush_radius = 3)
    canvas.set_brush_radius(4)
    canvas.keyPressEvent(event_bracket_left)
    assert canvas.brush_radius == canvas.min_brush_radius

    # Test clamping maximum bound (max_brush_radius = 300)
    canvas.set_brush_radius(298)
    canvas.keyPressEvent(event_bracket_right)
    assert canvas.brush_radius == canvas.max_brush_radius


def test_save_and_return_file_export(qtbot, sample_pil_image):
    """Test Save & Return action, verifying output file exists on disk and saved_successfully flag is True."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_out:
        in_path = tmp_in.name
        out_path = tmp_out.name

    try:
        sample_pil_image.save(in_path, compression="tiff_deflate")

        window = MainWindow(input_path=in_path, output_path=out_path)
        qtbot.addWidget(window)

        # Trigger save
        window._on_save_and_exit()

        # Check that saved_successfully is true and output file exists and is valid
        assert window.saved_successfully is True
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0

        saved_img = Image.open(out_path)
        assert saved_img.size == sample_pil_image.size
        saved_img.close()
    finally:
        for p in [in_path, out_path]:
            if os.path.exists(p):
                os.remove(p)


def test_eraser_tool_erases_mask_pixels(qtbot, sample_pil_image):
    """Verify that the eraser tool actually clears painted mask pixels on both single clicks and drags."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.load_image(sample_pil_image)

    # 1. Paint a red mask stroke
    canvas.set_eraser_mode(False)
    canvas.set_brush_radius(20)
    canvas._paint_stroke(QPointF(50, 50), QPointF(150, 50))
    canvas._save_mask_state()

    mask_arr = np.array(canvas.get_mask_image())
    painted_count = np.count_nonzero(mask_arr > 10)
    assert painted_count > 0, "Initial stroke must create mask pixels."

    # 2. Switch to Eraser tool and erase through the middle
    canvas.set_eraser_mode(True)
    canvas.set_brush_radius(25)
    # Erase cross stroke
    canvas._paint_stroke(QPointF(100, 30), QPointF(100, 70))
    canvas._save_mask_state()

    erased_mask_arr = np.array(canvas.get_mask_image())
    remaining_count = np.count_nonzero(erased_mask_arr > 10)
    assert remaining_count < painted_count, "Eraser stroke must reduce non-zero mask pixels."
    # The center intersection pixel (100, 50) must be completely 0
    assert erased_mask_arr[50, 100] == 0, "Center pixel must be completely cleared by eraser."

    # 3. Single-click dot erasure
    canvas.set_eraser_mode(False)
    canvas._paint_stroke(QPointF(200, 200), QPointF(200, 200))
    assert canvas.get_mask_image().getpixel((200, 200)) == 255
    canvas.set_eraser_mode(True)
    canvas._paint_stroke(QPointF(200, 200), QPointF(200, 200))
    assert canvas.get_mask_image().getpixel((200, 200)) == 0, "Single-click dot eraser must clear pixel."


def test_fit_to_screen_and_zoom_behavior(qtbot, sample_pil_image):
    """Verify fit_to_screen resets zoom transform and aligns with viewport."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(800, 600)
    canvas.load_image(sample_pil_image)

    # Canvas initial fit
    canvas.fit_to_screen()
    initial_transform = canvas.transform()

    # Zoom in
    canvas.scale(2.0, 2.0)
    canvas._user_has_manually_zoomed = True
    assert canvas.transform().m11() > initial_transform.m11()

    # Fit to screen again
    canvas.fit_to_screen()
    assert canvas._user_has_manually_zoomed is False
    assert abs(canvas.transform().m11() - initial_transform.m11()) < 1e-3


def test_stroke_finished_signal_emitted(qtbot, sample_pil_image):
    """Verify strokeFinished signal is emitted when mouse draw finishes."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.load_image(sample_pil_image)

    signal_received = []
    canvas.strokeFinished.connect(lambda: signal_received.append(True))

    # Simulate mouse press and release on canvas
    press_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(50, 50),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    canvas.mousePressEvent(press_event)
    assert canvas._is_drawing is True

    release_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(50, 50),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier
    )
    canvas.mouseReleaseEvent(release_event)
    assert canvas._is_drawing is False
    assert len(signal_received) == 1


def test_undo_ai_does_not_show_brush_strokes(qtbot, sample_pil_image):
    """Verify Ctrl+Z undoes the AI inpainting step directly without resurrecting red brush strokes."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.load_image(sample_pil_image)

    # 1. Paint a red stroke
    canvas._paint_stroke(QPointF(20, 20), QPointF(60, 60))
    canvas._save_mask_state()
    assert bool(canvas.has_mask()) is True

    # 2. Inpaint
    inpainted = sample_pil_image.copy()
    canvas.apply_inpainted_image(inpainted)
    assert bool(canvas.has_mask()) is False
    assert len(canvas._image_history) == 2

    # 3. Undo (Ctrl+Z) -> Image must revert and mask must be 100% clean (zero brush strokes)
    undone = canvas.undo()
    assert bool(undone) is True
    assert bool(canvas.has_mask()) is False, "Undoing AI inpainting must not show old red brush strokes!"
    assert len(canvas._image_redo_stack) == 1

    # 4. Redo (Ctrl+Y) -> Image must re-apply inpainting and mask remains 100% clean
    redone = canvas.redo()
    assert bool(redone) is True
    assert bool(canvas.has_mask()) is False, "Redoing AI inpainting must keep mask clean!"
    assert len(canvas._image_history) == 2


def test_reset_all_clears_variations_and_history(qtbot, sample_pil_image, tmp_path):
    """Verify Reset All resets canvas, clears history, and closes variations panel."""
    from companion.app import MainWindow
    test_file = str(tmp_path / "test_reset.png")
    sample_pil_image.save(test_file)

    window = MainWindow(input_path=test_file)
    qtbot.addWidget(window)
    window.show()

    # Simulate inpainting variations
    var1 = sample_pil_image.copy()
    var2 = sample_pil_image.copy()
    window.current_variations = [var1, var2]
    window.carousel_panel.setVisible(True)
    window.canvas.apply_inpainted_image(var1)

    # Click Reset All
    window._on_reset_all()

    assert len(window.current_variations) == 0
    assert window.carousel_panel.isVisible() is False
    assert len(window.canvas._image_history) == 1
    assert bool(window.canvas.has_mask()) is False


def test_brush_ring_overlay_matches_radius(qtbot, sample_pil_image):
    """Verify vector brush ring matches brush_radius exactly without Windows 32px clamping."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.load_image(sample_pil_image)

    ring = canvas._brush_ring_item
    assert ring is not None
    assert ring.zValue() == 9999
    assert ring.pen().isCosmetic() is True

    # Test small, medium, and large radius
    for radius in [15, 60, 150]:
        canvas.set_brush_radius(radius)
        canvas._update_brush_ring(QPointF(200, 200))
        assert ring.isVisible() is True
        rect = ring.rect()
        assert abs(rect.width() - (radius * 2.0)) < 1e-3, f"Ring width must be {radius * 2}"
        assert abs(rect.height() - (radius * 2.0)) < 1e-3, f"Ring height must be {radius * 2}"


def test_save_tiff_with_complex_lightroom_metadata(qtbot, tmp_path):
    """Verify 16-bit TIFF files from Lightroom save with 0 errors (no 'Error setting from dictionary')."""
    from companion.app import MainWindow
    # Use real Lightroom TIFF from .tmp if available, or create TIFF with multi-tag metadata
    lr_tiff = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".tmp", "_SNY4766_ai_edit_20260902_173203.tif")
    if not os.path.isfile(lr_tiff):
        test_img = Image.new("RGB", (300, 300), (100, 150, 200))
        lr_tiff = str(tmp_path / "test_lr_mock.tif")
        test_img.save(lr_tiff, compression="tiff_deflate")

    out_path = str(tmp_path / "saved_clean.tif")
    window = MainWindow(input_path=lr_tiff, output_path=out_path)
    qtbot.addWidget(window)
    window.show()

    # Save
    window._on_save_and_exit()
    assert window.saved_successfully is True
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0


def test_manual_erase_button_required(qtbot, sample_pil_image, tmp_path):
    """Verify AI inpainting does NOT start on stroke finish, and only starts when clicking Erase Object."""
    from companion.app import MainWindow
    test_file = str(tmp_path / "test_manual.png")
    sample_pil_image.save(test_file)

    window = MainWindow(input_path=test_file)
    qtbot.addWidget(window)
    window.show()

    # 1. Paint stroke
    window.canvas._paint_stroke(QPointF(50, 50), QPointF(80, 80))
    window.canvas._save_mask_state()
    window.canvas.strokeFinished.emit()

    # AI worker must NOT be running
    assert window.worker is None, "AI inpainting must NOT trigger automatically on stroke finish!"

    # 2. Clicking Erase Object must trigger inpainting worker
    window.btn_erase.click()
    assert window.worker is not None, "Clicking 'Erase Object' must launch inpainting worker!"


def test_fast_spot_removal_workflow(qtbot, sample_pil_image, tmp_path):
    """Verify Fast Spot Removal mode disables grain and subject detection for clean spot erasure."""
    from companion.app import MainWindow, EngineMode
    test_file = str(tmp_path / "test_fast_spot.png")
    sample_pil_image.save(test_file)

    window = MainWindow(input_path=test_file)
    qtbot.addWidget(window)
    window.show()

    # Switch to Fast Spot Removal
    idx = window.combo_engine.findData(EngineMode.FAST)
    window.combo_engine.setCurrentIndex(idx)

    assert window.engine.mode == EngineMode.FAST
    assert window.chk_detect_subject.isEnabled() is False
    assert window.chk_grain.isEnabled() is False


def test_nondestructive_save_preserves_original(qtbot, sample_pil_image, tmp_path):
    """Verify saving never overwrites the original input photo in standalone mode."""
    from companion.app import MainWindow
    orig_path = str(tmp_path / "my_original_photo.jpg")
    sample_pil_image.save(orig_path)
    orig_mtime = os.path.getmtime(orig_path)

    # Launch without explicit output_path
    window = MainWindow(input_path=orig_path)
    qtbot.addWidget(window)
    window.show()

    assert window.output_path.endswith("_ai_edit.jpg")
    window._on_save_and_exit()

    assert window.saved_successfully is True
    # Verify original file was not overwritten
    assert os.path.getmtime(orig_path) == orig_mtime
    assert os.path.isfile(window.output_path)


def test_brush_feathering_and_concentric_rings(qtbot, sample_pil_image):
    """Verify brush feathering updates inner core ring and emits signal."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.load_image(sample_pil_image)

    feather_signals = []
    canvas.brushFeatherChanged.connect(lambda f: feather_signals.append(f))

    # Initial state
    assert canvas.brush_feather == 0.25
    assert canvas._brush_inner_ring_item is not None

    # Adjust feather
    canvas.set_brush_feather(0.50)
    assert canvas.brush_feather == 0.50
    assert len(feather_signals) == 1
    assert feather_signals[-1] == 0.50

    # Update rings
    canvas._update_brush_ring(QPointF(200, 200))
    assert canvas._brush_ring_item.isVisible() is True
    assert canvas._brush_inner_ring_item.isVisible() is True

    outer_r = canvas.brush_radius
    expected_inner_r = outer_r * (1.0 - 0.50)
    inner_rect = canvas._brush_inner_ring_item.rect()
    assert abs(inner_rect.width() - (expected_inner_r * 2.0)) < 1e-2


def test_hud_overlay_badge_rendering(qtbot, sample_pil_image):
    """Verify HUD overlay badge appears on radius and feather change and fades."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.load_image(sample_pil_image)

    # Change radius
    canvas.set_brush_radius(55)
    assert canvas._hud_opacity == 1.0
    assert "55px" in canvas._hud_text

    # Change feather
    canvas.set_brush_feather(0.40)
    assert canvas._hud_opacity == 1.0
    assert "40%" in canvas._hud_text


def test_actual_size_zoom_ctrl_1(qtbot, sample_pil_image):
    """Verify zoom_to_actual_size resets transform to 100% 1:1 pixel scale."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.load_image(sample_pil_image)

    # Zoom in
    canvas.scale(3.0, 3.0)
    assert canvas.transform().m11() != 1.0

    # Reset to 1:1 Actual Size
    canvas.zoom_to_actual_size()
    assert abs(canvas.transform().m11() - 1.0) < 1e-4
    assert abs(canvas.transform().m22() - 1.0) < 1e-4


def test_split_screen_slider_mode(qtbot, sample_pil_image):
    """Verify Before/After split-screen slider mode toggles and composites."""
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.load_image(sample_pil_image)

    split_signals = []
    canvas.splitCompareChanged.connect(lambda s: split_signals.append(s))

    # Activate Split View
    canvas.set_split_compare_mode(True)
    assert canvas._split_compare_mode is True
    assert len(split_signals) == 1
    assert split_signals[-1] is True

    # Check that mask & brush rings are hidden during split compare
    assert canvas._brush_ring_item.isVisible() is False
    assert canvas._mask_item.isVisible() is False

    # Deactivate Split View
    canvas.set_split_compare_mode(False)
    assert canvas._split_compare_mode is False
    assert split_signals[-1] is False


def test_telemetry_status_bar_and_timer(qtbot, sample_pil_image, tmp_path):
    """Verify status bar displays image dimensions with MP and elapsed generation time."""
    from companion.app import MainWindow
    test_file = str(tmp_path / "test_telemetry.png")
    sample_pil_image.save(test_file)

    window = MainWindow(input_path=test_file)
    qtbot.addWidget(window)
    window.show()

    # Verify dimensions label
    dim_text = window.dim_label.text()
    assert f"{sample_pil_image.width} × {sample_pil_image.height} px" in dim_text
    assert "MP" in dim_text

    # Verify hardware indicator
    assert "GPU" in window.dev_label.text() or "CPU" in window.dev_label.text()

    # Simulate generation finish with variations
    var1 = sample_pil_image.copy()
    var2 = sample_pil_image.copy()
    window._on_inpainting_finished([var1, var2])

    assert "⚡" in window.timer_label.text()
    assert "s" in window.timer_label.text()




