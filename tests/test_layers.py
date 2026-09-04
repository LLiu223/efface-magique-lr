"""
test_layers.py
Automated tests for Efface Magique LR Non-Destructive Modification Layer System.

Covers:
- Layer data structure & thumbnail generation
- Layer UI panel widgets and visibility toggling
- Sequential layer creation from brush strokes
- Selecting and modifying the first modification layer
- Dynamic re-composition when editing or toggling layer visibility
- Layer deletion and base photo preview
"""

import os
import sys
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPixmap
import pytest

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.layers import ModificationLayer, create_layer_thumbnail
from companion.inpainting_engine import EngineMode
from companion.app import MainWindow


@pytest.fixture
def sample_test_image(tmp_path):
    """Create a synthetic 400x300 RGB image with distinctive features."""
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    arr[:, :] = [200, 210, 220]  # Light gray-blue background
    # Feature 1: Dark box on left
    arr[50:100, 50:100] = [30, 30, 30]
    # Feature 2: Dark box on right
    arr[180:230, 250:300] = [40, 40, 40]
    img = Image.fromarray(arr, mode="RGB")
    path = str(tmp_path / "test_layers_input.png")
    img.save(path)
    return path, img


def test_layer_dataclass_and_thumbnail(qapp):
    """Verify ModificationLayer properties and thumbnail generation."""
    base = Image.new("RGB", (200, 200), (128, 128, 128))
    mask = Image.new("L", (200, 200), 0)
    # Paint a 30x30 white square on mask
    mask_np = np.zeros((200, 200), dtype=np.uint8)
    mask_np[20:50, 20:50] = 255
    mask = Image.fromarray(mask_np, mode="L")

    result = Image.new("RGB", (200, 200), (100, 150, 200))
    layer = ModificationLayer(
        layer_id="test_mod_1",
        name="Modification 1",
        mask=mask,
        inpainted_image=result,
        engine_mode=EngineMode.FAST,
    )
    layer.update_thumbnail(base)

    assert layer.name == "Modification 1"
    assert layer.visible is True
    assert layer.thumbnail is not None
    assert isinstance(layer.thumbnail, QPixmap)
    assert not layer.thumbnail.isNull()


def test_layer_panel_widgets_present(qtbot, sample_test_image):
    """Verify layer sidebar panel and buttons exist and are connected."""
    path, _ = sample_test_image
    window = MainWindow(input_path=path)
    qtbot.addWidget(window)
    window.show()

    # Panel and toolbar controls
    assert hasattr(window, "layers_panel")
    assert window.layers_panel.isVisible() is True
    assert hasattr(window, "btn_layers_toggle")
    assert hasattr(window, "btn_new_layer")
    assert hasattr(window, "btn_delete_layer")
    assert hasattr(window, "base_layer_card")

    # Toggle layers panel
    window.btn_layers_toggle.setChecked(False)
    assert window.layers_panel.isVisible() is False
    window.btn_layers_toggle.setChecked(True)
    assert window.layers_panel.isVisible() is True


def test_sequential_layer_creation(qtbot, sample_test_image):
    """Simulate two sequential erase steps and verify layers are added to stack."""
    path, _ = sample_test_image
    window = MainWindow(input_path=path)
    qtbot.addWidget(window)
    window.show()

    # Step 1: Paint over feature 1 (left)
    window.canvas._paint_stroke(QPointF(60, 60), QPointF(90, 90))
    window.canvas._save_mask_state()
    assert window.canvas.has_mask() is True

    # Use FAST engine for test speed
    idx = window.combo_engine.findData(EngineMode.FAST)
    window.combo_engine.setCurrentIndex(idx)

    # Click erase and wait for worker thread to finish
    window.btn_erase.click()
    qtbot.waitUntil(lambda: window.worker is None and len(window.modification_layers) == 1, timeout=25000)

    assert len(window.modification_layers) == 1
    assert window.modification_layers[0].name == "Modification 1"

    # Step 2: Paint over feature 2 (right)
    window.canvas._paint_stroke(QPointF(260, 190), QPointF(290, 220))
    window.canvas._save_mask_state()

    window.btn_erase.click()
    qtbot.waitUntil(lambda: window.worker is None and len(window.modification_layers) == 2, timeout=25000)

    assert len(window.modification_layers) == 2
    assert window.modification_layers[1].name == "Modification 2"


def test_select_and_modify_first_modification(qtbot, sample_test_image):
    """Verify selecting Modification 1 when multiple modifications exist and modifying it."""
    path, _ = sample_test_image
    window = MainWindow(input_path=path)
    qtbot.addWidget(window)
    window.show()
    idx = window.combo_engine.findData(EngineMode.FAST)
    window.combo_engine.setCurrentIndex(idx)

    # Create Layer 1
    window.canvas._paint_stroke(QPointF(60, 60), QPointF(80, 80))
    window.canvas._save_mask_state()
    window.btn_erase.click()
    qtbot.waitUntil(lambda: window.worker is None and len(window.modification_layers) == 1, timeout=25000)

    # Create Layer 2
    window.canvas._paint_stroke(QPointF(260, 190), QPointF(280, 210))
    window.canvas._save_mask_state()
    window.btn_erase.click()
    qtbot.waitUntil(lambda: window.worker is None and len(window.modification_layers) == 2, timeout=25000)

    assert len(window.modification_layers) == 2

    # Click on the FIRST modification (index 0)
    window._select_layer(0)
    assert window.active_layer_index == 0
    assert "Update Modification 1" in window.btn_erase.text()
    assert window.layer_active_banner.isVisible() is True
    assert window.canvas.has_mask() is True

    # Modify Modification 1's mask: expand it with another stroke
    orig_mask_np = np.array(window.canvas.get_mask_image())
    orig_pixel_count = np.count_nonzero(orig_mask_np > 10)

    window.canvas._paint_stroke(QPointF(80, 80), QPointF(110, 110))
    window.canvas._save_mask_state()
    new_mask_np = np.array(window.canvas.get_mask_image())
    new_pixel_count = np.count_nonzero(new_mask_np > 10)
    assert new_pixel_count > orig_pixel_count

    # Update Modification 1
    window.btn_erase.click()
    qtbot.waitUntil(lambda: window.worker is None, timeout=25000)

    # Verify Modification 1 was updated
    assert len(window.modification_layers) == 2
    updated_layer_1 = window.modification_layers[0]
    updated_mask_pixels = np.count_nonzero(np.array(updated_layer_1.mask) > 10)
    assert updated_mask_pixels > orig_pixel_count

    # Verify Modification 2 is still present and visible
    assert window.modification_layers[1].name == "Modification 2"
    assert window.modification_layers[1].visible is True


def test_toggle_visibility_and_delete_layer(qtbot, sample_test_image):
    """Verify toggling layer visibility on/off and deleting a layer."""
    path, _ = sample_test_image
    window = MainWindow(input_path=path)
    qtbot.addWidget(window)
    window.show()
    idx = window.combo_engine.findData(EngineMode.FAST)
    window.combo_engine.setCurrentIndex(idx)

    # Create Layer 1
    window.canvas._paint_stroke(QPointF(60, 60), QPointF(80, 80))
    window.canvas._save_mask_state()
    window.btn_erase.click()
    qtbot.waitUntil(lambda: window.worker is None and len(window.modification_layers) == 1, timeout=25000)

    # Create Layer 2
    window.canvas._paint_stroke(QPointF(260, 190), QPointF(280, 210))
    window.canvas._save_mask_state()
    window.btn_erase.click()
    qtbot.waitUntil(lambda: window.worker is None and len(window.modification_layers) == 2, timeout=25000)

    assert len(window.modification_layers) == 2

    # Toggle Layer 1 off
    window._toggle_layer_visibility(0)
    assert window.modification_layers[0].visible is False

    # Toggle Layer 1 back on
    window._toggle_layer_visibility(0)
    assert window.modification_layers[0].visible is True

    # Delete Layer 1
    window._delete_layer(0)
    assert len(window.modification_layers) == 1
    # Remaining layer is the former Layer 2
    assert window.modification_layers[0].name == "Modification 2"

    # Base photo preview
    window._on_view_base_photo()
    assert window.canvas.get_current_image() is not None
    # Return to composite
    window._on_new_layer_clicked()
    assert window.active_layer_index is None


def test_layer_toggle_box_visibility_and_styling(qtbot, sample_test_image):
    """Verify that the modification layer toggle box is visible with prominent styling and updates in-place."""
    path, _ = sample_test_image
    window = MainWindow(input_path=path)
    qtbot.addWidget(window)
    window.show()
    idx = window.combo_engine.findData(EngineMode.FAST)
    window.combo_engine.setCurrentIndex(idx)

    # Create Layer 1
    window.canvas._paint_stroke(QPointF(60, 60), QPointF(80, 80))
    window.canvas._save_mask_state()
    window.btn_erase.click()
    qtbot.waitUntil(lambda: window.worker is None and len(window.modification_layers) == 1, timeout=25000)

    card = window.layer_card_widgets[0]
    assert hasattr(card, "btn_toggle")
    assert card.btn_toggle.text() == "✓"
    assert "0078d4" in card.btn_toggle.styleSheet() or "0f3d63" in card.btn_toggle.styleSheet()

    # Toggle Layer 1 off
    card.btn_toggle.click()
    assert window.modification_layers[0].visible is False
    assert card.btn_toggle.text() == ""
    assert "555555" in card.btn_toggle.styleSheet()

    # Toggle Layer 1 back on
    card.btn_toggle.click()
    assert window.modification_layers[0].visible is True
    assert card.btn_toggle.text() == "✓"
    assert "0078d4" in card.btn_toggle.styleSheet()


def test_instant_layer_toggle_caching(qtbot, sample_test_image):
    """Verify top layer toggle uses cached pixmaps directly without full re-compositing."""
    path, _ = sample_test_image
    window = MainWindow(input_path=path)
    qtbot.addWidget(window)
    window.show()
    idx = window.combo_engine.findData(EngineMode.FAST)
    window.combo_engine.setCurrentIndex(idx)

    # Create Layer 1
    window.canvas._paint_stroke(QPointF(60, 60), QPointF(80, 80))
    window.canvas._save_mask_state()
    window.btn_erase.click()
    qtbot.waitUntil(lambda: window.worker is None and len(window.modification_layers) == 1, timeout=25000)

    layer = window.modification_layers[0]
    assert layer.cached_pixmap is not None
    assert layer.composite_cache is not None

    # Toggle Layer 1 off: should immediately switch to base photo pixmap
    window._toggle_layer_visibility(0)
    assert window.canvas._curr_pixmap == window._base_pixmap

    # Toggle Layer 1 on: should immediately restore cached_pixmap
    window._toggle_layer_visibility(0)
    assert window.canvas._curr_pixmap == layer.cached_pixmap


def test_deleting_older_layer_removes_its_changes(qtbot, sample_test_image):
    """
    Verify that when deleting an older layer (Modification 1), the remaining
    newer layer (Modification 2) does NOT retain the older layer's changes.
    Feature 1's pixels must be cleanly restored to their original base image values.
    """
    path, orig_img = sample_test_image
    orig_np = np.array(orig_img)

    window = MainWindow(input_path=path)
    qtbot.addWidget(window)
    window.show()
    idx = window.combo_engine.findData(EngineMode.FAST)
    window.combo_engine.setCurrentIndex(idx)

    # Step 1: Create Modification 1 over Feature 1 (left dark box at 60..80, y: 60..80, value [30, 30, 30])
    assert np.all(orig_np[70, 70] == [30, 30, 30])
    assert np.all(orig_np[200, 270] == [40, 40, 40])

    window.canvas._paint_stroke(QPointF(60, 60), QPointF(80, 80))
    window.canvas._save_mask_state()
    window.btn_erase.click()
    qtbot.waitUntil(lambda: window.worker is None and len(window.modification_layers) == 1, timeout=25000)

    # Verify Feature 1 is inpainted (erased, no longer [30, 30, 30])
    curr_1 = np.array(window.canvas.get_current_image())
    assert not np.array_equal(curr_1[70, 70], [30, 30, 30])
    # Feature 2 is still untouched
    assert np.all(curr_1[200, 270] == [40, 40, 40])

    # Step 2: Create Modification 2 over Feature 2 (right dark box at 260..280, y: 190..210, value [40, 40, 40])
    window.canvas._paint_stroke(QPointF(260, 190), QPointF(280, 210))
    window.canvas._save_mask_state()
    window.btn_erase.click()
    qtbot.waitUntil(lambda: window.worker is None and len(window.modification_layers) == 2, timeout=25000)

    # Now both Feature 1 and Feature 2 are inpainted
    curr_2 = np.array(window.canvas.get_current_image())
    assert not np.array_equal(curr_2[70, 70], [30, 30, 30])
    assert not np.array_equal(curr_2[200, 270], [40, 40, 40])

    # Step 3: Delete Modification 1 (index 0)
    window._delete_layer(0)
    assert len(window.modification_layers) == 1
    assert window.modification_layers[0].name == "Modification 2"

    # Step 4: Verify that Feature 1 is RESTORED to [30, 30, 30] in both canvas and layer composite cache!
    curr_after = np.array(window.canvas.get_current_image())
    assert np.all(curr_after[70, 70] == [30, 30, 30]), (
        f"Expected Feature 1 at (70, 70) to be restored to [30, 30, 30], but found {curr_after[70, 70]}"
    )
    # Feature 2 MUST STILL be inpainted (erased)
    assert not np.array_equal(curr_after[200, 270], [40, 40, 40]), (
        f"Expected Feature 2 at (200, 270) to remain inpainted, but found {curr_after[200, 270]}"
    )

    # Step 5: Verify remaining layer's composite_cache also has Feature 1 restored
    layer2_cache = np.array(window.modification_layers[0].composite_cache)
    assert np.all(layer2_cache[70, 70] == [30, 30, 30])
    assert not np.array_equal(layer2_cache[200, 270], [40, 40, 40])
