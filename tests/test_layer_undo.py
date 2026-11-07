"""Undo/redo of layer add, delete and visibility toggle (job 010)."""

import dataclasses
import gc
import os
import sys
import weakref

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QMessageBox

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import companion.app as app_mod
from companion.app import MainWindow


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    # Reloading a photo with autosaved layers asks to restore; never block on a real dialog
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    p = str(tmp_path / "photo.png")
    Image.new("RGB", (120, 90), (180, 190, 200)).save(p)
    w = MainWindow(input_path=p)
    qtbot.addWidget(w)
    return w


def add_layer(window, box, color):
    """Simulate a finished erase: the variation is the current composite with box filled."""
    mask = Image.new("L", window.base_image.size, 0)
    mask.paste(255, box)
    var = window._get_composite_image()
    var.paste(color, box)
    window.last_used_mask = mask
    window._on_inpainting_finished([var])
    return window.modification_layers[-1]


def assert_matches_fresh_composite(window):
    shown = np.asarray(window.canvas.get_current_image().convert("RGB"))
    fresh = np.asarray(window._recompute_composite_stack(0))
    assert np.array_equal(shown, fresh)
    assert sorted(window.layer_card_widgets) == list(range(len(window.modification_layers)))


def test_undo_redo_add(window):
    a = add_layer(window, (10, 10, 40, 40), (255, 0, 0))
    b = add_layer(window, (60, 40, 100, 80), (0, 255, 0))
    window._on_undo()
    assert window.modification_layers == [a]
    assert_matches_fresh_composite(window)
    window._on_redo()
    assert window.modification_layers[1] is b
    assert_matches_fresh_composite(window)
    window._on_undo()
    window._on_undo()
    assert window.modification_layers == []
    assert np.array_equal(np.asarray(window.canvas.get_current_image()), np.asarray(window.base_image))


def test_undo_delete_restores_same_object_at_index(window):
    a = add_layer(window, (10, 10, 40, 40), (255, 0, 0))
    b = add_layer(window, (60, 40, 100, 80), (0, 255, 0))
    window._delete_layer(0)
    assert window.modification_layers == [b]
    window._on_undo()
    assert window.modification_layers[0] is a and window.modification_layers[1] is b
    assert_matches_fresh_composite(window)
    window._on_redo()
    assert window.modification_layers == [b]
    assert_matches_fresh_composite(window)


def test_undo_toggle(window):
    a = add_layer(window, (10, 10, 40, 40), (255, 0, 0))
    add_layer(window, (60, 40, 100, 80), (0, 255, 0))
    window._toggle_layer_visibility(0)
    assert a.visible is False
    window._on_undo()
    assert a.visible is True
    assert_matches_fresh_composite(window)
    window._on_redo()
    assert a.visible is False
    assert_matches_fresh_composite(window)


def test_new_op_clears_redo_and_buttons_follow_history(window):
    assert not window.btn_layer_undo.isEnabled() and not window.btn_layer_redo.isEnabled()
    add_layer(window, (10, 10, 40, 40), (255, 0, 0))
    assert window.btn_layer_undo.isEnabled() and not window.btn_layer_redo.isEnabled()
    window._on_undo()
    assert not window.btn_layer_undo.isEnabled() and window.btn_layer_redo.isEnabled()
    add_layer(window, (60, 40, 100, 80), (0, 255, 0))
    assert window._layer_redo == []
    assert not window.btn_layer_redo.isEnabled()
    assert window.btn_layer_undo.styleSheet() == window.btn_new_layer.styleSheet()
    assert "QPushButton:disabled" in window.btn_layer_undo.styleSheet()  # disabled state is visible
    assert "Ctrl+Z" in window.btn_layer_undo.toolTip() and "Ctrl+Y" in window.btn_layer_redo.toolTip()


def test_history_holds_tuples_and_depth_cap_frees_oldest(window, monkeypatch):
    monkeypatch.setattr(app_mod, "CONFIG", dataclasses.replace(app_mod.CONFIG, LAYER_HISTORY_DEPTH=3))
    add_layer(window, (10, 10, 40, 40), (255, 0, 0))
    doomed = add_layer(window, (60, 40, 100, 80), (0, 255, 0))
    window._delete_layer(1)
    ref = weakref.ref(doomed)
    del doomed
    for _ in range(3):
        window._toggle_layer_visibility(0)
    assert len(window._layer_undo) == 3
    assert all(op == "toggle" and idx == 0 for op, idx, _ in window._layer_undo)
    window.current_variations = []
    window.last_base_image = None
    gc.collect()
    assert ref() is None


def test_ctrl_z_undoes_stroke_before_layer(window):
    add_layer(window, (10, 10, 40, 40), (255, 0, 0))
    window.canvas._paint_stroke(QPointF(70, 50), QPointF(90, 70))
    window.canvas._save_mask_state()
    assert window.canvas.has_mask()
    window.shortcut_undo.activated.emit()
    assert not window.canvas.has_mask()
    assert len(window.modification_layers) == 1
    window.shortcut_redo_y.activated.emit()  # redo the stroke first
    assert window.canvas.has_mask() and len(window.modification_layers) == 1
    window.shortcut_undo.activated.emit()
    window.shortcut_undo.activated.emit()  # now the layer
    assert window.modification_layers == []
    window.shortcut_redo_shift_z.activated.emit()  # stroke redo
    window.shortcut_redo_shift_z.activated.emit()  # layer redo
    assert len(window.modification_layers) == 1


def test_load_and_reset_clear_history(window):
    add_layer(window, (10, 10, 40, 40), (255, 0, 0))
    window._on_reset_all()
    assert window._layer_undo == [] and window._layer_redo == []
    add_layer(window, (10, 10, 40, 40), (255, 0, 0))
    window.load_image_file(window.input_path)
    assert window._layer_undo == [] and window._layer_redo == []


def test_reerase_and_variation_pick_clear_redo(window):
    add_layer(window, (10, 10, 40, 40), (255, 0, 0))
    add_layer(window, (60, 40, 100, 80), (0, 255, 0))
    window._on_undo()
    assert window._layer_redo
    window._select_layer(0)
    window._on_inpainting_finished([window._get_composite_image()])  # in-place update path
    assert window._layer_redo == []
    assert len(window._layer_undo) == 1  # re-erase is not undoable

    add_layer(window, (60, 40, 100, 80), (0, 255, 0))
    window._on_undo()
    assert window._layer_redo
    window.current_variations = [window._get_composite_image()]
    window._on_select_variation(0)
    assert window._layer_redo == []
