"""Photo switching safety: per-photo output paths, deferred live selection, unsynced-layer prompt."""

import os
import sys

import pytest
from PIL import Image
from PyQt6.QtWidgets import QMessageBox

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.app import MainWindow
from companion.layers import ModificationLayer
from companion.inpainting_engine import EngineMode


@pytest.fixture
def photos(tmp_path):
    paths = []
    for name, color in (("A", (200, 0, 0)), ("B", (0, 200, 0))):
        p = str(tmp_path / f"{name}.tif")
        Image.new("RGB", (80, 60), color).save(p)
        paths.append(p)
    return paths


class _StubWorker:
    def deleteLater(self):
        pass


def _add_layer(window):
    img = Image.new("RGB", (80, 60))
    window.modification_layers.append(
        ModificationLayer(layer_id="x", name="Modification 1", mask=Image.new("L", (80, 60)),
                          inpainted_image=img, engine_mode=EngineMode.FAST)
    )


@pytest.fixture
def ask(monkeypatch):
    """Patch QMessageBox.question; set ask.answer, read ask.calls."""
    class Ask:
        answer = QMessageBox.StandardButton.No
        calls = []
    def fake(parent, title, text, *a, **k):
        Ask.calls.append(text)
        return Ask.answer
    monkeypatch.setattr(QMessageBox, "question", fake)
    return Ask


def test_each_photo_gets_its_own_output_path(qtbot, photos):
    a, b = photos
    window = MainWindow()
    qtbot.addWidget(window)
    window.load_image_file(a)
    out_a = window.output_path
    window.load_image_file(b)
    assert window.output_path != out_a
    assert window.output_path == os.path.splitext(b)[0] + "_ai_edit.tif"
    assert window.output_path.endswith("B_ai_edit.tif")


def test_constructor_output_path_survives_initial_load(qtbot, photos, tmp_path):
    out = str(tmp_path / "explicit.tif")
    window = MainWindow(input_path=photos[0], output_path=out)
    qtbot.addWidget(window)
    assert window.output_path == out


def test_selection_during_inpainting_is_applied_after_worker_finishes(qtbot, photos):
    a, b = photos
    window = MainWindow(input_path=a)
    qtbot.addWidget(window)
    assert window._pending_select is None
    window.worker = _StubWorker()
    window._on_live_photo_selected({"path": a, "photo_id": "1"})
    payload = {"path": b, "photo_id": "2"}
    window._on_live_photo_selected(payload)  # newer replaces older
    assert window._pending_select is payload
    assert window.input_path == a

    window._on_worker_thread_finished()
    assert window.worker is None
    assert window._pending_select is None
    assert window.input_path == b
    assert window.current_photo_id == "2"


def test_no_keeps_current_photo_and_drops_payload(qtbot, photos, ask):
    a, b = photos
    window = MainWindow(input_path=a)
    qtbot.addWidget(window)
    window.current_photo_id = "1"
    _add_layer(window)
    out, layers = window.output_path, list(window.modification_layers)

    window.worker = _StubWorker()
    window._on_live_photo_selected({"path": b, "photo_id": "2"})
    window._on_worker_thread_finished()

    assert ask.calls == ["Discard 1 unsaved layers?"]
    assert window.input_path == a
    assert window.output_path == out
    assert window.current_photo_id == "1"
    assert window.modification_layers == layers
    assert window._pending_select is None


def test_yes_loads_new_photo(qtbot, photos, ask):
    a, b = photos
    ask.answer = QMessageBox.StandardButton.Yes
    window = MainWindow(input_path=a)
    qtbot.addWidget(window)
    _add_layer(window)
    window._on_live_photo_selected({"path": b, "photo_id": "2"})
    assert len(ask.calls) == 1
    assert window.input_path == b
    assert window.current_photo_id == "2"
    assert window.modification_layers == []


def test_no_prompt_without_layers(qtbot, photos, ask):
    a, b = photos
    window = MainWindow(input_path=a)
    qtbot.addWidget(window)
    window._on_live_photo_selected({"path": b})
    assert ask.calls == []
    assert window.input_path == b


def test_no_prompt_right_after_sync(qtbot, photos, ask, tmp_path):
    a, b = photos
    window = MainWindow(input_path=a, output_path=str(tmp_path / "out.tif"))
    qtbot.addWidget(window)
    _add_layer(window)
    window._layers_synced = False
    window._on_sync_to_lightroom()
    assert window._layers_synced is True
    window._on_live_photo_selected({"path": b})
    assert ask.calls == []
    assert window.input_path == b
    assert window._layers_synced is False


def test_open_file_asks_and_no_keeps_photo(qtbot, photos, ask, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    a, b = photos
    window = MainWindow(input_path=a)
    qtbot.addWidget(window)
    _add_layer(window)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a_, **k: (b, ""))
    window._on_open_file()
    assert len(ask.calls) == 1
    assert window.input_path == a
