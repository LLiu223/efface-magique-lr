"""Job 019: preview-first Live Sync. /api/preview validation and the companion's preview state machine."""

import json
import logging
import os
import sys

import pytest
from PIL import Image
from PyQt6.QtWidgets import QFileDialog

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import companion.live_bridge as live_bridge
from companion.app import MainWindow
from companion.live_bridge import LiveBridgeServer
from tests.test_live_bridge import _raw_request
import tests.test_photo_switching as switching
from tests.test_photo_switching import _add_layer, _StubWorker

# Reuse the photo-switching fixtures
photos = switching.photos
ask = switching.ask

_JSON = {"Content-Type": "application/json"}


@pytest.fixture
def preview_dir(tmp_path, monkeypatch):
    d = tmp_path / ".tmp"
    d.mkdir()
    monkeypatch.setattr(live_bridge, "PREVIEW_DIR", str(d))
    return d


def _jpeg(path, size=(64, 40)):
    Image.new("RGB", size, (10, 120, 250)).save(str(path), "JPEG")
    return str(path)


# ---------------------------------------------------------------- bridge


class TestPreviewEndpoint:
    @pytest.fixture
    def server(self, qtbot):
        server = LiveBridgeServer(preferred_port=0)
        server.start()
        yield server
        server.stop()

    def _post(self, server, body, headers=_JSON):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        status, _, data = _raw_request(server.port, "POST", "/api/preview", headers, raw)
        return status, json.loads(data)

    def test_good_path_accepted(self, qtbot, server, preview_dir):
        good = _jpeg(preview_dir / "preview_42.jpg")
        got = []
        server.photo_previewed.connect(got.append)
        server.active_image_path = "C:/x/live_1.tif"
        status, data = self._post(server, {"path": good, "photo_id": "42"})
        assert status == 200 and data["photo_id"] == "42"
        qtbot.waitUntil(lambda: len(got) == 1)
        assert got[0]["path"] == good and isinstance(got[0]["received_monotonic"], float)
        assert server.active_image_path == "C:/x/live_1.tif"  # a preview is not the active image

    def test_export_failed_report_accepted(self, qtbot, server, preview_dir):
        got = []
        server.photo_previewed.connect(got.append)
        status, _ = self._post(server, {"photo_id": "42", "export_failed": True})
        assert status == 200
        qtbot.waitUntil(lambda: len(got) == 1)

    def test_bad_bodies_rejected(self, qtbot, server, preview_dir, tmp_path):
        good = _jpeg(preview_dir / "preview_42.jpg")
        other = tmp_path / "other"
        other.mkdir()
        sibling = tmp_path / ".tmp2"
        sibling.mkdir()
        cases = {
            "wrong dir": {"path": _jpeg(other / "preview_42.jpg"), "photo_id": "42"},
            ".tmp2 sibling": {"path": _jpeg(sibling / "preview_42.jpg"), "photo_id": "42"},
            "wrong name": {"path": _jpeg(preview_dir / "live_42.jpg"), "photo_id": "42"},
            "wrong ext": {"path": _jpeg(preview_dir / "preview_42.png"), "photo_id": "42"},
            "id mismatch": {"path": good, "photo_id": "43"},
            "traversal": {"path": str(preview_dir / "sub" / ".." / "preview_42.jpg"), "photo_id": "42"},
            "relative": {"path": "preview_42.jpg", "photo_id": "42"},
            "missing file": {"path": str(preview_dir / "preview_9.jpg"), "photo_id": "9"},
            "bad id chars": {"path": str(preview_dir / "preview_..jpg"), "photo_id": "."},
            "empty id": {"path": good, "photo_id": ""},
            "int id": {"path": good, "photo_id": 42},
            "path not str": {"path": [good], "photo_id": "42"},
            "no path": {"photo_id": "42"},
            "failed not true": {"photo_id": "42", "export_failed": 1},
            "failed with path": {"photo_id": "42", "export_failed": True, "path": good},
        }
        got = []
        server.photo_previewed.connect(got.append)
        for name, body in cases.items():
            status, data = self._post(server, body)
            assert status == 400 and "error" in data, name
        for body in (b"[]", b'"x"', b"1", b"null"):
            assert self._post(server, body)[0] == 400
        qtbot.wait(50)
        assert got == []

    def test_origin_rejected(self, qtbot, server, preview_dir):
        good = _jpeg(preview_dir / "preview_42.jpg")
        status, _ = self._post(server, {"path": good, "photo_id": "42"}, {**_JSON, "Origin": "http://evil.example"})
        assert status == 403

    @pytest.mark.parametrize("value,ok", [(0, True), (1234, True), (-1, False), (True, False), (1.5, False), ("9", False)])
    def test_select_export_ms(self, qtbot, server, tmp_path, value, ok):
        img = str(tmp_path / "live_1.tif")
        Image.new("RGB", (8, 8)).save(img)
        status, _, _ = _raw_request(server.port, "POST", "/api/select", _JSON,
                                    json.dumps({"path": img, "photo_id": "1", "export_ms": value}).encode())
        assert status == (200 if ok else 400)


# ---------------------------------------------------------------- companion


def _locked_widgets(w):
    return [w.btn_open, w.btn_erase, w.btn_sync, w.btn_undo, w.btn_redo, w.btn_reset,
            w.layers_panel, w.carousel_panel]


@pytest.fixture
def win(qtbot, photos):
    window = MainWindow(input_path=photos[0], live_mode=True)
    qtbot.addWidget(window)
    window.current_photo_id = "1"
    return window


def _preview(window, tmp_path, photo_id):
    window._on_live_photo_previewed({"path": _jpeg(tmp_path / f"preview_{photo_id}.jpg", (120, 60)),
                                     "photo_id": photo_id, "received_monotonic": 0.0})


def test_preview_locks_controls_and_leaves_state(win, tmp_path, photos, monkeypatch, caplog):
    base, layers, path = win.base_image, win.modification_layers, win.input_path
    caplog.set_level(logging.INFO)
    _preview(win, tmp_path, "2")
    assert win._preview["photo_id"] == "2"
    assert win.canvas._placeholder is not None and win.canvas._placeholder.width() == 120
    assert "Loading full resolution" in win.live_badge.text()
    assert "Loading full resolution" in win.statusBar().currentMessage()
    assert "preview shown photo_id=2" in caplog.text
    assert not any(w.isEnabled() for w in _locked_widgets(win))
    assert not win.agent_input.isEnabled() and not win.btn_agent_ask.isEnabled()
    assert win.base_image is base and win.modification_layers is layers
    assert (win.input_path, win.current_photo_id, win.base_image16) == (path, "1", None)
    assert win.output_path == os.path.splitext(path)[0] + "_ai_edit.tif"
    assert win.canvas.get_current_image() is not None  # canvas scene untouched under the placeholder

    # Handlers refuse too (keyboard shortcuts bypass disabled buttons)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: pytest.fail("Open File ran"))
    monkeypatch.setattr(win.live_bridge, "queue_import", lambda *a, **k: pytest.fail("synced a preview"))
    monkeypatch.setattr(win, "_start_inpainting_pipeline", lambda *a, **k: pytest.fail("erase ran"))
    win._on_open_file()
    win._on_sync_to_lightroom()
    assert win.save_image_to_disk() is None
    win._on_run_inpainting()
    win._on_generate_more_variations()
    win._on_agent_ask()
    win._on_reset_all()
    win._on_new_layer_clicked()
    win._on_undo()
    win._on_redo()
    win._set_run_controls_enabled(True)  # e.g. a deferred model switch finishing meanwhile
    assert not win.btn_erase.isEnabled()
    assert win._preview is not None and win.input_path == path


def test_full_select_reenables_and_logs(win, tmp_path, photos, caplog):
    caplog.set_level(logging.INFO)
    _preview(win, tmp_path, "2")
    win._on_live_photo_selected({"path": photos[1], "photo_id": "2", "export_ms": 1500})
    assert win._preview is None and win.canvas._placeholder is None
    assert win.input_path == photos[1] and win.current_photo_id == "2"
    assert all(w.isEnabled() for w in _locked_widgets(win))
    assert "Live Synced" in win.live_badge.text()
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("full loaded"))
    assert "photo_id=2" in line and "export_ms=1500" in line and "load_ms=" in line
    assert "since_preview_ms=none" not in line


def test_newer_preview_supersedes_older(win, tmp_path, photos):
    _preview(win, tmp_path, "2")
    _preview(win, tmp_path, "3")
    assert win._preview["photo_id"] == "3"
    win._on_live_photo_selected({"path": photos[2], "photo_id": "3"})
    assert win._preview is None and win.input_path == photos[2]


def test_select_for_loaded_photo_ends_preview(win, tmp_path, photos):
    _add_layer(win)
    win._layers_synced = True
    layers = list(win.modification_layers)
    _preview(win, tmp_path, "2")
    assert win._preview is not None  # synced layers do not block a preview
    win._on_live_photo_selected({"path": photos[0], "photo_id": "1"})  # early return: same photo with layers
    assert win._preview is None and win.canvas._placeholder is None
    assert win.input_path == photos[0] and win.modification_layers == layers
    assert win.btn_sync.isEnabled()


def test_preview_ignored_cases(win, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    _preview(win, tmp_path, "1")  # already loaded
    win.worker = _StubWorker()
    _preview(win, tmp_path, "2")
    win.worker = None
    win._confirming_switch = True
    _preview(win, tmp_path, "2")
    win._confirming_switch = False
    assert win._preview is None and win.canvas._placeholder is None
    for reason in ("photo already loaded", "a worker is running", "a prompt is open"):
        assert reason in caplog.text


def test_no_preview_over_unsynced_layers_and_no_keeps_them(win, tmp_path, photos, ask):
    _add_layer(win)
    layers, out = list(win.modification_layers), win.output_path
    _preview(win, tmp_path, "2")
    assert win._preview is None and win.canvas._placeholder is None
    assert ask.calls == []  # never prompt for a preview
    win._on_live_photo_selected({"path": photos[1], "photo_id": "2"})
    assert ask.calls == ["Discard 1 unsaved layer?"]
    assert (win.input_path, win.current_photo_id, win.output_path) == (photos[0], "1", out)
    assert win.modification_layers == layers


def test_preview_triggers_no_autosave_or_recovery(win, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(win, "_offer_recovery", lambda: calls.append("recovery"))
    monkeypatch.setattr(win, "_autosave", lambda: calls.append("autosave"))
    win._autosave_dirty = False
    _preview(win, tmp_path, "2")
    win._on_autosave_timer()
    assert calls == [] and not win._autosave_dirty
    import companion.app as app
    assert not os.path.exists(app._RECOVERY_ROOT) or os.listdir(app._RECOVERY_ROOT) == []


def test_timeout_restores_previous_photo(win, tmp_path, photos, qtbot):
    win._preview_timer.setInterval(50)
    _preview(win, tmp_path, "2")
    qtbot.waitUntil(lambda: win._preview is None, timeout=2000)
    assert win.canvas._placeholder is None and win.input_path == photos[0]
    assert all(w.isEnabled() for w in _locked_widgets(win))
    msg = win.statusBar().currentMessage()
    assert "export failed or timed out" in msg and os.path.basename(photos[0]) in msg


def test_reported_export_failure_restores_previous_photo(win, tmp_path, photos):
    _preview(win, tmp_path, "2")
    win._on_live_photo_previewed({"photo_id": "9", "export_failed": True})  # another photo: ignored
    assert win._preview is not None
    win._on_live_photo_previewed({"photo_id": "2", "export_failed": True})
    assert win._preview is None and win.canvas._placeholder is None and not win._preview_timer.isActive()
    assert win.input_path == photos[0] and win.current_photo_id == "1"
    assert all(w.isEnabled() for w in _locked_widgets(win))
    assert "still on A.tif" in win.statusBar().currentMessage()


def test_failure_without_loaded_photo_reenables_open(qtbot, tmp_path):
    window = MainWindow(live_mode=True)
    qtbot.addWidget(window)
    _preview(window, tmp_path, "2")
    assert not window.btn_open.isEnabled()
    window._on_live_photo_previewed({"photo_id": "2", "export_failed": True})
    assert window.btn_open.isEnabled() and window.canvas._placeholder is None
    assert window.base_image is None and "no photo loaded" in window.statusBar().currentMessage()
