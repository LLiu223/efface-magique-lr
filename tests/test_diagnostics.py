"""Version, Copy diagnostics, bridge connected state, last_error and LaMa load-failure warning (job 011)."""

import os
import sys
import types
import json
import urllib.request

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import companion
from companion.diagnostics import build_diagnostics
from companion.live_bridge import LiveBridgeServer
from companion.app import MainWindow, HelpGuideDialog


def _fake_lama_module(monkeypatch, fail: bool):
    mod = types.ModuleType("simple_lama_inpainting")

    class SimpleLama:
        def __init__(self, device=None):
            if fail:
                raise OSError("big-lama.pt not found")

        def __call__(self, img, mask):
            return img

    mod.SimpleLama = SimpleLama
    monkeypatch.setitem(sys.modules, "simple_lama_inpainting", mod)


def _make_window(qtbot, **kw):
    window = MainWindow(bridge_port=0, **kw)
    qtbot.addWidget(window)
    window._prewarm_thread.join(timeout=10)  # never let the thread race teardown
    return window


def test_version_is_1_1_0():
    assert companion.__version__ == "1.1.0"


def test_build_diagnostics_normal():
    text = build_diagnostics({
        "app_version": "1.1.0", "python": "3.11.4", "torch": "2.3.0", "pillow": "10.3.0",
        "pyqt6": "6.7.0", "device": "cuda", "gpu_name": "RTX 3070", "lama_model": "loaded",
        "bridge_port": 51739, "bridge_connected": True, "last_error": None,
    })
    lines = text.splitlines()
    assert lines == [
        "App version: 1.1.0",
        "Python: 3.11.4",
        "torch: 2.3.0",
        "Pillow: 10.3.0",
        "PyQt6: 6.7.0",
        "Device: cuda (RTX 3070)",
        "LaMa model: loaded",
        "Bridge port: 51739",
        "Bridge connected: yes",
        "Last error: none",
    ]


def test_build_diagnostics_missing_and_none_values():
    text = build_diagnostics({"device": None, "gpu_name": None, "bridge_connected": False, "last_error": "boom"})
    assert "App version: unknown" in text
    assert "Device: unknown" in text
    assert "Bridge port: unknown" in text
    assert "Bridge connected: no" in text
    assert "Last error: boom" in text
    assert len(build_diagnostics({}).splitlines()) == 10


def test_ping_version_and_connected_after_request(qtbot):
    server = LiveBridgeServer(preferred_port=0)
    port = server.start()
    try:
        assert server.is_connected is False
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/ping")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            assert json.loads(resp.read())["version"] == companion.__version__
        assert server.is_connected is True
        server.last_request_time -= 6.0
        assert server.is_connected is False
    finally:
        server.stop()


def test_rejected_request_does_not_count_as_connected(qtbot):
    server = LiveBridgeServer(preferred_port=0)
    port = server.start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/ping", headers={"Origin": "http://evil.example"})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(req, timeout=3.0)
        assert server.last_request_time is None
        assert server.is_connected is False
    finally:
        server.stop()


def test_help_dialog_version_and_copy_diagnostics(qtbot, monkeypatch):
    _fake_lama_module(monkeypatch, fail=False)
    window = _make_window(qtbot)
    assert f"v{companion.__version__}" in window.logo_label.toolTip()
    dialog = HelpGuideDialog(window)
    qtbot.addWidget(dialog)
    assert dialog.version_label.text() == "Version 1.1.0"
    assert dialog.tabs.count() == 4

    QApplication.clipboard().clear()
    dialog.btn_copy_diagnostics.click()
    text = QApplication.clipboard().text()
    assert "App version: 1.1.0" in text
    assert f"Bridge port: {window.live_bridge_port}" in text
    assert "LaMa model: loaded" in text
    dialog.close()
    window.close()


def test_model_load_failure_shows_status_warning(qtbot, monkeypatch):
    _fake_lama_module(monkeypatch, fail=True)
    shown = []
    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: shown.append(a))
    window = _make_window(qtbot)

    assert window.engine.load_error and "big-lama.pt" in window.engine.load_error
    qtbot.waitUntil(lambda: window.model_warning_label.isVisibleTo(window), timeout=3000)
    msg = window.model_warning_label.text()
    assert "weights missing or install broken" in msg
    assert "will not work in Fast mode" in msg
    assert "big-lama.pt" in window.last_error
    assert shown == []  # never modal
    assert "LaMa model: failed" in window.diagnostics_text()
    window.close()


def test_no_warning_when_model_loads(qtbot, monkeypatch):
    _fake_lama_module(monkeypatch, fail=False)
    window = _make_window(qtbot)
    qtbot.wait(50)
    assert window.engine.load_error is None
    assert not window.model_warning_label.isVisibleTo(window)
    assert window.last_error is None
    window.close()


def test_last_error_set_by_inpainting_open_and_save_failures(qtbot, monkeypatch, tmp_path):
    _fake_lama_module(monkeypatch, fail=False)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)
    window = _make_window(qtbot)

    window._on_inpainting_error("CUDA out of memory")
    assert window.last_error == "Inpainting error: CUDA out of memory"

    bad = tmp_path / "broken.tif"
    bad.write_bytes(b"not an image")
    window.load_image_file(str(bad))
    assert window.last_error.startswith("Could not load image:")

    from PIL import Image
    good = tmp_path / "ok.png"
    Image.new("RGB", (8, 8)).save(good)
    window.load_image_file(str(good))
    blocker = tmp_path / "out.png"
    blocker.mkdir()  # a directory where the file should go makes the save fail
    window.output_path = str(blocker)
    assert window.save_image_to_disk() is None
    assert window.last_error.startswith("Could not save file:")
    assert "Last error: Could not save file:" in window.diagnostics_text()
    window.close()
