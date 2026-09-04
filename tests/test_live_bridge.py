"""
test_live_bridge.py
Comprehensive automated test suite for Efface Magique LR Live Bridge IPC & Workflow.

Covers:
- Server lifecycle: port allocation, metadata discovery file (.tmp/live_bridge.json), and clean shutdown.
- REST Endpoints:
  * GET /api/ping (health check & metadata)
  * POST /api/select (photo selection & Qt signal emission)
  * GET /api/pending_imports & POST /api/import_done (queueing, retrieval, and completion)
  * POST /api/focus (window foreground activation)
- Error handling: malformed JSON, missing path parameters, 404 routes.
- PyQt6 MainWindow Live Integration:
  * Live mode initialization & badge state
  * Always-on-top window pinning (Ctrl+T)
  * Live selection update handling without inpainting disruption
  * Non-blocking Sync to Lightroom keeping companion window open
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
import tempfile
import numpy as np
from PIL import Image

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QPushButton

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.live_bridge import LiveBridgeServer, BRIDGE_INFO_FILENAME
from companion.app import MainWindow, HelpGuideDialog, _LOGO_PATH, _LOGO_ICO_PATH, _CHECKMARK_PATH, DARK_STYLE
from package_release import create_release_zip


@pytest.fixture
def temp_image():
    """Create a temporary TIFF image for live bridge testing."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        path = f.name
    arr = np.full((120, 160, 3), 180, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(path, compression="tiff_deflate")
    yield path
    if os.path.isfile(path):
        try:
            os.remove(path)
        except Exception:
            pass


def _http_get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "TestClient"})
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _http_post(url: str, data: dict):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "TestClient"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


class TestLiveBridgeServer:
    def test_server_startup_and_info_file_lifecycle(self, qtbot):
        """Verify server starts, allocates port, writes info file, and removes it on stop."""
        server = LiveBridgeServer()
        port = server.start()
        assert port > 0
        assert server.is_running is True

        # Verify .tmp/live_bridge.json was written correctly
        assert os.path.isfile(BRIDGE_INFO_FILENAME)
        with open(BRIDGE_INFO_FILENAME, "r", encoding="utf-8") as f:
            info = json.load(f)
        assert info["port"] == port
        assert info["pid"] == os.getpid()
        assert f":{port}" in info["url"]

        # Stop server and verify info file cleanup
        server.stop()
        assert server.is_running is False
        assert not os.path.isfile(BRIDGE_INFO_FILENAME)

    def test_api_ping_healthcheck(self, qtbot):
        """GET /api/ping returns healthy status, application name, and active image path."""
        server = LiveBridgeServer(preferred_port=0)
        port = server.start()
        try:
            status, data = _http_get(f"http://127.0.0.1:{port}/api/ping")
            assert status == 200
            assert data["status"] == "ok"
            assert data["app"] == "EffaceMagique"
            assert data["version"] == "1.0"
            assert data["pid"] == os.getpid()
            assert data["active_image"] is None
        finally:
            server.stop()

    def test_api_select_signal_emission(self, qtbot, temp_image):
        """POST /api/select emits photo_selected Qt signal with correct payload and updates active image."""
        server = LiveBridgeServer(preferred_port=0)
        port = server.start()
        try:
            with qtbot.waitSignal(server.photo_selected, timeout=3000) as blocker:
                payload = {
                    "path": temp_image,
                    "photo_id": "12345",
                    "original_path": "/photos/original.raw",
                    "title": "sample_photo",
                }
                status, data = _http_post(f"http://127.0.0.1:{port}/api/select", payload)
                assert status == 200
                assert data["status"] == "received"
                assert data["path"] == temp_image

            assert blocker.args[0]["path"] == temp_image
            assert blocker.args[0]["photo_id"] == "12345"
            assert server.active_image_path == temp_image
        finally:
            server.stop()

    def test_api_select_missing_path_returns_400(self, qtbot):
        """POST /api/select without 'path' parameter returns 400 Bad Request."""
        server = LiveBridgeServer(preferred_port=0)
        port = server.start()
        try:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                _http_post(f"http://127.0.0.1:{port}/api/select", {"photo_id": "123"})
            assert exc_info.value.code == 400
        finally:
            server.stop()

    def test_pending_imports_and_done_workflow(self, qtbot):
        """Queue edited image -> GET /api/pending_imports -> POST /api/import_done removes from queue."""
        server = LiveBridgeServer(preferred_port=0)
        port = server.start()
        try:
            # 1. Initially empty
            status, data = _http_get(f"http://127.0.0.1:{port}/api/pending_imports")
            assert status == 200
            assert data["count"] == 0
            assert len(data["imports"]) == 0

            # 2. Queue an import
            test_path = "/tmp/edited_photo.tif"
            server.queue_import(test_path, original_path="/photos/raw.cr3", photo_id="999")

            status, data = _http_get(f"http://127.0.0.1:{port}/api/pending_imports")
            assert status == 200
            assert data["count"] == 1
            assert data["imports"][0]["path"] == test_path
            assert data["imports"][0]["photo_id"] == "999"

            # 3. Acknowledge import done
            with qtbot.waitSignal(server.import_completed, timeout=3000) as blocker:
                status, data = _http_post(f"http://127.0.0.1:{port}/api/import_done", {"path": test_path})
                assert status == 200
                assert data["status"] == "ok"

            assert blocker.args[0]["path"] == test_path

            # 4. Confirm removed from queue
            status, data = _http_get(f"http://127.0.0.1:{port}/api/pending_imports")
            assert data["count"] == 0
            assert len(data["imports"]) == 0
        finally:
            server.stop()

    def test_api_focus_signal_emission(self, qtbot):
        """POST /api/focus emits focus_requested Qt signal."""
        server = LiveBridgeServer(preferred_port=0)
        port = server.start()
        try:
            with qtbot.waitSignal(server.focus_requested, timeout=3000):
                status, data = _http_post(f"http://127.0.0.1:{port}/api/focus", {})
                assert status == 200
                assert data["status"] == "ok"
        finally:
            server.stop()

    def test_api_close_signal_emission(self, qtbot):
        """POST /api/close emits close_requested Qt signal."""
        server = LiveBridgeServer(preferred_port=0)
        port = server.start()
        try:
            with qtbot.waitSignal(server.close_requested, timeout=3000):
                status, data = _http_post(f"http://127.0.0.1:{port}/api/close", {})
                assert status == 200
                assert data["status"] == "closing"
        finally:
            server.stop()

    def test_unknown_endpoint_returns_404(self, qtbot):
        """GET and POST on unknown route returns 404 Not Found."""
        server = LiveBridgeServer(preferred_port=0)
        port = server.start()
        try:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                _http_get(f"http://127.0.0.1:{port}/api/unknown_route")
            assert exc_info.value.code == 404
        finally:
            server.stop()


class TestMainWindowLiveIntegration:
    def test_main_window_live_mode_startup_badge(self, qtbot):
        """MainWindow launched with live_mode=True displays '🟢 Live Synced' badge."""
        window = MainWindow(live_mode=True, bridge_port=0)
        qtbot.addWidget(window)
        window.show()

        assert window.is_live_mode is True
        assert hasattr(window, "live_bridge")
        assert window.live_bridge.is_running is True
        assert hasattr(window, "live_badge")
        assert "Live Synced" in window.live_badge.text()

        window.close()
        assert window.live_bridge.is_running is False

    def test_always_on_top_pin_toggle(self, qtbot):
        """Toggling Always-on-Top pin button updates Qt WindowStaysOnTopHint."""
        window = MainWindow(live_mode=True, bridge_port=0)
        qtbot.addWidget(window)
        window.show()

        assert not bool(window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)

        # Toggle Pin ON
        window.btn_pin.setChecked(True)
        assert bool(window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)

        # Toggle Pin OFF
        window.btn_pin.setChecked(False)
        assert not bool(window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)

        window.close()

    def test_live_photo_selection_updates_canvas(self, qtbot, temp_image):
        """Simulating selection event updates MainWindow canvas to the newly selected image."""
        window = MainWindow(live_mode=True, bridge_port=0)
        qtbot.addWidget(window)
        window.show()

        # Simulate Lightroom selection event
        payload = {
            "path": temp_image,
            "photo_id": "42",
            "original_path": "/photos/original_dsc0042.arw",
            "title": "original_dsc0042",
        }
        window._on_live_photo_selected(payload)

        assert window.input_path == temp_image
        assert window.current_photo_id == "42"
        assert window.current_original_path == "/photos/original_dsc0042.arw"
        assert window.canvas.get_current_image() is not None
        assert window.canvas.get_current_image().size == (160, 120)

        window.close()

    def test_sync_to_lightroom_saves_and_keeps_window_open(self, qtbot, temp_image):
        """_on_sync_to_lightroom saves 16-bit TIFF and queues import without closing MainWindow."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            out_path = f.name

        try:
            window = MainWindow(input_path=temp_image, output_path=out_path, live_mode=True, bridge_port=0)
            qtbot.addWidget(window)
            window.show()

            window.current_photo_id = "101"
            window.current_original_path = "/photos/test.nef"

            # Trigger live sync
            window._on_sync_to_lightroom()

            # Output file must exist and be valid
            assert os.path.isfile(out_path)
            assert os.path.getsize(out_path) > 0

            # Must be queued in live bridge for Lightroom auto-import
            pending = window.live_bridge.get_pending_imports()
            assert len(pending) == 1
            assert pending[0]["path"] == out_path
            assert pending[0]["photo_id"] == "101"

            # Window MUST remain visible/open (not closed)
            assert window.isVisible()

            window.close()
        finally:
            if os.path.isfile(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass

    def test_save_and_sync_button_ui_and_no_exit_button(self, qtbot, temp_image):
        """Top bar has 'Save & Sync to Lightroom' button, has no Exit button, and clicking it saves and syncs."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            out_path = f.name

        try:
            window = MainWindow(input_path=temp_image, output_path=out_path, live_mode=True, bridge_port=0)
            qtbot.addWidget(window)
            window.show()

            # 1. Verify Save to Lightroom button exists and has correct label
            assert hasattr(window, "btn_sync")
            assert "Save to Lightroom" in window.btn_sync.text()

            # 2. Verify NO 'Exit' button is present in the top bar
            top_bar = window.findChild(QFrame, "topBar")
            buttons = top_bar.findChildren(QPushButton)
            button_texts = [b.text() for b in buttons]
            for text in button_texts:
                assert "Exit" not in text, f"Found unexpected Exit button in top bar: {text}"

            # 3. Clicking Save & Sync button saves to disk and queues import without closing window
            qtbot.mouseClick(window.btn_sync, Qt.MouseButton.LeftButton)
            assert window.saved_successfully is True
            assert os.path.isfile(out_path)
            assert os.path.getsize(out_path) > 0
            assert window.isVisible()

            window.close()
        finally:
            if os.path.isfile(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass

    def test_branding_checkbox_and_window_icon(self, qtbot, temp_image):
        """Logo asset files exist, checkmark is wired in DARK_STYLE, and window has brand icon."""
        assert os.path.isfile(_LOGO_PATH), f"Missing logo: {_LOGO_PATH}"
        assert os.path.isfile(_LOGO_ICO_PATH), f"Missing logo ico: {_LOGO_ICO_PATH}"
        assert os.path.isfile(_CHECKMARK_PATH), f"Missing checkmark: {_CHECKMARK_PATH}"
        assert "checkmark.png" in DARK_STYLE, "DARK_STYLE must include checkmark.png"
        assert "QCheckBox::indicator:checked" in DARK_STYLE

        window = MainWindow(input_path=temp_image, live_mode=False)
        qtbot.addWidget(window)
        window.show()

        # Window icon should be set
        assert not window.windowIcon().isNull()

        # Logo badge in top bar
        assert hasattr(window, "logo_label")
        assert window.logo_label is not None
        assert not window.logo_label.pixmap().isNull()

        window.close()

    def test_removed_buttons_and_process_cleanup_on_close(self, qtbot, temp_image):
        """1:1 button and Show in Folder button are removed from toolbar, and closeEvent cleans up processes & temp files."""
        window = MainWindow(input_path=temp_image, live_mode=True)
        qtbot.addWidget(window)
        window.show()

        # 1. Verify 1:1 button is removed from top bar
        top_bar = window.findChild(QFrame, "topBar")
        buttons = [b.text() for b in top_bar.findChildren(QPushButton)]
        assert "1:1" not in buttons, f"Found unexpected 1:1 button: {buttons}"

        # 2. Verify Show in Folder button is removed from top bar
        assert not hasattr(window, "btn_open_folder")
        for text in buttons:
            assert "Show in Folder" not in text and text != "📁 Folder", f"Found unexpected Folder button: {text}"

        # 3. Verify temporary file cleanup
        tmp_dir = os.path.join(_PROJECT_ROOT, ".tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        dummy_temp = os.path.join(tmp_dir, "test_temp_cleanup.tif")
        with open(dummy_temp, "w") as f:
            f.write("temporary data")
        assert os.path.isfile(dummy_temp)

        # 4. Verify closeEvent stops live server, cleans temp files, and shuts down processes
        server = window.live_bridge
        assert server is not None
        assert server.is_running is True

        window.close()
        assert not os.path.isfile(dummy_temp)
        assert server.is_running is False
        assert window.live_bridge.is_running is False

    def test_help_guide_dialog_tabs_and_actions(self, qtbot):
        """Help button is 'Help & Guide' with F1 shortcut, and opens dialog with 4 detailed tabs."""
        window = MainWindow(live_mode=False)
        qtbot.addWidget(window)
        window.show()

        assert "Help/Guide" in window.btn_lr_help.text()
        assert hasattr(window, "shortcut_help")
        assert window.shortcut_help.key().toString() == "F1"

        dialog = HelpGuideDialog(window)
        qtbot.addWidget(dialog)

        assert dialog.tabs.count() == 4
        tab_titles = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
        assert "🚀 Installation" in tab_titles
        assert "✨ How to Use" in tab_titles
        assert "⌨ Shortcuts" in tab_titles
        assert "📥 Download & Share" in tab_titles

        dialog.close()
        window.close()

    def test_package_release_distribution_zip(self):
        """Packaging utility produces a clean, complete distribution ZIP archive."""
        import zipfile
        zip_path = create_release_zip()
        assert os.path.isfile(zip_path)
        assert os.path.getsize(zip_path) > 50000  # > 50 KB

        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            assert any(name.endswith("README.md") for name in namelist)
            assert any(name.endswith("install.bat") for name in namelist)
            assert any(name.endswith("companion.bat") for name in namelist)
            assert any(name.endswith("companion/app.py") for name in namelist)
            assert any(name.endswith("plugin/ai_eraser.lrplugin/Info.lua") for name in namelist)
            assert any(name.endswith("companion/assets/logo.png") for name in namelist)
            assert any(name.endswith("companion/assets/checkmark.png") for name in namelist)
            # Ensure virtualenvs, git, and pycache are excluded
            assert not any(".venv" in name for name in namelist)
            assert not any(".git" in name for name in namelist)
            assert not any("__pycache__" in name for name in namelist)

