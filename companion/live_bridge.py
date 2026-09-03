"""
live_bridge.py
Efface Magique LR - High-Performance Live IPC Bridge
Lightweight, thread-safe local HTTP bridge connecting Adobe Lightroom Classic
with the Python AI Companion app for instantaneous, zero-latency image synchronization.
"""

import os
import sys
import json
import socket
import logging
import threading
from typing import Optional, Dict, Any, List
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger("LiveBridge")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
DEFAULT_BRIDGE_PORT = 51739
BRIDGE_INFO_FILENAME = os.path.join(_PROJECT_ROOT, ".tmp", "live_bridge.json")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class LiveBridgeRequestHandler(BaseHTTPRequestHandler):
    """Handles incoming REST commands from Adobe Lightroom Classic SDK."""

    def log_message(self, format, *args):
        # Suppress routine HTTP request logging to avoid stdout clutter
        logger.debug("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))

    def _send_json_response(self, status_code: int, data: Dict[str, Any]):
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _get_bridge_server(self) -> Optional["LiveBridgeServer"]:
        server = getattr(self, "server", None)
        if server is None:
            return None
        bridge = getattr(server, "bridge_server", None)
        if bridge is None or not getattr(bridge, "is_running", False):
            return None
        return bridge

    def do_GET(self):
        bridge_server = self._get_bridge_server()
        if bridge_server is None:
            self._send_json_response(503, {"error": "Server stopping or unavailable"})
            return

        if self.path == "/api/ping":
            self._send_json_response(200, {
                "status": "ok",
                "app": "EffaceMagique",
                "version": "1.0",
                "pid": os.getpid(),
                "active_image": bridge_server.active_image_path,
            })
            return

        elif self.path == "/api/pending_imports":
            imports = bridge_server.get_pending_imports()
            self._send_json_response(200, {
                "status": "ok",
                "count": len(imports),
                "imports": imports,
            })
            return

        self._send_json_response(404, {"error": "Endpoint not found"})

    def do_POST(self):
        bridge_server = self._get_bridge_server()
        if bridge_server is None:
            self._send_json_response(503, {"error": "Server stopping or unavailable"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception as e:
            self._send_json_response(400, {"error": f"Invalid JSON payload: {e}"})
            return

        if self.path == "/api/select":
            image_path = payload.get("path")
            if not image_path:
                self._send_json_response(400, {"error": "Missing 'path' parameter"})
                return

            logger.info(f"Received selection notification from Lightroom: {image_path}")
            bridge_server.active_image_path = image_path
            try:
                bridge_server.photo_selected.emit(payload)
            except (RuntimeError, AttributeError):
                pass
            self._send_json_response(200, {
                "status": "received",
                "path": image_path,
                "photo_id": payload.get("photo_id", ""),
            })
            return

        elif self.path == "/api/import_done":
            imported_path = payload.get("path")
            if imported_path:
                bridge_server.remove_pending_import(imported_path)
                try:
                    bridge_server.import_completed.emit(payload)
                except (RuntimeError, AttributeError):
                    pass
                logger.info(f"Acknowledged import completion in Lightroom: {imported_path}")
            self._send_json_response(200, {"status": "ok"})
            return

        elif self.path == "/api/focus":
            try:
                bridge_server.focus_requested.emit()
            except (RuntimeError, AttributeError):
                pass
            self._send_json_response(200, {"status": "ok"})
            return

        elif self.path == "/api/close":
            self._send_json_response(200, {"status": "closing"})
            try:
                bridge_server.close_requested.emit()
            except (RuntimeError, AttributeError):
                pass
            return

        self._send_json_response(404, {"error": "Endpoint not found"})


class LiveBridgeServer(QObject):
    """
    Thread-safe IPC Server that listens for Adobe Lightroom Classic events.
    Emits Qt signals on the main thread for seamless UI updates.
    """
    photo_selected = pyqtSignal(dict)
    focus_requested = pyqtSignal()
    import_completed = pyqtSignal(dict)
    close_requested = pyqtSignal()

    def __init__(self, preferred_port: int = DEFAULT_BRIDGE_PORT, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.preferred_port = preferred_port
        self.port: Optional[int] = None
        self.http_server: Optional[ThreadedHTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.active_image_path: Optional[str] = None
        self._pending_imports: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.is_running = False

    def start(self) -> int:
        """Start the background HTTP server and write metadata to .tmp/live_bridge.json."""
        if self.is_running:
            return self.port or self.preferred_port

        # Attempt to bind preferred port, fallback to dynamic ephemeral port (0) if occupied
        if self.preferred_port == 0:
            server_address = ("127.0.0.1", 0)
            self.http_server = ThreadedHTTPServer(server_address, LiveBridgeRequestHandler)
            self.port = self.http_server.server_port
        else:
            try:
                server_address = ("127.0.0.1", self.preferred_port)
                self.http_server = ThreadedHTTPServer(server_address, LiveBridgeRequestHandler)
                self.port = self.preferred_port
            except OSError:
                logger.warning(f"Port {self.preferred_port} in use; allocating OS ephemeral port...")
                server_address = ("127.0.0.1", 0)
                self.http_server = ThreadedHTTPServer(server_address, LiveBridgeRequestHandler)
                self.port = self.http_server.server_port

        self.http_server.bridge_server = self
        self.is_running = True

        self.server_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        self.server_thread.start()

        self._write_info_file()
        logger.info(f"Live Bridge Server running on http://127.0.0.1:{self.port} (PID: {os.getpid()})")
        return self.port

    def stop(self):
        """Cleanly terminate the server and remove info file."""
        self.is_running = False
        if self.http_server:
            try:
                self.http_server.shutdown()
            except Exception:
                pass
            try:
                self.http_server.server_close()
            except Exception:
                pass
            self.http_server = None

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=1.0)
            self.server_thread = None

        self._remove_info_file()
        logger.info("Live Bridge Server stopped.")

    def queue_import(self, export_path: str, original_path: Optional[str] = None, photo_id: Optional[str] = None):
        """Queue an edited image for Lightroom to re-import and stack."""
        with self._lock:
            # Avoid duplicate queuing for same path
            for item in self._pending_imports:
                if item.get("path") == export_path:
                    return
            self._pending_imports.append({
                "path": export_path,
                "original_path": original_path,
                "photo_id": photo_id,
            })
            logger.info(f"Queued pending import for Lightroom: {export_path}")

    def get_pending_imports(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._pending_imports)

    def remove_pending_import(self, path: str):
        with self._lock:
            self._pending_imports = [item for item in self._pending_imports if item.get("path") != path]

    def _write_info_file(self):
        """Save server port and process ID to .tmp/live_bridge.json for Lightroom plugin discovery."""
        try:
            os.makedirs(os.path.dirname(BRIDGE_INFO_FILENAME), exist_ok=True)
            info = {
                "port": self.port,
                "pid": os.getpid(),
                "url": f"http://127.0.0.1:{self.port}",
            }
            with open(BRIDGE_INFO_FILENAME, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write bridge info file {BRIDGE_INFO_FILENAME}: {e}")

    def _remove_info_file(self):
        """Remove info file when stopping."""
        try:
            if os.path.isfile(BRIDGE_INFO_FILENAME):
                os.remove(BRIDGE_INFO_FILENAME)
        except Exception as e:
            logger.debug(f"Could not remove bridge info file: {e}")
