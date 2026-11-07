"""
Plain-text diagnostics for bug reports (Help dialog "Copy diagnostics").

Privacy: this text is meant to be pasted publicly, so it only holds versions,
device, bridge port, booleans and the last error message. No paths, env vars or usernames.
"""

from typing import Any, Dict


def _val(info: Dict[str, Any], key: str, default: str = "unknown") -> str:
    v = info.get(key)
    return default if v is None or v == "" else str(v)


def build_diagnostics(info: Dict[str, Any]) -> str:
    """Return one line per field. Missing or None values become 'unknown' (or 'none' for last error)."""
    device = _val(info, "device")
    if info.get("gpu_name"):
        device = f"{device} ({info['gpu_name']})"
    connected = info.get("bridge_connected")
    lines = [
        f"App version: {_val(info, 'app_version')}",
        f"Python: {_val(info, 'python')}",
        f"torch: {_val(info, 'torch')}",
        f"Pillow: {_val(info, 'pillow')}",
        f"PyQt6: {_val(info, 'pyqt6')}",
        f"Device: {device}",
        f"LaMa model: {_val(info, 'lama_model')}",
        f"Bridge port: {_val(info, 'bridge_port')}",
        f"Bridge connected: {'unknown' if connected is None else ('yes' if connected else 'no')}",
        f"Last error: {_val(info, 'last_error', 'none')}",
    ]
    return "\n".join(lines)
