import sys

import pytest


@pytest.fixture(autouse=True)
def _isolated_recovery_root(tmp_path, monkeypatch):
    """Keep crash-recovery autosaves out of the real project .recovery/ folder."""
    app = sys.modules.get("companion.app")
    if app is not None:
        monkeypatch.setattr(app, "_RECOVERY_ROOT", str(tmp_path / ".recovery"))
        # closeEvent saves the window layout; never let a test touch the owner's real QSettings
        from PyQt6.QtCore import QSettings

        ini = str(tmp_path / "isolated_settings.ini")
        monkeypatch.setattr(app, "_open_settings", lambda: QSettings(ini, QSettings.Format.IniFormat))


@pytest.fixture(autouse=True)
def _no_ollama_network(monkeypatch):
    """The assistant model picker asks a local Ollama for its models; tests never reach the network."""
    llm = sys.modules.get("companion.agent.llm")
    if llm is None:  # not imported by this test module: nothing can ask Ollama
        return

    def offline(url, timeout):
        raise OSError("network disabled in tests")

    monkeypatch.setattr(llm, "_fetch_tags", offline)
