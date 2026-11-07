"""(g) Assistant usage dialog populates from a temp DB; export; clear needs confirmation; settings saved."""

import os
import sys
import time

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QFileDialog, QMessageBox

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.agent import KEY_OLLAMA_URL, KEY_STORE_PROMPTS, KEY_TOKEN_BUDGET  # noqa: E402
from companion.agent.usage import UsageStore  # noqa: E402
from companion.agent.usage_dialog import RECENT_COLS, TOTAL_COLS, UsageDialog  # noqa: E402


def _dialog(qtbot, tmp_path):
    st = UsageStore(str(tmp_path / "u.db"), prices={"openai/gpt-4o": (2.5, 10.0)})
    now = time.time()
    st.record(request_id="r1", provider="ollama", model="llama3.1", input_tokens=100, output_tokens=20, latency_ms=900, ts=now)
    st.record(request_id="r1", provider="ollama", model="llama3.1", input_tokens=50, output_tokens=10, latency_ms=100, ts=now)
    st.record(request_id="r2", provider="openai", model="gpt-4o", input_tokens=1000, output_tokens=100, latency_ms=400, ts=now)
    st.record(request_id="r3", provider="openai", model="mystery", input_tokens=1, output_tokens=1, latency_ms=1, ts=now)
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    dlg = UsageDialog(st, settings)
    qtbot.addWidget(dlg)
    return dlg, st, settings


def _cell(table, cols, row, col):
    return table.item(row, cols.index(col)).text()


def test_dialog_populates_totals_and_recent(qtbot, tmp_path):
    dlg, _, _ = _dialog(qtbot, tmp_path)
    t = dlg.tbl_totals
    assert t.rowCount() == 3 and dlg.tbl_recent.rowCount() == 4
    rows = {(_cell(t, TOTAL_COLS, r, "provider"), _cell(t, TOTAL_COLS, r, "model")): r for r in range(t.rowCount())}
    r = rows[("ollama", "llama3.1")]
    assert _cell(t, TOTAL_COLS, r, "calls") == "2" and _cell(t, TOTAL_COLS, r, "input_tokens") == "150"
    assert _cell(t, TOTAL_COLS, r, "cost") == "$0.0000" and _cell(t, TOTAL_COLS, r, "avg_latency_ms") == "500"
    assert _cell(t, TOTAL_COLS, rows[("openai", "gpt-4o")], "cost") == "$0.0035"
    assert _cell(t, TOTAL_COLS, rows[("openai", "mystery")], "cost") == "-"  # unknown price: NULL, not a guess
    assert _cell(dlg.tbl_recent, RECENT_COLS, 0, "request_id") in {"r1", "r2", "r3"}


def test_clear_requires_confirmation(qtbot, tmp_path, monkeypatch):
    dlg, st, _ = _dialog(qtbot, tmp_path)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    dlg.btn_clear.click()
    assert len(st.recent()) == 4
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dlg.btn_clear.click()
    assert st.recent() == [] and dlg.tbl_recent.rowCount() == 0 and dlg.tbl_totals.rowCount() == 0


def test_export_csv(qtbot, tmp_path, monkeypatch):
    dlg, _, _ = _dialog(qtbot, tmp_path)
    out = tmp_path / "out.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "CSV (*.csv)"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    dlg.btn_export.click()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("id,ts,request_id,provider,model") and len(lines) == 5


def test_settings_saved_and_remote_ollama_warned(qtbot, tmp_path):
    dlg, _, settings = _dialog(qtbot, tmp_path)
    assert dlg.lbl_url_warn.text() == ""
    dlg.ed_ollama.setText("http://192.168.1.20:11434")
    assert "not localhost" in dlg.lbl_url_warn.text()
    dlg.spin_budget.setValue(5000)
    dlg.chk_prompts.setChecked(True)
    dlg.accept()
    assert settings.value(KEY_OLLAMA_URL) == "http://192.168.1.20:11434"
    assert settings.value(KEY_TOKEN_BUDGET, type=int) == 5000
    assert settings.value(KEY_STORE_PROMPTS, type=bool) is True
