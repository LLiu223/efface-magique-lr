"""Assistant usage admin dialog: totals, recent calls, CSV export, clear, and assistant settings."""

import datetime as _dt

from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from companion.agent import KEY_OLLAMA_URL, KEY_STORE_PROMPTS, KEY_TOKEN_BUDGET
from companion.agent.llm import DEFAULT_OLLAMA_URL, is_local_url
from companion.config import CONFIG
from companion.ui import APP_STYLE, qss

TOTAL_COLS = ["day", "provider", "model", "calls", "input_tokens", "output_tokens", "cost", "avg_latency_ms", "errors"]
RECENT_COLS = ["ts", "request_id", "provider", "model", "input_tokens", "output_tokens", "latency_ms", "cost", "status"]


def _fmt(col, v):
    if v is None:
        return "-" if col == "cost" else ""
    if col == "ts":
        return _dt.datetime.fromtimestamp(v).strftime("%Y-%m-%d %H:%M:%S")
    if col == "cost":
        return f"${v:.4f}"
    if col == "avg_latency_ms":
        return f"{v:.0f}"
    return str(v)


class UsageDialog(QDialog):
    def __init__(self, store, settings, parent=None):
        super().__init__(parent)
        self.store, self.settings = store, settings
        self.setWindowTitle("Assistant usage")
        self.resize(900, 620)
        self.setStyleSheet(APP_STYLE)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.ed_ollama = QLineEdit(settings.value(KEY_OLLAMA_URL, DEFAULT_OLLAMA_URL, type=str))
        self.ed_ollama.textChanged.connect(self._update_url_warning)
        self.lbl_url_warn = QLabel("")
        self.lbl_url_warn.setStyleSheet(qss("color: $warning;"))
        self.spin_budget = QSpinBox()
        self.spin_budget.setRange(1000, 1_000_000)
        self.spin_budget.setSingleStep(1000)
        self.spin_budget.setValue(settings.value(KEY_TOKEN_BUDGET, CONFIG.AGENT_TOKEN_BUDGET, type=int))
        self.chk_prompts = QCheckBox("Store prompt text in the usage log (off by default)")
        self.chk_prompts.setChecked(settings.value(KEY_STORE_PROMPTS, False, type=bool))
        form.addRow("Ollama URL:", self.ed_ollama)
        form.addRow("", self.lbl_url_warn)
        form.addRow("Token budget per request:", self.spin_budget)
        form.addRow("", self.chk_prompts)
        layout.addLayout(form)
        self._update_url_warning()

        layout.addWidget(QLabel("Totals per day / provider / model"))
        self.tbl_totals = self._table(TOTAL_COLS)
        layout.addWidget(self.tbl_totals)
        layout.addWidget(QLabel("Recent calls"))
        self.tbl_recent = self._table(RECENT_COLS)
        layout.addWidget(self.tbl_recent)

        row = QHBoxLayout()
        self.btn_export = QPushButton("Export CSV...")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_clear = QPushButton("Clear log...")
        self.btn_clear.clicked.connect(self._on_clear)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        row.addWidget(self.btn_export)
        row.addWidget(self.btn_clear)
        row.addStretch(1)
        row.addWidget(btn_close)
        layout.addLayout(row)
        self.refresh()

    @staticmethod
    def _table(cols):
        t = QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        return t

    @staticmethod
    def _fill(table, cols, rows):
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, col in enumerate(cols):
                table.setItem(r, c, QTableWidgetItem(_fmt(col, row.get(col))))

    def refresh(self):
        self._fill(self.tbl_totals, TOTAL_COLS, self.store.summary(0))
        self._fill(self.tbl_recent, RECENT_COLS, self.store.recent(50))

    def _update_url_warning(self):
        url = self.ed_ollama.text().strip()
        self.lbl_url_warn.setText("" if not url or is_local_url(url) else
                                  "Warning: this Ollama host is not localhost; prompts leave this computer.")

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export usage log", "assistant_usage.csv", "CSV (*.csv)")
        if path:
            n = self.store.export_csv(path)
            QMessageBox.information(self, "Export", f"Exported {n} rows.")

    def _on_clear(self):
        answer = QMessageBox.question(
            self, "Clear usage log", "Delete every assistant usage row? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.store.clear()
            self.refresh()

    def done(self, result):
        # Settings are saved on any close (Close button, Esc, window X)
        self.settings.setValue(KEY_OLLAMA_URL, self.ed_ollama.text().strip() or DEFAULT_OLLAMA_URL)
        self.settings.setValue(KEY_TOKEN_BUDGET, self.spin_budget.value())
        self.settings.setValue(KEY_STORE_PROMPTS, self.chk_prompts.isChecked())
        super().done(result)
