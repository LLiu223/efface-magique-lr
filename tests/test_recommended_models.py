"""Job 023: qwen3:8b is the default and recommended model, thinking off, listed (with a pull hint) before it is pulled."""

import json
import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import companion.app as app  # noqa: E402
from companion.agent import llm  # noqa: E402
from companion.agent.harness import AgentResult  # noqa: E402


@pytest.fixture
def agent_on(monkeypatch):
    monkeypatch.setattr(app.agent_pkg, "agent_available", lambda: True)


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _window(qtbot, monkeypatch, installed):
    monkeypatch.setattr(llm, "_fetch_tags", lambda url, timeout: json.dumps(
        {"models": [{"name": n} for n in installed]}).encode())
    w = app.MainWindow()
    qtbot.addWidget(w)
    qtbot.waitUntil(lambda: w._ollama_installed is not None, timeout=3000)
    return w


def test_fresh_settings_default_to_qwen3_8b_and_a_saved_model_is_kept(qtbot, agent_on, monkeypatch):
    assert app.agent_pkg.DEFAULT_AGENT_MODEL == "qwen3:8b"
    w = _window(qtbot, monkeypatch, [])
    assert w.ed_agent_model.currentText() == "qwen3:8b"
    w._settings.setValue(app.agent_pkg.KEY_MODEL, "llama3.1")
    w2 = _window(qtbot, monkeypatch, [])
    assert w2.ed_agent_model.currentText() == "llama3.1"


def test_recommended_listed_first_with_a_pull_hint_when_missing(qtbot, agent_on, monkeypatch):
    w = _window(qtbot, monkeypatch, ["llama3.1:latest", "qwen3.5:4b"])
    items = _items(w.ed_agent_model)
    assert items[:2] == ["qwen3:8b", "qwen3.5:4b"] and "llama3.1:latest" in items
    assert len(items) == len(set(items))
    assert "recommended" in w.ed_agent_model.itemData(0, app._MODEL_NOTE_ROLE)
    assert "light" in w.ed_agent_model.itemData(1, app._MODEL_NOTE_ROLE)
    assert w.ed_agent_model.itemData(2, app._MODEL_NOTE_ROLE) is None
    assert w.ed_agent_model.currentText() == "qwen3:8b"  # the bare name: what is saved and sent to Ollama
    assert w.agent_model_hint.text() == "not installed: run ollama pull qwen3:8b"
    assert not w.agent_model_hint.isHidden()
    assert w.agent_model_hint.textFormat() == app.Qt.TextFormat.PlainText
    w.ed_agent_model.setCurrentIndex(1)
    assert w._settings.value(app.agent_pkg.KEY_MODEL) == "qwen3.5:4b" and w.agent_model_hint.isHidden()
    w.ed_agent_model.setCurrentText("llama3.1")  # matches llama3.1:latest
    assert w.agent_model_hint.isHidden()


def test_delegate_shows_the_note_but_the_text_stays_bare(qtbot, agent_on, monkeypatch):
    from PyQt6.QtWidgets import QStyleOptionViewItem
    w = _window(qtbot, monkeypatch, [])
    opt = QStyleOptionViewItem()
    w.ed_agent_model.itemDelegate().initStyleOption(opt, w.ed_agent_model.model().index(0, 0))
    assert opt.text == "qwen3:8b  (recommended, 5.2 GB)" and w.ed_agent_model.itemText(0) == "qwen3:8b"


def test_unreachable_ollama_shows_no_pull_hint(qtbot, agent_on):
    w = app.MainWindow()  # conftest makes /api/tags fail
    qtbot.addWidget(w)
    qtbot.waitUntil(lambda: "not reachable" in w.ed_agent_model.toolTip(), timeout=3000)
    assert w.agent_model_hint.isHidden() and w.ed_agent_model.currentText() == "qwen3:8b"


def test_ollama_404_becomes_the_pull_hint(qtbot, agent_on, monkeypatch):
    w = _window(qtbot, monkeypatch, [])
    w._on_agent_result(AgentResult(ok=False, reason="Assistant error: ResponseError: model 'qwen3:8b' not found "
                                                    "(status code: 404)"))
    assert w.agent_answer.text().startswith("qwen3:8b is not installed: run ollama pull qwen3:8b")


def test_qwen3_thinking_off_others_untouched():
    pytest.importorskip("langchain_ollama")
    from langchain_core.messages import HumanMessage
    for name in ("qwen3:8b", "qwen3.5:4b", "Qwen3:4b"):
        m = llm.make_llm("ollama", name)
        assert m.reasoning is False and m._chat_params([HumanMessage("hi")])["think"] is False
    m = llm.make_llm("ollama", "llama3.1")
    assert m.reasoning is None and m._chat_params([HumanMessage("hi")])["think"] is None
