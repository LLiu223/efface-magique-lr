"""Job 022: design tokens, contrast, hierarchy, Erase always visible, inline assistant answer, model picker."""

import json
import os
import sys

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import companion.app as app  # noqa: E402
from companion.agent import llm  # noqa: E402
from companion.agent.harness import AgentResult  # noqa: E402
from companion.ui import DARK_STYLE, TOKENS  # noqa: E402


def _luminance(hex_colour):
    rgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(fg, bg):
    hi, lo = sorted((_luminance(TOKENS[fg]), _luminance(TOKENS[bg])), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# Every text/background pair the stylesheets use (disabled text is exempt from WCAG)
TEXT_PAIRS = [
    ("text", "bg"), ("text", "surface"), ("text", "control"), ("text", "control_hover"),
    ("text", "control_checked"), ("text", "field"),
    ("text_muted", "surface"), ("text_muted", "control"), ("text_muted", "field"), ("text_muted", "status"),
    ("on_accent", "accent"), ("on_accent", "accent_hover"), ("on_accent", "accent_pressed"),
    ("success", "surface"), ("success", "control"), ("success", "control_hover"), ("success", "status"),
    ("warning", "status"),
]


@pytest.mark.parametrize("fg,bg", TEXT_PAIRS)
def test_text_tokens_meet_wcag_aa(fg, bg):
    assert contrast(fg, bg) >= 4.5, (fg, bg, round(contrast(fg, bg), 2))


def test_focus_ring_is_visible_against_the_chrome():
    for bg in ("surface", "control", "field"):
        assert contrast("focus", bg) >= 3.0  # non-text UI component contrast


def test_chrome_greys_are_neutral_and_stylesheet_is_built_from_tokens():
    coloured = {"accent", "accent_hover", "accent_pressed", "accent_disabled", "focus", "success",
                "success_border", "warning"}
    for name, value in TOKENS.items():
        if value.startswith("#") and name not in coloured:
            assert value[1:3] == value[3:5] == value[5:7], (name, value)
    assert "$" not in DARK_STYLE
    for state in ("hover", "pressed", "disabled", "focus"):
        for widget in ("QPushButton", "QComboBox", "QLineEdit"):
            assert f"{widget}:{state}" in DARK_STYLE or f", {widget}:{state}" in DARK_STYLE, (widget, state)
        assert f"QCheckBox::indicator:{state}" in DARK_STYLE, state


def _show(qtbot, size):
    w = app.MainWindow()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w.resize(*size)
    qtbot.wait(30)
    QApplication.processEvents()
    return w


@pytest.mark.parametrize("size", [(960, 540), (1920, 1080)])
def test_erase_and_save_visible_and_whole(qtbot, size):
    w = _show(qtbot, size)
    for btn in (w.btn_erase, w.btn_sync):
        assert btn.isVisible() and btn.width() >= btn.sizeHint().width(), btn.text()
        r = btn.rect().translated(btn.mapTo(w, btn.rect().topLeft()))
        assert w.rect().contains(r), (btn.text(), r, w.rect())
    assert w.assistant_bar.isVisible()
    assert w.middle_splitter.geometry().top() >= w.assistant_bar.geometry().bottom()


def _face(btn):
    img = btn.grab().toImage()
    return img.pixelColor(img.width() // 2, 3).name()  # inside the border, above the label


def test_erase_is_the_only_accent_filled_button(qtbot):
    w = _show(qtbot, (1280, 800))
    for b in (w.btn_erase, w.btn_fit, w.btn_sync):
        b.clearFocus()
    w.canvas.setFocus()
    assert w.btn_erase.objectName() == "primaryAction" and w.btn_sync.objectName() == "syncAction"
    assert _face(w.btn_erase) == TOKENS["accent"]
    assert _face(w.btn_fit) == TOKENS["control"]
    assert _face(w.btn_layers_toggle) != TOKENS["accent"]  # checked toggles are not filled with the accent
    assert _face(w.btn_sync) != TOKENS["accent"]


@pytest.fixture
def no_modal(monkeypatch):
    def fail(*a, **k):
        raise AssertionError(f"modal box shown: {a[1:3]}")
    monkeypatch.setattr(QMessageBox, "information", fail)


def test_assistant_answer_is_inline_not_modal(qtbot, no_modal):
    w = _show(qtbot, (1280, 800))
    assert not w.agent_answer_panel.isVisible()
    w._on_agent_result(AgentResult(ok=True, text="I found no <b>cat</b>, so nothing is selected."))
    assert w.agent_answer.text() == "I found no <b>cat</b>, so nothing is selected."
    assert w.agent_answer.textFormat() == app.Qt.TextFormat.PlainText  # LLM text is never rendered as HTML
    assert w.agent_answer_panel.isVisible() and "found no" in w.statusBar().currentMessage()
    w.btn_agent_answer_close.click()
    assert not w.agent_answer_panel.isVisible()

    w._on_agent_result(AgentResult(ok=False, reason="Refused tool call run_shell"))
    assert "Refused tool call" in w.agent_answer.text() and w.agent_answer_panel.isVisible()


# Model picker (owner amendment): installed Ollama models, cloud ids, editable, saved as before

@pytest.fixture
def agent_on(monkeypatch):
    monkeypatch.setattr(app.agent_pkg, "agent_available", lambda: True)


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_ollama_models_listed_from_api_tags(qtbot, agent_on, monkeypatch):
    seen = []

    def tags(url, timeout):
        seen.append((url, timeout))
        return json.dumps({"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.1:latest"}]}).encode()

    monkeypatch.setattr(llm, "_fetch_tags", tags)
    w = app.MainWindow()
    qtbot.addWidget(w)
    assert w.combo_agent_provider.currentText() == "ollama"
    qtbot.waitUntil(lambda: "qwen2.5:7b" in _items(w.ed_agent_model), timeout=3000)
    assert seen == [("http://localhost:11434/api/tags", 1.0)]
    assert w.ed_agent_model.currentText() == "llama3.1"  # the saved value is kept and listed too
    assert set(_items(w.ed_agent_model)) == {"llama3.1:latest", "qwen2.5:7b", "llama3.1"}
    w.ed_agent_model.setCurrentIndex(_items(w.ed_agent_model).index("qwen2.5:7b"))
    assert w._settings.value(app.agent_pkg.KEY_MODEL) == "qwen2.5:7b"


def test_unreachable_ollama_keeps_saved_value_with_hint(qtbot, agent_on, no_modal):
    w = app.MainWindow()  # conftest makes every /api/tags call fail
    qtbot.addWidget(w)
    qtbot.waitUntil(lambda: "not reachable" in w.ed_agent_model.toolTip(), timeout=3000)
    assert w.ed_agent_model.currentText() == "llama3.1"
    w.combo_agent_provider.setCurrentText("anthropic")
    w.combo_agent_provider.setCurrentText("ollama")  # a user switch also says so in the status bar
    qtbot.waitUntil(lambda: "not reachable" in w.statusBar().currentMessage(), timeout=3000)


def test_cloud_provider_lists_known_ids_and_stays_editable(qtbot, agent_on):
    w = app.MainWindow()
    qtbot.addWidget(w)
    w.combo_agent_provider.setCurrentText("openai")
    assert "gpt-4o-mini" in _items(w.ed_agent_model) and w.ed_agent_model.isEditable()
    w.ed_agent_model.setCurrentText("my-fine-tune")  # a name that is not in the list
    assert w._settings.value(app.agent_pkg.KEY_MODEL) == "my-fine-tune"


def test_model_list_only_read_from_a_local_ollama(monkeypatch):
    monkeypatch.setattr(llm, "_fetch_tags", lambda *a: pytest.fail("must not connect"))
    with pytest.raises(ValueError):
        llm.list_ollama_models("http://10.0.0.5:11434")
