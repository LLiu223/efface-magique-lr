"""Job 026: light/dark toggle applies every token style at once, is saved in QSettings and defaults to light."""

import os
import sys

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import companion.app as app  # noqa: E402
import companion.ui as ui  # noqa: E402
from companion.agent.usage_dialog import UsageDialog  # noqa: E402


@pytest.fixture(autouse=True)
def back_to_light():
    yield
    ui.apply_theme("light")  # TOKENS is module state: never leak dark into later tests


def _window(qtbot):
    w = app.MainWindow()
    qtbot.addWidget(w)
    return w


def _sheets(root):
    return {id(x): x.styleSheet() for x in [root, *root.findChildren(QWidget)]}


class _Store:
    def summary(self, *a, **k):
        return []

    def recent(self, *a, **k):
        return []


def _usage(qtbot, w):
    d = UsageDialog(_Store(), w._settings, w)
    qtbot.addWidget(d)
    return d


def test_new_window_defaults_to_light(qtbot):
    w = _window(qtbot)
    assert not w.btn_theme.isChecked()
    assert ui.TOKENS["bg"] == ui.PALETTES["light"]["bg"]
    assert w.canvas.backgroundBrush().color().name() == ui.PALETTES["light"]["canvas"]


def test_unknown_saved_theme_falls_back_to_light(qtbot):
    s = app._open_settings()
    s.setValue(app.THEME_SETTINGS_KEY, "purple")
    assert app.restore_theme(s) == "light"
    assert not _window(qtbot).btn_theme.isChecked()


def test_toggle_rerenders_every_style(qtbot):
    w = _window(qtbot)
    w.carousel_panel.setVisible(True)
    light = _sheets(w)
    light_help = _sheets(ui.HelpGuideDialog(w))
    w.btn_theme.setChecked(True)
    assert ui.TOKENS["bg"] == ui.PALETTES["dark"]["bg"]
    assert w.canvas.backgroundBrush().color().name() == ui.PALETTES["dark"]["canvas"]
    assert w.styleSheet() == ui.APP_STYLE and ui.PALETTES["dark"]["bg"] in ui.APP_STYLE
    dark = _sheets(w)
    assert [k for k in light if k in dark and light[k] and light[k] == dark[k] and "#" in light[k]] == []
    h = ui.HelpGuideDialog(w)  # opened after the switch
    qtbot.addWidget(h)
    assert sorted(_sheets(h).values()) != sorted(light_help.values())
    for root in (w, h, _usage(qtbot, w)):
        for x in [root, *root.findChildren(QWidget)]:
            assert "$" not in x.styleSheet(), (type(x).__name__, x.objectName())
    assert ui.PALETTES["dark"]["bg"] in _usage(qtbot, w).styleSheet()


def test_toggle_keeps_layer_state(qtbot):
    from PIL import Image
    from companion.layers import ModificationLayer
    w = _window(qtbot)
    w.base_image = Image.new("RGB", (64, 48), (200, 200, 200))
    for i in range(2):
        layer = ModificationLayer(layer_id=str(i), name=f"L{i}", mask=Image.new("L", (64, 48), 0),
                                  inpainted_image=w.base_image, visible=i == 0)
        layer.update_thumbnail(w.base_image)
        w.modification_layers.append(layer)
    w.active_layer_index = 1
    w._refresh_layers_ui()
    old_thumb = w.modification_layers[0].thumbnail.cacheKey()
    w.btn_theme.setChecked(True)
    assert w.active_layer_index == 1 and [x.visible for x in w.modification_layers] == [True, False]
    assert len(w.layer_card_widgets) == 2
    card = w.layer_card_widgets[1]
    assert ui.PALETTES["dark"]["accent_tint"] in card.styleSheet()
    assert w.modification_layers[0].thumbnail.cacheKey() != old_thumb  # its TOKENS["control"] margin is redrawn


def test_choice_persists_across_windows(qtbot):
    w = _window(qtbot)
    w.btn_theme.setChecked(True)
    assert w._settings.value(app.THEME_SETTINGS_KEY) == "dark"
    w.close()
    ui.apply_theme("light")  # a fresh start: the module is light until the setting is read
    w2 = _window(qtbot)
    assert w2.btn_theme.isChecked() and ui.TOKENS["bg"] == ui.PALETTES["dark"]["bg"]
    assert w2.canvas.backgroundBrush().color().name() == ui.PALETTES["dark"]["canvas"]
    w2.btn_theme.setChecked(False)
    assert w2._settings.value(app.THEME_SETTINGS_KEY) == "light"


def test_shortcut_toggles_the_button(qtbot):
    w = _window(qtbot)
    assert w.shortcut_theme.key().toString() == "Ctrl+Shift+D"
    w.shortcut_theme.activated.emit()  # what the key press sends (the house pattern, see test_layer_undo)
    assert w.btn_theme.isChecked() and ui.TOKENS["bg"] == ui.PALETTES["dark"]["bg"]
    assert "Ctrl+Shift+D" in w.btn_theme.toolTip()


def test_colour_scheme_follows_the_theme(qtbot):
    hints = QApplication.instance().styleHints()
    if not hasattr(hints, "setColorScheme"):
        pytest.skip("Qt < 6.8")
    w = _window(qtbot)
    w.btn_theme.setChecked(True)
    assert hints.colorScheme() == Qt.ColorScheme.Dark
    w.btn_theme.setChecked(False)
    assert hints.colorScheme() == Qt.ColorScheme.Light
