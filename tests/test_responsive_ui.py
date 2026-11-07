"""Job 020: the top bar wraps instead of clipping, Erase / Save stay whole, layout survives a restart."""

import os
import sys

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QCheckBox, QPushButton

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import companion.app as app  # noqa: E402

SIZES = [(960, 540), (1280, 720), (1366, 768), (1920, 1080), (2560, 1440)]


@pytest.fixture
def ini(tmp_path, monkeypatch):
    path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(app, "_open_settings", lambda: QSettings(path, QSettings.Format.IniFormat))
    monkeypatch.setattr(QApplication, "quit", lambda *a: None)  # closeEvent quits the app
    return path


def _show(qtbot, size):
    w = app.MainWindow()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w.resize(*size)
    qtbot.wait(30)
    QApplication.processEvents()
    return w


def _assert_not_clipped(w):
    bar = w.top_bar.rect()
    for btn in (w.btn_erase, w.btn_sync):
        assert btn.isVisible()
        assert btn.width() >= btn.sizeHint().width(), btn.text()
    assert w.btn_sync.geometry().right() <= w.primary_group.width()
    assert w.primary_group.geometry().right() <= bar.right()
    for group in w.top_bar_groups:
        assert group.isVisible()
        g = group.geometry().translated(group.parentWidget().mapTo(w.top_bar, group.parentWidget().rect().topLeft()))
        assert bar.contains(g), (g, bar)
        for child in group.findChildren(QPushButton) + group.findChildren(QCheckBox):
            if child.isVisible():
                assert child.width() >= child.sizeHint().width(), child.text()
        for i in range(group.layout().count()):
            child = group.layout().itemAt(i).widget()
            if child is not None and child.isVisible():
                assert child.width() >= child.minimumSizeHint().width(), child
                assert child.geometry().right() <= group.width(), child
    # Groups never overlap each other (wrapped, not stacked)
    rects = [g.geometry() for g in w.top_bar_groups]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not a.intersects(b), (a, b)
    # Canvas starts below the (possibly multi-row) bar
    assert w.middle_splitter.geometry().top() >= w.top_bar.geometry().bottom()
    header = w.btn_new_layer
    assert header.isVisible()
    assert header.mapTo(w.layers_panel, header.rect().bottomRight()).x() <= w.layers_panel.width()
    assert w.btn_layer_undo.width() >= w.btn_layer_undo.sizeHint().width()


@pytest.mark.parametrize("size", SIZES)
def test_top_bar_fits_at_size(qtbot, ini, size):
    w = _show(qtbot, size)
    _assert_not_clipped(w)


def test_runtime_assistant_buttons_rewrap(qtbot, ini):
    w = _show(qtbot, (960, 540))
    for b in (w.btn_agent_cancel, w.btn_agent_apply, w.btn_agent_discard):
        b.setVisible(True)
    qtbot.wait(30)
    QApplication.processEvents()
    _assert_not_clipped(w)


def test_minimum_size_and_layers_toggle(qtbot, ini):
    w = _show(qtbot, (1366, 768))
    assert w.minimumWidth() == 960 and w.minimumHeight() == 540
    canvas_w = w.canvas.width()
    w.btn_layers_toggle.setChecked(False)
    QApplication.processEvents()
    assert not w.layers_panel.isVisible()
    assert w.canvas.width() > canvas_w
    w.btn_layers_toggle.setChecked(True)
    QApplication.processEvents()
    assert w.layers_panel.isVisible()
    assert not w.middle_splitter.childrenCollapsible()


def test_geometry_and_splitter_survive_restart(qtbot, ini):
    w = _show(qtbot, (1100, 700))
    w.middle_splitter.setSizes([700, 380])
    QApplication.processEvents()
    sizes = w.middle_splitter.sizes()
    w.close()
    stored = QSettings(ini, QSettings.Format.IniFormat)
    assert stored.value(app.WINDOW_GEOMETRY_KEY) and stored.value(app.WINDOW_SPLITTER_KEY)

    w2 = app.MainWindow()  # no resize: the size must come from the ini
    qtbot.addWidget(w2)
    avail = w2.screen().availableGeometry()
    if avail.width() >= 1100 and avail.height() >= 700:
        assert (w2.width(), w2.height()) == (1100, 700)
    w2.show()
    qtbot.waitExposed(w2)
    QApplication.processEvents()
    assert w2.middle_splitter.sizes()[1] == sizes[1]


def test_oversized_geometry_is_clamped_to_screen(qtbot, ini):
    w = app.MainWindow()
    qtbot.addWidget(w)
    avail = w.screen().availableGeometry()
    w.setGeometry(avail.x() + avail.width() - 50, avail.y(), avail.width() + 600, avail.height() + 400)
    QSettings(ini, QSettings.Format.IniFormat).setValue(app.WINDOW_GEOMETRY_KEY, w.saveGeometry())

    w2 = app.MainWindow()
    qtbot.addWidget(w2)
    g = w2.geometry()
    avail = w2.screen().availableGeometry()  # restoreGeometry may pick another monitor
    assert g.width() <= max(avail.width(), w2.minimumWidth())
    assert g.height() <= max(avail.height(), w2.minimumHeight())
    assert avail.contains(g.topLeft())


def test_corrupt_settings_are_ignored(qtbot, ini):
    s = QSettings(ini, QSettings.Format.IniFormat)
    s.setValue(app.WINDOW_GEOMETRY_KEY, b"garbage")
    s.setValue(app.WINDOW_SPLITTER_KEY, b"\x00\x01junk")
    s.sync()
    w = app.MainWindow()
    qtbot.addWidget(w)
    assert (w.width(), w.height()) == (1400, 920)
