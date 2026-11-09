"""
ui.py
Efface Magique LR - User Interface Components, Styling, and Overlays

Provides:
- Light/dark palettes, app stylesheet (APP_STYLE) and apply_theme() and application brand assets
- LoadingSpinnerOverlay: Smooth hardware-rendered loading spinner overlay for async tasks
- HelpGuideDialog: Rich interactive User Guide, Installation, Shortcuts, and Packaging dialog
- FlowLayout: left-to-right layout that wraps whole items onto new rows (responsive top bar)
"""

import os
import sys
from string import Template
from typing import Optional

from PyQt6.QtCore import Qt, QPoint, QRect, QSize, QTimer, QUrl
from PyQt6.QtGui import (
    QIcon,
    QColor,
    QPainter,
    QPen,
    QBrush,
    QPixmap,
    QDesktopServices,
)
from PyQt6.QtWidgets import (
    QWidget,
    QLayout,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QDialog,
    QTabWidget,
    QTextBrowser,
    QApplication,
)

from companion import __version__
from companion.diagnostics import build_diagnostics

# Application Brand Assets
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
_ASSETS_DIR = os.path.join(_CURRENT_DIR, "assets")
_CHECKMARK_PATH = os.path.join(_ASSETS_DIR, "checkmark.png").replace("\\", "/")
_LOGO_PATH = os.path.join(_ASSETS_DIR, "logo.png").replace("\\", "/")
_LOGO_ICO_PATH = os.path.join(_ASSETS_DIR, "logo.ico").replace("\\", "/")


# -----------------------------------------------------------------------------
# Light Theme Stylesheet
# -----------------------------------------------------------------------------

# Design tokens: the one place for chrome colours, spacing, radii and type sizes.
# Greys are neutral (R=G=B) so the chrome never tints how the photo's colours are judged;
# the accent is kept for the primary action, checked toggles, focus and selection.
PALETTES = {
    "light": {
        "bg": "#ffffff",              # window ground, dialogs, layer panel
        "surface": "#f7f7f7",         # top bar, assistant bar, popups, tooltips
        "control": "#ececec",         # secondary button / combo face
        "control_hover": "#e0e0e0",
        "control_pressed": "#d4d4d4",
        "control_checked": "#dadada",  # checked toggle: a darker grey plus an accent edge
        "field": "#ffffff",           # text fields
        "border": "#c8c8c8",
        "border_strong": "#a0a0a0",
        "status": "#f0f0f0",          # status bar
        "canvas": "#d4d4d4",          # photo surround: grey, not white, so a high-key photo's edge stays visible
        "text": "#1f1f1f",
        "text_muted": "#595959",
        "text_disabled": "#a0a0a0",   # exempt from WCAG contrast
        "on_accent": "#ffffff",
        "accent": "#0a64c0",          # fills and borders
        "accent_text": "#0a64c0",     # accent used as a text colour
        "accent_hover": "#1570cc",
        "accent_pressed": "#0b5aa8",
        "accent_disabled": "#c5d8ee",
        "accent_tint": "#e6f0fa",     # active layer card / editing banner / checked variation
        "focus": "#1a73e8",
        "success": "#17693a",         # Save to Lightroom label / Live badge text
        "success_border": "#3a9a62",
        "warning": "#8f4f00",
        "danger": "#c62828",          # delete-layer hover fill
    },
    "dark": {
        "bg": "#1e1e1e",
        "surface": "#262626",
        "control": "#333333",
        "control_hover": "#3d3d3d",
        "control_pressed": "#1f1f1f",
        "control_checked": "#474747",  # checked toggle: a lighter grey plus an accent edge
        "field": "#1f1f1f",
        "border": "#424242",
        "border_strong": "#5a5a5a",
        "status": "#181818",
        "canvas": "#181818",
        "text": "#e8e8e8",
        "text_muted": "#a8a8a8",
        "text_disabled": "#666666",
        "on_accent": "#ffffff",
        "accent": "#0a64c0",
        "accent_text": "#5ab0ff",     # #0a64c0 is under 3:1 as text on the dark greys
        "accent_hover": "#1570cc",
        "accent_pressed": "#0b5aa8",
        "accent_disabled": "#2a3440",
        "accent_tint": "#15283c",
        "focus": "#5ab0ff",
        "success": "#6fcf97",
        "success_border": "#2f8f5b",
        "warning": "#f0a030",
        "danger": "#d32f2f",
    },
}
THEMES = tuple(PALETTES)
DEFAULT_THEME = "light"

TOKENS = {
    **PALETTES[DEFAULT_THEME],
    # Spacing scale (px, 4-pt steps)
    "sp1": "4px", "sp2": "8px", "sp3": "12px", "sp4": "16px",
    # Radii
    "r_sm": "3px", "r_md": "4px",
    # Font sizes
    "fs_sm": "11px", "fs_md": "12px", "fs_lg": "13px",
}

def qss(text: str) -> str:
    """Fill $token placeholders in a stylesheet from TOKENS."""
    return Template(text).substitute(TOKENS)


APP_STYLE_TEMPLATE = """
QMainWindow, QDialog {
    background-color: $bg;
    color: $text;
}
QWidget {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
QToolBar {
    background-color: $surface;
    border-bottom: 1px solid $border;
    padding: 6px 10px;
    spacing: $sp2;
}
QStatusBar {
    background-color: $status;
    border-top: 1px solid $border;
    color: $text_muted;
    font-size: $fs_sm;
}
QPushButton {
    background-color: $control;
    color: $text;
    border: 1px solid $border;
    border-radius: $r_md;
    padding: 5px $sp3;
    font-size: $fs_md;
    font-weight: 500;
}
QPushButton:hover {
    background-color: $control_hover;
    border-color: $border_strong;
}
QPushButton:pressed {
    background-color: $control_pressed;
}
QPushButton:checked {
    background-color: $control_checked;
    border-color: $accent;
    color: $text;
}
QPushButton:focus {
    border-color: $focus;
}
QPushButton:disabled {
    background-color: $surface;
    color: $text_disabled;
    border-color: $border;
}
QPushButton#primaryAction {
    background-color: $accent;
    color: $on_accent;
    font-weight: bold;
    border: 1px solid $accent;
    padding: 6px $sp4;
}
QPushButton#primaryAction:hover {
    background-color: $accent_hover;
    border-color: $accent_hover;
}
QPushButton#primaryAction:pressed {
    background-color: $accent_pressed;
}
QPushButton#primaryAction:focus {
    border-color: $focus;
}
QPushButton#primaryAction:disabled {
    background-color: $accent_disabled;
    color: $text_disabled;
    border-color: $accent_disabled;
}
QPushButton#saveAction, QPushButton#syncAction {
    color: $success;
    font-weight: bold;
    border: 1px solid $success_border;
    padding: 6px $sp4;
}
QPushButton#saveAction:hover, QPushButton#syncAction:hover {
    background-color: $control_hover;
    border-color: $success;
}
QPushButton#saveAction:pressed, QPushButton#syncAction:pressed {
    background-color: $control_pressed;
}
QPushButton#saveAction:focus, QPushButton#syncAction:focus {
    border-color: $focus;
}
QPushButton#saveAction:disabled, QPushButton#syncAction:disabled {
    color: $text_disabled;
    border-color: $border;
}
QComboBox {
    background-color: $control;
    color: $text;
    border: 1px solid $border;
    border-radius: $r_md;
    padding: 4px $sp2;
    font-size: $fs_md;
}
QComboBox:hover {
    background-color: $control_hover;
    border-color: $border_strong;
}
QComboBox:pressed, QComboBox:on {
    background-color: $control_pressed;
}
QComboBox:focus {
    border-color: $focus;
}
QComboBox:disabled {
    background-color: $surface;
    color: $text_disabled;
    border-color: $border;
}
QComboBox QAbstractItemView {
    background-color: $surface;
    color: $text;
    selection-background-color: $accent;
    selection-color: $on_accent;
    border: 1px solid $border;
}
QLineEdit {
    background-color: $field;
    color: $text;
    border: 1px solid $border;
    border-radius: $r_md;
    padding: 4px $sp2;
    font-size: $fs_md;
    selection-background-color: $accent;
    selection-color: $on_accent;
}
QLineEdit:hover {
    border-color: $border_strong;
}
QLineEdit:pressed, QLineEdit:focus {
    border-color: $focus;
}
QLineEdit:disabled {
    background-color: $surface;
    color: $text_disabled;
    border-color: $border;
}
QSlider::groove:horizontal {
    border: 1px solid $border;
    height: 4px;
    background: $field;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: $text_muted;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: $bg;
    border: 1px solid $border_strong;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: $control_hover;
}
QSlider::handle:horizontal:focus {
    border-color: $focus;
}
QProgressBar {
    border: 1px solid $border;
    border-radius: $r_sm;
    text-align: center;
    color: $text;
    background-color: $field;
    height: 16px;
    font-size: 10px;
}
QProgressBar::chunk {
    background-color: $text_muted;
    border-radius: 2px;
}
QCheckBox {
    color: $text;
    font-size: $fs_md;
    spacing: 6px;
}
QCheckBox:disabled {
    color: $text_disabled;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid $border_strong;
    border-radius: $r_sm;
    background-color: $field;
}
QCheckBox::indicator:hover {
    border-color: $text_muted;
    background-color: $control;
}
QCheckBox::indicator:pressed {
    background-color: $control_pressed;
}
QCheckBox::indicator:focus {
    border-color: $focus;
}
QCheckBox::indicator:checked {
    background-color: $accent;
    border-color: $accent;
    image: url("__CHECKMARK_URL__");
}
QCheckBox::indicator:checked:hover {
    background-color: $accent_hover;
    border-color: $accent_hover;
    image: url("__CHECKMARK_URL__");
}
QCheckBox::indicator:checked:focus {
    border-color: $focus;
}
QCheckBox::indicator:disabled {
    border-color: $border;
    background-color: $surface;
}
QLabel {
    color: $text_muted;
    font-size: $fs_md;
}
QToolTip {
    background-color: $surface;
    color: $text;
    border: 1px solid $border_strong;
}
QMessageBox {
    background-color: $bg;
}
QMessageBox QLabel {
    color: $text;
}
QSpinBox, QTableView {
    background-color: $field;
    color: $text;
    border: 1px solid $border;
    selection-background-color: $accent;
    selection-color: $on_accent;
}
QHeaderView::section {
    background-color: $surface;
    color: $text;
    border: none;
    border-right: 1px solid $border;
    border-bottom: 1px solid $border;
    padding: 2px $sp1;
}
QScrollBar:vertical {
    background: $surface;
    width: 12px;
    margin: 0;
}
QScrollBar:horizontal {
    background: $surface;
    height: 12px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: $border;
    min-height: 24px;
    border-radius: $r_md;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: $border;
    min-width: 24px;
    border-radius: $r_md;
    margin: 2px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: $border_strong;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: none;
}
""".replace("__CHECKMARK_URL__", _CHECKMARK_PATH)
APP_STYLE = qss(APP_STYLE_TEMPLATE)  # the active palette's app stylesheet; apply_theme() rebuilds it

_HELP_PAGE_STYLE = "background-color: $bg; color: $text; border: none; padding: 12px; font-size: 12px;"

_TEMPLATE_PROP = "_qss_template"


def styled(widget: QWidget, template: str) -> None:
    """Set a $token stylesheet on a widget and remember the template so apply_theme() can re-render it."""
    widget.setProperty(_TEMPLATE_PROP, template)
    widget.setStyleSheet(qss(template))


def apply_theme(name: str) -> str:
    """Switch every token-built style to PALETTES[name] (unknown names fall back to light); returns the name used."""
    global APP_STYLE
    if name not in PALETTES:
        name = DEFAULT_THEME
    TOKENS.update(PALETTES[name])
    APP_STYLE = qss(APP_STYLE_TEMPLATE)
    app = QApplication.instance()
    if app is None:
        return name
    if hasattr(app.styleHints(), "setColorScheme"):  # Qt 6.8+: native dialogs and message boxes follow the theme
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark if name == "dark" else Qt.ColorScheme.Light)
    if app.styleSheet():  # main() sets it; tests leave it empty
        app.setStyleSheet(APP_STYLE)
    for w in app.allWidgets():
        template = w.property(_TEMPLATE_PROP)
        if template:
            w.setStyleSheet(qss(template))
    for w in app.topLevelWidgets():
        hook = getattr(w, "on_theme_applied", None)  # painted surfaces and rebuilt cards (MainWindow)
        if callable(hook):
            hook()
    for w in app.allWidgets():
        w.update()  # canvas viewport, spinner, HUD and toast read TOKENS when they paint
    return name


# -----------------------------------------------------------------------------
# Smooth Loading Spinner Overlay
# -----------------------------------------------------------------------------

class FlowLayout(QLayout):
    """Port of Qt's FlowLayout example: items go left to right and wrap to a new row when full."""

    def __init__(self, parent=None, spacing: int = 6, reserve_height: bool = False):
        super().__init__(parent)
        self._items = []
        self._reserve_height = reserve_height
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        needed = self._do_layout(rect, test_only=False)
        # A short window would squeeze the bar below its wrapped height; reserve the rows instead
        host = self.parentWidget()
        if self._reserve_height and host is not None and host.minimumHeight() != needed:
            host.setMinimumHeight(needed)

    def sizeHint(self):
        # Everything on one row; minimumSize() is the narrowest it can wrap to
        items = [i for i in self._items if not i.isEmpty()]
        size = QSize(sum(i.sizeHint().width() for i in items) + self.spacing() * max(0, len(items) - 1),
                     max((i.sizeHint().height() for i in items), default=0))
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_height = r.x(), r.y(), 0
        row = []  # (item, x, w, h) of the current row, placed once its height is known

        def flush():
            # Centre each item vertically in its row, like QHBoxLayout does
            if not test_only:
                for item, ix, iw, ih in row:
                    item.setGeometry(QRect(QPoint(ix, y + (line_height - ih) // 2), QSize(iw, ih)))
            row.clear()

        for item in self._items:
            if item.isEmpty():  # hidden widgets take no room
                continue
            hint = item.sizeHint()
            if x + hint.width() > r.right() + 1 and line_height > 0:
                flush()
                x, y, line_height = r.x(), y + line_height + self.spacing(), 0
            # An item wider than the whole row gets the row width (a nested FlowLayout wraps inside it)
            w = min(hint.width(), max(item.minimumSize().width(), r.right() + 1 - x))
            h = item.heightForWidth(w) if item.hasHeightForWidth() else hint.height()
            row.append((item, x, w, h))
            x += w + self.spacing()
            line_height = max(line_height, h)
        flush()
        return y + line_height - rect.y() + m.bottom()


class LoadingSpinnerOverlay(QWidget):
    """
    Hardware-rendered semi-transparent overlay with a smooth spinning indicator
    and message badge. Displayed during asynchronous AI inpainting or image loading.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._angle: float = 0.0
        self._message: str = "Processing..."
        self._subtext: str = ""
        
        self._timer = QTimer(self)
        self._timer.setInterval(25)  # 40 FPS smooth rotation
        self._timer.timeout.connect(self._rotate)
        
        if parent is not None:
            parent.installEventFilter(self)
            self.resize(parent.size())
            
        self.hide()

    def _rotate(self):
        self._angle = (self._angle + 12.0) % 360.0
        self.update()

    def start(self, message: str = "Processing...", subtext: str = ""):
        """Show overlay and start spinner animation."""
        self._message = message
        self._subtext = subtext
        if self.parentWidget() is not None:
            self.resize(self.parentWidget().size())
            self.raise_()
        self.show()
        if not self._timer.isActive():
            self._timer.start()

    def set_message(self, message: str, subtext: str = ""):
        """Update displayed progress message."""
        self._message = message
        if subtext:
            self._subtext = subtext
        self.update()

    def stop(self):
        """Stop animation and hide overlay."""
        self._timer.stop()
        self.hide()

    def eventFilter(self, obj, event):
        if obj == self.parentWidget() and event.type() == event.Type.Resize:
            self.resize(self.parentWidget().size())
        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent light scrim background
        scrim = QColor(TOKENS["bg"])
        scrim.setAlpha(175)
        painter.fillRect(self.rect(), scrim)

        cx = self.width() / 2.0
        cy = self.height() / 2.0 - 15.0

        # Draw card container
        card_w = 320.0
        card_h = 130.0
        card_rect = QRect(int(cx - card_w / 2.0), int(cy - 40.0), int(card_w), int(card_h))
        painter.setBrush(QBrush(QColor(TOKENS["surface"])))
        painter.setPen(QPen(QColor(TOKENS["accent"]), 1.5))
        painter.drawRoundedRect(card_rect, 10, 10)

        # Draw rotating spinner arc
        spinner_radius = 22.0
        spinner_rect = QRect(
            int(cx - spinner_radius),
            int(cy - 20.0 - spinner_radius),
            int(spinner_radius * 2),
            int(spinner_radius * 2),
        )

        # Background track
        painter.setPen(QPen(QColor(TOKENS["border"]), 3.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(spinner_rect)

        # Rotating active arc
        painter.save()
        painter.setPen(QPen(QColor(TOKENS["accent"]), 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        start_angle = int(self._angle * 16)
        span_angle = int(90 * 16)
        painter.drawArc(spinner_rect, start_angle, span_angle)
        painter.restore()

        # Draw progress text
        painter.setPen(QColor(TOKENS["text"]))
        font = painter.font()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        text_rect = QRect(int(cx - card_w / 2.0), int(cy + 18.0), int(card_w), 24)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._message)

        if self._subtext:
            painter.setPen(QColor(TOKENS["text_muted"]))
            font.setPointSize(9)
            font.setBold(False)
            painter.setFont(font)
            sub_rect = QRect(int(cx - card_w / 2.0), int(cy + 44.0), int(card_w), 20)
            painter.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, self._subtext)

        painter.end()


# -----------------------------------------------------------------------------
# User Guide & Setup Dialog
# -----------------------------------------------------------------------------

class HelpGuideDialog(QDialog):
    """Rich interactive User Guide, Installation, Shortcuts, and Distribution packaging dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Efface Magique LR - User Guide & Setup")
        self.resize(780, 600)
        if os.path.isfile(_LOGO_PATH):
            self.setWindowIcon(QIcon(_LOGO_PATH))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Header banner
        header = QHBoxLayout()
        if os.path.isfile(_LOGO_PATH):
            logo_lbl = QLabel()
            pix = QPixmap(_LOGO_PATH).scaled(56, 56, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(pix)
            logo_lbl.setStyleSheet("margin-right: 8px;")
            header.addWidget(logo_lbl)

        header_text = QVBoxLayout()
        title = QLabel("Efface Magique LR (LaMa Variations Eraser)")
        styled(title, "font-size: 18px; font-weight: bold; color: $text;")
        subtitle = QLabel("Local, private AI inpainting companion for Adobe Lightroom Classic")
        styled(subtitle, "font-size: 12px; color: $accent_text; font-weight: 500;")
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        self.version_label = QLabel(f"Version {__version__}")
        styled(self.version_label, "font-size: 11px; color: $text_muted;")
        header_text.addWidget(self.version_label)
        header.addLayout(header_text)
        header.addStretch(1)
        layout.addLayout(header)

        # Tabs
        self.tabs = QTabWidget(self)
        styled(self.tabs, """
            QTabWidget::pane {
                border: 1px solid $border;
                background-color: $bg;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: $control;
                color: $text_muted;
                padding: 8px 18px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 500;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background-color: $bg;
                color: $text;
                border-bottom: 2px solid $accent;
            }
            QTabBar::tab:hover {
                color: $text;
            }
        """)

        # Tab 1: Installation
        tab_install = QTextBrowser()
        tab_install.setOpenExternalLinks(True)
        styled(tab_install, _HELP_PAGE_STYLE)
        tab_install.setHtml(qss("""
            <h2 style="color: $success; margin-top: 0;">🚀 Installation &amp; Setup Guide</h2>
            <p>Efface Magique LR runs 100% locally and privately on your machine, integrating natively with Adobe Lightroom Classic.</p>
            
            <h3 style="color: $accent_text;">Step 1: Install Dependencies (1-Click)</h3>
            <ul>
                <li><b>Windows:</b> Double-click <code>install.bat</code> in the app directory, or run it in Command Prompt / PowerShell.</li>
                <li><b>macOS / Linux:</b> Open Terminal in the app folder and run <code>chmod +x install.sh && ./install.sh</code>.</li>
            </ul>
            <p style="color: $text_muted;"><i>The installer automatically configures a local Python virtual environment (<code>.venv</code>) with all AI PyTorch and Simple-LaMa dependencies.</i></p>

            <h3 style="color: $accent_text;">Step 2: Add the Plugin to Adobe Lightroom Classic</h3>
            <ol>
                <li>Open <b>Adobe Lightroom Classic</b>.</li>
                <li>Open the Plug-in Manager by going to: <b>File &gt; Plug-in Manager...</b> (or press <code>Ctrl + Alt + Shift + ,</code> on Windows / <code>Cmd + Opt + Shift + ,</code> on Mac).</li>
                <li>Click the <b>Add</b> button at the bottom-left corner of the dialog.</li>
                <li>Select the plugin folder: <br/><code style="background-color: $surface; padding: 2px 6px; color: $warning;">plugin/ai_eraser.lrplugin</code></li>
                <li>Verify that the status dot is <b style="color: $success;">🟢 Installed and running</b>.</li>
                <li>Click <b>Done</b>. The plugin is now active!</li>
            </ol>
        """))
        self.tabs.addTab(tab_install, "🚀 Installation")

        # Tab 2: How to Use
        tab_usage = QTextBrowser()
        tab_usage.setOpenExternalLinks(True)
        styled(tab_usage, _HELP_PAGE_STYLE)
        tab_usage.setHtml(qss("""
            <h2 style="color: $success; margin-top: 0;">✨ How to Use Efface Magique LR</h2>

            <h3 style="color: $accent_text;">⚡ Quick Start Guide</h3>
            <ol>
                <li>In Lightroom Classic, select: <b>File &gt; Plug-in Extras &gt; ⚡ AI Generative Eraser (Live Window)...</b></li>
                <li>The companion window appears displaying your active photo.</li>
                <li><b>Switching photos in Lightroom</b> (arrow keys or clicking the filmstrip) instantly syncs the new photo into this window with zero lag!</li>
                <li>Paint over tourists, wires, power poles, or blemishes using the red brush (use <code>[</code> and <code>]</code> to adjust size).</li>
                <li>Toggle <b>🎯 Subject</b> to automatically lock contours to object boundaries.</li>
                <li>Click <b>✨ Erase Object</b> (or press <b>Enter</b>) to generate inpainting.</li>
                <li>Review the 3 candidate variation cards and pick your favorite result.</li>
                <li>Click <b>📥 Save &amp; Sync to Lightroom</b> (<code>Ctrl + S</code>) — the image is saved as a TIFF (untouched pixels keep full 16-bit precision; the erased area is computed in 8-bit and merged back; preview is 8-bit) and auto-stacked into your Lightroom catalog!</li>
                <li>Click <b>📌 Pin</b> (<code>Ctrl + T</code>) to keep the companion floating on top of Lightroom while you work.</li>
            </ol>

            <h3 style="color: $accent_text;">Assistant (optional)</h3>
            <p>Install <code>requirements-agent.txt</code>, pick a provider (ollama, anthropic, openai, gemini) and a model
            name, then type a request such as <i>remove the person on the left</i> and press <b>Ask</b>.
            The assistant detects objects, proposes an erase and shows a preview; nothing is added to your layers
            until you click <b>Apply</b> (<b>Discard</b> throws it away, <b>Stop</b> cancels a running request).</p>
            <p><b>Limit:</b> only these object classes can be located: person, car, bus, bicycle, motorbike, aeroplane,
            boat, train, bottle, chair, sofa, diningtable, pottedplant, tvmonitor, bird, cat, dog, horse, sheep, cow.
            Anything else (a bin, a sign, a wire) is usually <b>not found</b>; paint over it with the brush instead.</p>
            <p><b>Privacy:</b> only your request text and region labels are sent to the model; images, file names and
            EXIF are never sent. Ollama on localhost stays on this computer; Anthropic, OpenAI and Gemini are cloud
            services. API keys are read only from ANTHROPIC_API_KEY, OPENAI_API_KEY and GOOGLE_API_KEY.
            <b>Usage</b> shows tokens and cost per call.</p>
        """))
        self.tabs.addTab(tab_usage, "✨ How to Use")

        # Tab 3: Shortcuts
        tab_shortcuts = QTextBrowser()
        styled(tab_shortcuts, _HELP_PAGE_STYLE)
        tab_shortcuts.setHtml(qss("""
            <h2 style="color: $success; margin-top: 0;">⌨ Keyboard Shortcuts Reference</h2>
            <table width="100%" cellpadding="6" cellspacing="0" style="border-collapse: collapse; border: 1px solid $border;">
                <tr style="background-color: $surface; color: $text; border-bottom: 2px solid $accent;">
                    <th align="left" style="padding: 6px 10px;">Shortcut</th>
                    <th align="left" style="padding: 6px 10px;">Action</th>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>Ctrl + S</code> / <code>Ctrl + Shift + S</code></td>
                    <td><b>⚡ Save &amp; Sync to Lightroom</b> (Auto-stacks into catalog &amp; stays open)</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>Enter</code> / <code>Return</code></td>
                    <td><b>✨ Erase Object</b> (Run AI inpainting on marked areas)</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>Right mouse drag</code></td>
                    <td>Remove painted mask (eraser brush while held)</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>Ctrl + T</code></td>
                    <td><b>📌 Toggle Pin (Always on Top)</b> floating above Lightroom</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>[</code> / <code>]</code></td>
                    <td>Decrease / Increase Brush Radius</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>Ctrl + Z</code></td>
                    <td>↶ Undo stroke or inpainting step</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>Ctrl + Y</code> / <code>Ctrl + Shift + Z</code></td>
                    <td>↷ Redo stroke or inpainting step</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>Spacebar</code> (Hold) / <code>\\</code></td>
                    <td>👁 Instant Before / After Comparison</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>Y</code></td>
                    <td>◫ Interactive Split-Screen Comparison Slider</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>Ctrl + 0</code> / <code>F</code></td>
                    <td>🔍 Fit photo to screen view</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>Ctrl + 1</code></td>
                    <td>1:1 View at 100% pixel scale</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>Ctrl + Shift + D</code></td>
                    <td>Switch between light and dark mode (remembered)</td>
                </tr>
                <tr style="border-bottom: 1px solid $border;">
                    <td style="padding: 6px 10px;"><code>F1</code></td>
                    <td>💡 Open this User Guide &amp; Setup Dialog</td>
                </tr>
            </table>
        """))
        self.tabs.addTab(tab_shortcuts, "⌨ Shortcuts")

        # Tab 4: Download & Share
        tab_share_widget = QWidget()
        share_vbox = QVBoxLayout(tab_share_widget)
        share_vbox.setContentsMargins(14, 14, 14, 14)
        share_vbox.setSpacing(14)

        share_text = QTextBrowser()
        share_text.setOpenExternalLinks(True)
        styled(share_text, "background-color: transparent; color: $text; border: none; font-size: 12px;")
        share_text.setHtml(qss("""
            <h2 style="color: $success; margin-top: 0;">📥 Easy Download &amp; Share for Other Users</h2>
            <p>You can easily distribute Efface Magique LR to other photographers or friends without requiring Git:</p>
            <ol>
                <li>Click the button below to package a clean, portable <b>Efface-Magique-LR.zip</b> archive.</li>
                <li>Share this ZIP file with any colleague or photographer via Google Drive, Dropbox, USB drive, or GitHub Release.</li>
                <li>The recipient simply unzips the folder and double-clicks <code>install.bat</code> (Windows) or <code>./install.sh</code> (macOS/Linux).</li>
                <li>In Lightroom, they add <code>plugin/ai_eraser.lrplugin</code> and they are ready to edit!</li>
            </ol>
        """))
        share_vbox.addWidget(share_text)

        btn_package = QPushButton("📦 Package Distribution ZIP for Other Users")
        styled(btn_package, """
            QPushButton {
                background-color: $accent;
                color: $on_accent;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: $accent_hover;
            }
        """)
        btn_package.clicked.connect(self._on_package_distribution)
        share_vbox.addWidget(btn_package)
        share_vbox.addStretch(1)

        self.tabs.addTab(tab_share_widget, "📥 Download & Share")
        layout.addWidget(self.tabs)

        # Bottom button row
        bottom_row = QHBoxLayout()
        btn_readme = QPushButton("📖 Open Full README.md")
        btn_readme.setToolTip("Open complete repository README document in default viewer")
        btn_readme.clicked.connect(self._on_open_readme)
        bottom_row.addWidget(btn_readme)

        self.btn_copy_diagnostics = QPushButton("📋 Copy diagnostics")
        self.btn_copy_diagnostics.setToolTip("Copy versions, device, bridge status and the last error for a bug report.\nThe last error can include a full file path; check it before posting publicly.")
        self.btn_copy_diagnostics.clicked.connect(self._on_copy_diagnostics)
        bottom_row.addWidget(self.btn_copy_diagnostics)

        bottom_row.addStretch(1)

        btn_close = QPushButton("Close")
        btn_close.setFixedWidth(90)
        btn_close.clicked.connect(self.accept)
        bottom_row.addWidget(btn_close)
        layout.addLayout(bottom_row)

    def _on_package_distribution(self):
        try:
            from package_release import create_release_zip
            zip_path = create_release_zip()
            if sys.platform == "win32" and os.path.isfile(zip_path):
                import subprocess
                subprocess.Popen(f'explorer /select,"{os.path.abspath(zip_path)}"')
            QMessageBox.information(
                self,
                "Package Ready",
                f"Successfully created distribution package:\n\n{zip_path}\n\nOther users can unzip this archive and run install.bat to get started!"
            )
        except Exception as e:
            QMessageBox.critical(self, "Packaging Error", f"Failed to create release ZIP:\n{e}")

    def _on_copy_diagnostics(self):
        provider = getattr(self.parent(), "diagnostics_text", None)
        text = provider() if callable(provider) else build_diagnostics({"app_version": __version__})
        QApplication.clipboard().setText(text)

    def _on_open_readme(self):
        readme_path = os.path.join(_PROJECT_ROOT, "README.md")
        if os.path.isfile(readme_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(readme_path))
        else:
            QMessageBox.warning(self, "Not Found", f"README.md not found at:\n{readme_path}")
