"""
ui.py
Efface Magique LR - User Interface Components, Styling, and Overlays

Provides:
- Dark theme stylesheet (DARK_STYLE) and application brand assets
- LoadingSpinnerOverlay: Smooth hardware-rendered loading spinner overlay for async tasks
- HelpGuideDialog: Rich interactive User Guide, Installation, Shortcuts, and Packaging dialog
"""

import os
import sys
from typing import Optional

from PyQt6.QtCore import Qt, QRect, QTimer, QSize, QUrl, QPointF
from PyQt6.QtGui import (
    QIcon,
    QFont,
    QColor,
    QPainter,
    QPen,
    QBrush,
    QPixmap,
    QDesktopServices,
)
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QDialog,
    QTabWidget,
    QTextBrowser,
)

# Application Brand Assets
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
_ASSETS_DIR = os.path.join(_CURRENT_DIR, "assets")
_CHECKMARK_PATH = os.path.join(_ASSETS_DIR, "checkmark.png").replace("\\", "/")
_LOGO_PATH = os.path.join(_ASSETS_DIR, "logo.png").replace("\\", "/")
_LOGO_ICO_PATH = os.path.join(_ASSETS_DIR, "logo.ico").replace("\\", "/")


# -----------------------------------------------------------------------------
# Dark Theme Stylesheet
# -----------------------------------------------------------------------------

DARK_STYLE = """
QMainWindow {
    background-color: #1e1e1e;
    color: #e0e0e0;
}
QWidget {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
QToolBar {
    background-color: #252526;
    border-bottom: 1px solid #333333;
    padding: 6px 10px;
    spacing: 8px;
}
QStatusBar {
    background-color: #181818;
    border-top: 1px solid #2d2d2d;
    color: #999999;
    font-size: 11px;
}
QPushButton {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #383838;
    border-color: #4a4a4a;
}
QPushButton:pressed {
    background-color: #202020;
}
QPushButton:checked {
    background-color: #0078d4;
    border-color: #0086f0;
    color: #ffffff;
    font-weight: bold;
}
QPushButton:disabled {
    background-color: #242424;
    color: #555555;
    border-color: #303030;
}
QPushButton#primaryAction {
    background-color: #0078d4;
    color: #ffffff;
    font-weight: bold;
    border: 1px solid #0086f0;
    padding: 7px 18px;
    border-radius: 4px;
}
QPushButton#primaryAction:hover {
    background-color: #1084d8;
}
QPushButton#primaryAction:disabled {
    background-color: #1b354b;
    color: #6688aa;
    border-color: #23425d;
}
QPushButton#saveAction, QPushButton#syncAction {
    background-color: #107c41;
    color: #ffffff;
    font-weight: bold;
    border: 1px solid #169b52;
    padding: 7px 18px;
    border-radius: 4px;
}
QPushButton#saveAction:hover, QPushButton#syncAction:hover {
    background-color: #148f4b;
}
QComboBox {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 500;
}
QComboBox:hover {
    border-color: #0078d4;
}
QComboBox QAbstractItemView {
    background-color: #252526;
    color: #ffffff;
    selection-background-color: #0078d4;
    selection-color: #ffffff;
    border: 1px solid #3c3c3c;
}
QLineEdit {
    background-color: #262626;
    color: #ffffff;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}
QLineEdit:focus {
    border: 1px solid #0078d4;
}
QSlider::groove:horizontal {
    border: 1px solid #333333;
    height: 4px;
    background: #252526;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #0078d4;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 1px solid #555555;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #0078d4;
    border-color: #0086f0;
}
QProgressBar {
    border: 1px solid #333333;
    border-radius: 3px;
    text-align: center;
    color: #ffffff;
    background-color: #1a1a1a;
    height: 16px;
    font-size: 10px;
}
QProgressBar::chunk {
    background-color: #0078d4;
    border-radius: 2px;
}
QCheckBox {
    color: #cccccc;
    font-size: 12px;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #444444;
    border-radius: 3px;
    background-color: #262626;
}
QCheckBox::indicator:hover {
    border-color: #0078d4;
    background-color: #2f2f2f;
}
QCheckBox::indicator:checked {
    background-color: #0078d4;
    border-color: #0086f0;
    image: url("__CHECKMARK_URL__");
}
QCheckBox::indicator:checked:hover {
    background-color: #1084d8;
    border-color: #1a94e8;
    image: url("__CHECKMARK_URL__");
}
QCheckBox::indicator:disabled {
    border-color: #383838;
    background-color: #1a1a1a;
}
QLabel {
    color: #cccccc;
    font-size: 12px;
}
""".replace("__CHECKMARK_URL__", _CHECKMARK_PATH)


# -----------------------------------------------------------------------------
# Smooth Loading Spinner Overlay
# -----------------------------------------------------------------------------

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

        # Semi-transparent dark scrim background
        painter.fillRect(self.rect(), QColor(18, 18, 18, 175))

        cx = self.width() / 2.0
        cy = self.height() / 2.0 - 15.0

        # Draw card container
        card_w = 320.0
        card_h = 130.0
        card_rect = QRect(int(cx - card_w / 2.0), int(cy - 40.0), int(card_w), int(card_h))
        painter.setBrush(QBrush(QColor(28, 28, 30, 235)))
        painter.setPen(QPen(QColor(0, 120, 212, 180), 1.5))
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
        painter.setPen(QPen(QColor(50, 50, 55, 200), 3.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(spinner_rect)

        # Rotating active arc
        painter.save()
        painter.setPen(QPen(QColor(0, 150, 255), 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        start_angle = int(self._angle * 16)
        span_angle = int(90 * 16)
        painter.drawArc(spinner_rect, start_angle, span_angle)
        painter.restore()

        # Draw progress text
        painter.setPen(QColor(240, 240, 240))
        font = painter.font()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        text_rect = QRect(int(cx - card_w / 2.0), int(cy + 18.0), int(card_w), 24)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._message)

        if self._subtext:
            painter.setPen(QColor(160, 160, 160))
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
        title = QLabel("Efface Magique LR (AI Generative Eraser)")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        subtitle = QLabel("Local, private AI inpainting companion for Adobe Lightroom Classic")
        subtitle.setStyleSheet("font-size: 12px; color: #0078d4; font-weight: 500;")
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header.addLayout(header_text)
        header.addStretch(1)
        layout.addLayout(header)

        # Tabs
        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                background-color: #242424;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #1e1e1e;
                color: #aaaaaa;
                padding: 8px 18px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 500;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
                color: #ffffff;
                border-bottom: 2px solid #0078d4;
            }
            QTabBar::tab:hover {
                color: #ffffff;
            }
        """)

        # Tab 1: Installation
        tab_install = QTextBrowser()
        tab_install.setOpenExternalLinks(True)
        tab_install.setStyleSheet("background-color: #242424; color: #dddddd; border: none; padding: 12px; font-size: 12px;")
        tab_install.setHtml("""
            <h2 style="color: #4ade80; margin-top: 0;">🚀 Installation &amp; Setup Guide</h2>
            <p>Efface Magique LR runs 100% locally and privately on your machine, integrating natively with Adobe Lightroom Classic.</p>
            
            <h3 style="color: #60a5fa;">Step 1: Install Dependencies (1-Click)</h3>
            <ul>
                <li><b>Windows:</b> Double-click <code>install.bat</code> in the app directory, or run it in Command Prompt / PowerShell.</li>
                <li><b>macOS / Linux:</b> Open Terminal in the app folder and run <code>chmod +x install.sh && ./install.sh</code>.</li>
            </ul>
            <p style="color: #888888;"><i>The installer automatically configures a local Python virtual environment (<code>.venv</code>) with all AI PyTorch and Simple-LaMa dependencies.</i></p>

            <h3 style="color: #60a5fa;">Step 2: Add the Plugin to Adobe Lightroom Classic</h3>
            <ol>
                <li>Open <b>Adobe Lightroom Classic</b>.</li>
                <li>Open the Plug-in Manager by going to: <b>File &gt; Plug-in Manager...</b> (or press <code>Ctrl + Alt + Shift + ,</code> on Windows / <code>Cmd + Opt + Shift + ,</code> on Mac).</li>
                <li>Click the <b>Add</b> button at the bottom-left corner of the dialog.</li>
                <li>Select the plugin folder: <br/><code style="background-color: #181818; padding: 2px 6px; color: #f59e0b;">plugin/ai_eraser.lrplugin</code></li>
                <li>Verify that the status dot is <b style="color: #4ade80;">🟢 Installed and running</b>.</li>
                <li>Click <b>Done</b>. The plugin is now active!</li>
            </ol>
        """)
        self.tabs.addTab(tab_install, "🚀 Installation")

        # Tab 2: How to Use
        tab_usage = QTextBrowser()
        tab_usage.setOpenExternalLinks(True)
        tab_usage.setStyleSheet("background-color: #242424; color: #dddddd; border: none; padding: 12px; font-size: 12px;")
        tab_usage.setHtml("""
            <h2 style="color: #4ade80; margin-top: 0;">✨ How to Use Efface Magique LR</h2>

            <h3 style="color: #60a5fa;">⚡ Option A: Seamless Live Window Sync (Recommended)</h3>
            <ol>
                <li>In Lightroom Classic, select: <b>File &gt; Plug-in Extras &gt; ⚡ AI Generative Eraser (Live Window)...</b></li>
                <li>The companion window appears displaying your active photo.</li>
                <li><b>Switching photos in Lightroom</b> (arrow keys or clicking the filmstrip) instantly syncs the new photo into this window with zero lag!</li>
                <li>Paint over tourists, wires, power poles, or blemishes using the red brush (use <code>[</code> and <code>]</code> to adjust size).</li>
                <li>Toggle <b>🎯 Subject</b> to automatically lock contours to object boundaries.</li>
                <li>Click <b>✨ Erase Object</b> (or press <b>Enter</b>) to generate inpainting.</li>
                <li>Review the 3 candidate variation cards and pick your favorite result.</li>
                <li>Click <b>📥 Save &amp; Sync to Lightroom</b> (<code>Ctrl + S</code>) — the image is losslessly saved as a 16-bit TIFF and auto-stacked into your Lightroom catalog!</li>
                <li>Click <b>📌 Pin</b> (<code>Ctrl + T</code>) to keep the companion floating on top of Lightroom while you work.</li>
            </ol>

            <h3 style="color: #60a5fa;">🪄 Option B: Single Photo Mode</h3>
            <p>Select any photo in Lightroom and click: <b>File &gt; Plug-in Extras &gt; 🪄 AI Generative Eraser (Single Photo)...</b></p>
        """)
        self.tabs.addTab(tab_usage, "✨ How to Use")

        # Tab 3: Shortcuts
        tab_shortcuts = QTextBrowser()
        tab_shortcuts.setStyleSheet("background-color: #242424; color: #dddddd; border: none; padding: 12px; font-size: 12px;")
        tab_shortcuts.setHtml("""
            <h2 style="color: #4ade80; margin-top: 0;">⌨ Keyboard Shortcuts Reference</h2>
            <table width="100%" cellpadding="6" cellspacing="0" style="border-collapse: collapse; border: 1px solid #3c3c3c;">
                <tr style="background-color: #1a1a1a; color: #ffffff; border-bottom: 2px solid #0078d4;">
                    <th align="left" style="padding: 6px 10px;">Shortcut</th>
                    <th align="left" style="padding: 6px 10px;">Action</th>
                </tr>
                <tr style="border-bottom: 1px solid #333333;">
                    <td style="padding: 6px 10px;"><code>Ctrl + S</code> / <code>Ctrl + Shift + S</code></td>
                    <td><b>⚡ Save &amp; Sync to Lightroom</b> (Auto-stacks into catalog &amp; stays open)</td>
                </tr>
                <tr style="border-bottom: 1px solid #333333;">
                    <td style="padding: 6px 10px;"><code>Enter</code> / <code>Return</code></td>
                    <td><b>✨ Erase Object</b> (Run AI inpainting on marked areas)</td>
                </tr>
                <tr style="border-bottom: 1px solid #333333;">
                    <td style="padding: 6px 10px;"><code>Ctrl + T</code></td>
                    <td><b>📌 Toggle Pin (Always on Top)</b> floating above Lightroom</td>
                </tr>
                <tr style="border-bottom: 1px solid #333333;">
                    <td style="padding: 6px 10px;"><code>[</code> / <code>]</code></td>
                    <td>Decrease / Increase Brush Radius</td>
                </tr>
                <tr style="border-bottom: 1px solid #333333;">
                    <td style="padding: 6px 10px;"><code>Ctrl + Z</code></td>
                    <td>↶ Undo stroke or inpainting step</td>
                </tr>
                <tr style="border-bottom: 1px solid #333333;">
                    <td style="padding: 6px 10px;"><code>Ctrl + Y</code> / <code>Ctrl + Shift + Z</code></td>
                    <td>↷ Redo stroke or inpainting step</td>
                </tr>
                <tr style="border-bottom: 1px solid #333333;">
                    <td style="padding: 6px 10px;"><code>Spacebar</code> (Hold) / <code>\\</code></td>
                    <td>👁 Instant Before / After Comparison</td>
                </tr>
                <tr style="border-bottom: 1px solid #333333;">
                    <td style="padding: 6px 10px;"><code>Y</code></td>
                    <td>◫ Interactive Split-Screen Comparison Slider</td>
                </tr>
                <tr style="border-bottom: 1px solid #333333;">
                    <td style="padding: 6px 10px;"><code>Ctrl + 0</code> / <code>F</code></td>
                    <td>🔍 Fit photo to screen view</td>
                </tr>
                <tr style="border-bottom: 1px solid #333333;">
                    <td style="padding: 6px 10px;"><code>Ctrl + 1</code></td>
                    <td>1:1 View at 100% pixel scale</td>
                </tr>
                <tr style="border-bottom: 1px solid #333333;">
                    <td style="padding: 6px 10px;"><code>F1</code></td>
                    <td>💡 Open this User Guide &amp; Setup Dialog</td>
                </tr>
            </table>
        """)
        self.tabs.addTab(tab_shortcuts, "⌨ Shortcuts")

        # Tab 4: Download & Share
        tab_share_widget = QWidget()
        share_vbox = QVBoxLayout(tab_share_widget)
        share_vbox.setContentsMargins(14, 14, 14, 14)
        share_vbox.setSpacing(14)

        share_text = QTextBrowser()
        share_text.setOpenExternalLinks(True)
        share_text.setStyleSheet("background-color: transparent; color: #dddddd; border: none; font-size: 12px;")
        share_text.setHtml("""
            <h2 style="color: #4ade80; margin-top: 0;">📥 Easy Download &amp; Share for Other Users</h2>
            <p>You can easily distribute Efface Magique LR to other photographers or friends without requiring Git:</p>
            <ol>
                <li>Click the button below to package a clean, portable <b>Efface-Magique-LR.zip</b> archive.</li>
                <li>Share this ZIP file with any colleague or photographer via Google Drive, Dropbox, USB drive, or GitHub Release.</li>
                <li>The recipient simply unzips the folder and double-clicks <code>install.bat</code> (Windows) or <code>./install.sh</code> (macOS/Linux).</li>
                <li>In Lightroom, they add <code>plugin/ai_eraser.lrplugin</code> and they are ready to edit!</li>
            </ol>
        """)
        share_vbox.addWidget(share_text)

        btn_package = QPushButton("📦 Package Distribution ZIP for Other Users")
        btn_package.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1084d8;
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

    def _on_open_readme(self):
        readme_path = os.path.join(_PROJECT_ROOT, "README.md")
        if os.path.isfile(readme_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(readme_path))
        else:
            QMessageBox.warning(self, "Not Found", f"README.md not found at:\n{readme_path}")
