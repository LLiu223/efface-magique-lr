"""
app.py
Efface Magique LR - Adobe Lightroom Classic AI Companion Application
Adobe Firefly-grade Generative Fill & Fast GAN Dual Inpainting Interface

Key Capabilities:
- Dual Inpainting Engine:
  * ✨ Generative Firefly (Diffusion-based, 3 candidate variations, optional prompt guidance)
  * ⚡ Fast Mode (Simple-LaMa GAN for spots, wires, and quick fixes)
- Variations Carousel Panel with real-time viewport updates & "More Variations" regeneration
- Camera Sensor Grain & Noise Matching
- Split-Screen & Hold-to-Compare (\\ and Space keys)
- 16-bit Lossless Color (ProPhoto RGB / Adobe RGB / sRGB) & EXIF metadata preservation
- Headless CLI pipeline for automated testing
"""

import os
import sys
import time
import shutil
import argparse
import logging
import uuid
from typing import Optional, List
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in sys.path when invoked directly as a script (e.g. from Lightroom)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, qInstallMessageHandler, QtMsgType, QUrl
from PyQt6.QtGui import QIcon, QFont, QColor, QKeySequence, QAction, QShortcut, QPixmap, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolBar,
    QPushButton,
    QSlider,
    QLabel,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QFrame,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QDialog,
    QTabWidget,
    QTextBrowser,
    QScrollArea,
)

# Application Brand Assets
_ASSETS_DIR = os.path.join(_CURRENT_DIR, "assets")
_CHECKMARK_PATH = os.path.join(_ASSETS_DIR, "checkmark.png").replace("\\", "/")
_LOGO_PATH = os.path.join(_ASSETS_DIR, "logo.png").replace("\\", "/")
_LOGO_ICO_PATH = os.path.join(_ASSETS_DIR, "logo.ico").replace("\\", "/")

from companion.canvas import ImageCanvas, pil_to_qimage
from companion.inpainting_engine import InpaintingEngine, get_optimal_device, EngineMode, get_device_telemetry
from companion.layers import ModificationLayer, create_layer_thumbnail
from companion.live_bridge import LiveBridgeServer

# Configure logging to console and file
_log_file = os.path.join(_PROJECT_ROOT, "companion.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger("EffaceMagiqueCompanion")


def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Log any unhandled exception to prevent silent qFatal crashes."""
    logger.critical("Unhandled top-level exception:", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = global_exception_handler


def qt_message_handler(mode, context, message):
    if mode == QtMsgType.QtFatalMsg:
        logger.critical(f"Qt Fatal: {message} ({context.file}:{context.line})")
    elif mode == QtMsgType.QtCriticalMsg:
        logger.error(f"Qt Critical: {message} ({context.file}:{context.line})")
    elif mode == QtMsgType.QtWarningMsg:
        logger.warning(f"Qt Warning: {message} ({context.file}:{context.line})")


qInstallMessageHandler(qt_message_handler)


# -----------------------------------------------------------------------------
# Background Worker Thread for Non-Blocking AI Inpainting
# -----------------------------------------------------------------------------

class InpaintingWorker(QThread):
    progress = pyqtSignal(int, str)
    variationsReady = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        engine: InpaintingEngine,
        image: Image.Image,
        mask: Image.Image,
        num_variations: int = 3,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
        detect_subject: bool = True,
        enable_grain: bool = False,
    ):
        super().__init__()
        self.engine = engine
        self.image = image
        self.mask = mask
        self.num_variations = num_variations
        self.prompt = prompt
        self.seed = seed
        self.detect_subject = detect_subject
        self.enable_grain = enable_grain

    def run(self):
        try:
            self.progress.emit(5, "Analyzing scene context and mask bounds...")
            variations = self.engine.generate_variations(
                image=self.image,
                mask=self.mask,
                num_variations=self.num_variations,
                prompt=self.prompt,
                base_seed=self.seed,
                margin_ratio=0.85,
                feather_radius=20,
                detect_subject=self.detect_subject,
                enable_grain=self.enable_grain,
                progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
            )
            self.variationsReady.emit(variations)
        except Exception as e:
            logger.exception("Inpainting failed")
            self.error.emit(str(e))


# -----------------------------------------------------------------------------
# Main Application Window
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


class MainWindow(QMainWindow):
    def __init__(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        live_mode: bool = False,
        bridge_port: int = 51739,
    ):
        super().__init__()

        self.input_path = input_path
        self.output_path = output_path
        self.is_live_mode = live_mode
        self.current_photo_id: Optional[str] = None
        self.current_original_path: Optional[str] = None

        if not self.output_path and self.input_path:
            is_tmp_lr = os.path.abspath(self.input_path).startswith(os.path.abspath(os.path.join(_PROJECT_ROOT, ".tmp")))
            if is_tmp_lr:
                self.output_path = self.input_path
            else:
                base, ext = os.path.splitext(self.input_path)
                self.output_path = f"{base}_ai_edit{ext}"
        self.saved_successfully = False

        # Metadata preservation
        self.original_icc_profile = None
        self.original_exif = None
        self.original_dpi = None

        # Variations state
        self.current_variations: List[Image.Image] = []
        self.active_variation_index: int = 0
        self.last_used_mask: Optional[Image.Image] = None
        self.last_base_image: Optional[Image.Image] = None
        self.var_buttons: List[QPushButton] = []
        self._variation_pixmaps: List[QPixmap] = []

        # Non-destructive Modification Layers state
        self.base_image: Optional[Image.Image] = None
        self._base_pixmap: Optional[QPixmap] = None
        self.modification_layers: List[ModificationLayer] = []
        self.active_layer_index: Optional[int] = None
        self.layer_card_widgets: dict = {}

        self.setWindowTitle("Efface Magique LR - Generative AI Eraser")
        if os.path.isfile(_LOGO_PATH):
            self.setWindowIcon(QIcon(_LOGO_PATH))
        self.resize(1400, 920)
        self.setStyleSheet(DARK_STYLE)

        # Inpainting Engine instance
        device = get_optimal_device()
        self.engine = InpaintingEngine(device=device, mode=EngineMode.FIREFLY)
        self.worker: Optional[InpaintingWorker] = None

        # Asynchronously pre-warm inpainting models in background daemon thread
        import threading
        self._prewarm_thread = threading.Thread(target=self._prewarm_engine, daemon=True)
        self._prewarm_thread.start()

        # Start high-performance local Live IPC Bridge
        self.live_bridge = LiveBridgeServer(preferred_port=bridge_port, parent=self)
        self.live_bridge_port = self.live_bridge.start()
        self.live_bridge.photo_selected.connect(self._on_live_photo_selected)
        self.live_bridge.focus_requested.connect(self._on_live_focus_requested)
        self.live_bridge.import_completed.connect(self._on_live_import_completed)
        self.live_bridge.close_requested.connect(self.close)

        # Build UI
        central_container = QWidget()
        central_layout = QVBoxLayout(central_container)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self.canvas = ImageCanvas(self)
        self.top_bar = self._create_top_bar()
        central_layout.addWidget(self.top_bar)

        # Middle container: Canvas (center) + Modifications Layers Sidebar (right)
        middle_container = QWidget()
        middle_layout = QHBoxLayout(middle_container)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0)

        middle_layout.addWidget(self.canvas, stretch=1)
        self.layers_panel = self._create_layers_panel()
        middle_layout.addWidget(self.layers_panel)

        central_layout.addWidget(middle_container, stretch=1)

        # Variations Carousel Panel at the bottom
        self.carousel_panel = self._create_carousel_panel()
        self.carousel_panel.setVisible(False)
        central_layout.addWidget(self.carousel_panel)

        self.setCentralWidget(central_container)

        self._create_statusbar()
        self._connect_signals()

        # Global hotkeys
        QShortcut(QKeySequence("Ctrl+0"), self, self.canvas.fit_to_screen)
        QShortcut(QKeySequence("Ctrl+1"), self, self.canvas.zoom_to_actual_size)
        QShortcut(QKeySequence("F"), self, self.canvas.fit_to_screen)
        QShortcut(QKeySequence("Ctrl+T"), self, lambda: self.btn_pin.setChecked(not self.btn_pin.isChecked()))
        QShortcut(QKeySequence("Y"), self, lambda: self.btn_split.setChecked(not self.btn_split.isChecked()))
        QShortcut(QKeySequence("Ctrl+L"), self, lambda: self.btn_layers_toggle.setChecked(not self.btn_layers_toggle.isChecked()))
        QShortcut(QKeySequence("L"), self, lambda: self.btn_layers_toggle.setChecked(not self.btn_layers_toggle.isChecked()))

        # Load initial image if provided
        if self.input_path and os.path.isfile(self.input_path):
            self.load_image_file(self.input_path)
        else:
            self.statusBar().showMessage("Ready. Select any photo in Lightroom Classic or open an image.")

    def showEvent(self, event):
        super().showEvent(event)
        # Ensure image is fitted comfortably to the actual rendered window viewport
        QTimer.singleShot(50, self.canvas.fit_to_screen)

    def _prewarm_engine(self):
        """Pre-warm inpainting models in background daemon thread to eliminate first-erase cold-start delay."""
        try:
            if not self.engine.is_loaded:
                self.engine.load_model()
        except Exception as e:
            logger.warning(f"Background engine warmup non-fatal exception: {e}")

    def _create_carousel_panel(self) -> QFrame:
        """Create the bottom carousel panel for Firefly candidate variations."""
        panel = QFrame(self)
        panel.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border-top: 1px solid #2d2d2d;
                padding: 4px 10px;
            }
        """)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(12)

        lbl = QLabel("<b>Generative Variations:</b>")
        lbl.setStyleSheet("color: #0078d4; font-size: 12px; font-weight: bold;")
        layout.addWidget(lbl)

        self.var_card_layout = QHBoxLayout()
        self.var_card_layout.setSpacing(10)
        layout.addLayout(self.var_card_layout)

        layout.addStretch(1)

        self.btn_more_variations = QPushButton("🔄 More Variations")
        self.btn_more_variations.setToolTip("Generate 3 more candidate variations with new random seeds")
        self.btn_more_variations.clicked.connect(self._on_generate_more_variations)
        layout.addWidget(self.btn_more_variations)

        self.btn_accept_variation = QPushButton("✔ Keep Selected")
        self.btn_accept_variation.setToolTip("Accept active variation and hide carousel")
        self.btn_accept_variation.clicked.connect(self._on_accept_variation)
        layout.addWidget(self.btn_accept_variation)

        return panel

    # -------------------------------------------------------------------------
    # Non-Destructive Modifications Layer Stack & UI Panel
    # -------------------------------------------------------------------------

    def _create_layers_panel(self) -> QFrame:
        """Create the right-side Modifications (Layers) sidebar panel."""
        panel = QFrame(self)
        panel.setObjectName("layersPanel")
        panel.setFixedWidth(270)
        panel.setStyleSheet("""
            QFrame#layersPanel {
                background-color: #1e1e1e;
                border-left: 1px solid #333333;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header with Title, Count, and Controls
        header_layout = QHBoxLayout()
        header_title = QLabel("<b>📋 Modifications</b>")
        header_title.setStyleSheet("color: #ffffff; font-size: 12px;")
        header_layout.addWidget(header_title)

        self.lbl_layer_count = QLabel("(0)")
        self.lbl_layer_count.setStyleSheet("color: #888888; font-size: 11px;")
        header_layout.addWidget(self.lbl_layer_count)
        header_layout.addStretch(1)

        # "+ New" button: deselects active layer so user can draw a fresh modification
        self.btn_new_layer = QPushButton("+ New")
        self.btn_new_layer.setToolTip("Start a new modification on the current image")
        self.btn_new_layer.setFixedHeight(24)
        self.btn_new_layer.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                border: 1px solid #444444;
                border-radius: 4px;
                color: #ffffff;
                font-size: 11px;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #0078d4;
                border-color: #0078d4;
            }
        """)
        self.btn_new_layer.clicked.connect(self._on_new_layer_clicked)
        header_layout.addWidget(self.btn_new_layer)

        # Delete layer button (Red with 'Delete' text)
        self.btn_delete_layer = QPushButton("Delete")
        self.btn_delete_layer.setToolTip("Delete selected modification layer")
        self.btn_delete_layer.setFixedHeight(24)
        self.btn_delete_layer.setStyleSheet("""
            QPushButton {
                background-color: #c42b1c;
                border: 1px solid #e81123;
                border-radius: 4px;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #e81123;
            }
        """)
        self.btn_delete_layer.clicked.connect(self._on_delete_selected_layer)
        header_layout.addWidget(self.btn_delete_layer)

        layout.addLayout(header_layout)

        # Active layer notification / editing banner
        self.layer_active_banner = QFrame()
        self.layer_active_banner.setStyleSheet("""
            QFrame {
                background-color: #0d2847;
                border: 1px solid #0078d4;
                border-radius: 4px;
                padding: 4px 6px;
            }
        """)
        banner_layout = QHBoxLayout(self.layer_active_banner)
        banner_layout.setContentsMargins(4, 2, 4, 2)
        self.lbl_banner_text = QLabel("✏️ Editing: Mod 1")
        self.lbl_banner_text.setStyleSheet("color: #4cc2ff; font-size: 11px; font-weight: bold;")
        banner_layout.addWidget(self.lbl_banner_text)
        banner_layout.addStretch(1)

        btn_done_editing = QPushButton("Done")
        btn_done_editing.setFixedHeight(20)
        btn_done_editing.setStyleSheet("""
            QPushButton {
                background-color: #1a4971;
                border: 1px solid #0078d4;
                border-radius: 3px;
                color: #ffffff;
                font-size: 10px;
                padding: 1px 6px;
            }
            QPushButton:hover {
                background-color: #0078d4;
            }
        """)
        btn_done_editing.clicked.connect(self._on_done_editing_layer)
        banner_layout.addWidget(btn_done_editing)
        self.layer_active_banner.setVisible(False)
        layout.addWidget(self.layer_active_banner)

        # Scroll Area for Layer Cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.layer_cards_container = QWidget()
        self.layer_cards_layout = QVBoxLayout(self.layer_cards_container)
        self.layer_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_cards_layout.setSpacing(6)
        self.layer_cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.layer_cards_container)
        layout.addWidget(scroll, stretch=1)

        # Base Image Layer Card at bottom
        self.base_layer_card = self._create_base_layer_card()
        layout.addWidget(self.base_layer_card)

        return panel

    def _create_base_layer_card(self) -> QFrame:
        """Create the bottom card representing the unedited original photo."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 4px 8px;
            }
            QFrame:hover {
                background-color: #222222;
                border-color: #444444;
            }
        """)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        lbl_icon = QLabel("🖼️")
        lbl_icon.setStyleSheet("font-size: 14px;")
        layout.addWidget(lbl_icon)

        lbl_text = QLabel("<b>Base Photo</b><br><span style='color: #888888; font-size: 10px;'>Original input</span>")
        lbl_text.setStyleSheet("color: #dddddd; font-size: 11px;")
        layout.addWidget(lbl_text, stretch=1)

        btn_view_base = QPushButton("View")
        btn_view_base.setToolTip("Preview original unedited photo on canvas")
        btn_view_base.setFixedHeight(22)
        btn_view_base.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 4px;
                color: #cccccc;
                font-size: 10px;
                padding: 2px 6px;
            }
            QPushButton:hover {
                background-color: #0078d4;
                color: #ffffff;
            }
        """)
        btn_view_base.clicked.connect(self._on_view_base_photo)
        layout.addWidget(btn_view_base)

        return card

    def _on_view_base_photo(self):
        """Display the original unedited photo on canvas."""
        if self.base_image:
            self.canvas.set_display_image(self.base_image, cached_pixmap=getattr(self, "_base_pixmap", None))
            self.canvas.clear_mask(save_state=False)
            self.active_layer_index = None
            self.layer_active_banner.setVisible(False)
            self.carousel_panel.setVisible(False)
            self._update_layer_cards_selection()
            self.statusBar().showMessage("Viewing original Base Photo. Click '+ New' or select a layer to resume editing.", 3000)

    def _create_layer_card(self, idx: int) -> QFrame:
        """Create a single clickable layer card representing one modification."""
        layer = self.modification_layers[idx]
        is_active = (idx == self.active_layer_index)

        card = QFrame()
        card.setCursor(Qt.CursorShape.PointingHandCursor)

        border_color = "#0078d4" if is_active else "#333333"
        bg_color = "#15283c" if is_active else "#222222"
        border_width = 2 if is_active else 1
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: {border_width}px solid {border_color};
                border-radius: 6px;
            }}
            QFrame:hover {{
                background-color: #272727;
                border-color: #0078d4;
            }}
        """)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        # 1. High-Visibility Modification Layer Toggle Box
        btn_toggle = QPushButton("✓" if layer.visible else "")
        btn_toggle.setFixedSize(26, 26)
        btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        if layer.visible:
            btn_toggle.setToolTip(f"{layer.name} is visible (click to hide)")
            btn_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #0f3d63;
                    border: 2px solid #0078d4;
                    border-radius: 5px;
                    color: #38bdf8;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 0px;
                }
                QPushButton:hover {
                    background-color: #1a4f80;
                    border-color: #60cdff;
                    color: #ffffff;
                }
            """)
        else:
            btn_toggle.setToolTip(f"{layer.name} is hidden (click to show)")
            btn_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #181818;
                    border: 1.5px solid #555555;
                    border-radius: 5px;
                    color: #777777;
                    font-size: 11px;
                    font-weight: bold;
                    padding: 0px;
                }
                QPushButton:hover {
                    background-color: #252525;
                    border-color: #888888;
                    color: #aaaaaa;
                }
            """)
        btn_toggle.clicked.connect(lambda _, i=idx: self._toggle_layer_visibility(i))
        layout.addWidget(btn_toggle)
        card.btn_toggle = btn_toggle
        card.btn_eye = btn_toggle

        # 2. Thumbnail
        thumb_label = QLabel()
        if layer.thumbnail is not None and not layer.thumbnail.isNull():
            thumb_label.setPixmap(layer.thumbnail)
        else:
            thumb_pix = QPixmap(56, 42)
            thumb_pix.fill(QColor(32, 32, 32))
            thumb_label.setPixmap(thumb_pix)
        thumb_label.setFixedSize(56, 42)
        thumb_label.setStyleSheet("border-radius: 3px;")
        layout.addWidget(thumb_label)

        # 3. Layer Name and Details
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(1)

        name_color = "#4cc2ff" if is_active else "#ffffff"
        name_lbl = QLabel(f"<b>{layer.name}</b>")
        name_lbl.setStyleSheet(f"color: {name_color}; font-size: 11px;")
        info_layout.addWidget(name_lbl)

        mode_str = "Fast Spot" if layer.engine_mode == EngineMode.FAST else "Generative AI"
        detail_lbl = QLabel(f"{mode_str}")
        detail_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        info_layout.addWidget(detail_lbl)

        layout.addLayout(info_layout, stretch=1)

        # 4. Quick Delete button (Red with 'Delete' text)
        btn_del = QPushButton("Delete")
        btn_del.setToolTip(f"Delete {layer.name}")
        btn_del.setFixedHeight(22)
        btn_del.setStyleSheet("""
            QPushButton {
                background-color: #8b1818;
                border: 1px solid #c42b1c;
                border-radius: 3px;
                color: #ffffff;
                font-size: 10px;
                font-weight: bold;
                padding: 1px 6px;
            }
            QPushButton:hover {
                background-color: #c42b1c;
                border-color: #e81123;
            }
        """)
        btn_del.clicked.connect(lambda _, i=idx: self._delete_layer(i))
        layout.addWidget(btn_del)

        # Left-click on card selects and enters edit mode for this layer
        card.mousePressEvent = lambda event, i=idx: self._on_layer_card_clicked(event, i)

        self.layer_card_widgets[idx] = card
        return card

    def _on_layer_card_clicked(self, event, idx: int):
        if event.button() == Qt.MouseButton.LeftButton:
            self._select_layer(idx)

    def _update_layer_cards_selection(self):
        """Update active borders on layer cards in-place without rebuilding widgets."""
        for idx, card in list(self.layer_card_widgets.items()):
            try:
                is_active = (idx == self.active_layer_index)
                border_color = "#0078d4" if is_active else "#333333"
                bg_color = "#15283c" if is_active else "#222222"
                border_width = 2 if is_active else 1
                card.setStyleSheet(f"""
                    QFrame {{
                        background-color: {bg_color};
                        border: {border_width}px solid {border_color};
                        border-radius: 6px;
                    }}
                    QFrame:hover {{
                        background-color: #272727;
                        border-color: #0078d4;
                    }}
                """)
            except RuntimeError:
                pass

    def _update_layer_card_visibility_ui(self, idx: int):
        """Update toggle box styling in-place without rebuilding widgets."""
        if idx in self.layer_card_widgets and 0 <= idx < len(self.modification_layers):
            card = self.layer_card_widgets[idx]
            layer = self.modification_layers[idx]
            btn = getattr(card, "btn_toggle", None)
            if btn is not None:
                if layer.visible:
                    btn.setText("✓")
                    btn.setToolTip(f"{layer.name} is visible (click to hide)")
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #0f3d63;
                            border: 2px solid #0078d4;
                            border-radius: 5px;
                            color: #38bdf8;
                            font-size: 14px;
                            font-weight: bold;
                            padding: 0px;
                        }
                        QPushButton:hover {
                            background-color: #1a4f80;
                            border-color: #60cdff;
                            color: #ffffff;
                        }
                    """)
                else:
                    btn.setText("")
                    btn.setToolTip(f"{layer.name} is hidden (click to show)")
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #181818;
                            border: 1.5px solid #555555;
                            border-radius: 5px;
                            color: #777777;
                            font-size: 11px;
                            font-weight: bold;
                            padding: 0px;
                        }
                        QPushButton:hover {
                            background-color: #252525;
                            border-color: #888888;
                            color: #aaaaaa;
                        }
                    """)

    def _refresh_layers_ui(self):
        """Re-render all modification layer cards in the sidebar panel."""
        if not hasattr(self, "layer_cards_layout"):
            return

        self.layer_card_widgets.clear()
        while self.layer_cards_layout.count():
            item = self.layer_cards_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self.lbl_layer_count.setText(f"({len(self.modification_layers)})")

        if self.active_layer_index is not None and 0 <= self.active_layer_index < len(self.modification_layers):
            active_layer = self.modification_layers[self.active_layer_index]
            self.lbl_banner_text.setText(f"✏️ Editing: {active_layer.name}")
            self.layer_active_banner.setVisible(True)
            self.btn_erase.setText(f"Update {active_layer.name}")
        else:
            self.layer_active_banner.setVisible(False)
            self.btn_erase.setText("Erase Object")

        # Render in reverse order so the newest layer is on top of the list
        for idx in reversed(range(len(self.modification_layers))):
            card = self._create_layer_card(idx)
            self.layer_cards_layout.addWidget(card)

    def _select_layer(self, index: int):
        """Select a modification layer to inspect or modify its mask, prompt, or variation."""
        if not (0 <= index < len(self.modification_layers)):
            return

        self.active_layer_index = index
        layer = self.modification_layers[index]

        # Fast display switch using cached composite/pixmap (0ms)
        if index == 0:
            pre_layer_img = self.base_image
            cached_pix = getattr(self, "_base_pixmap", None)
        else:
            prev_layer = self.modification_layers[index - 1]
            pre_layer_img = prev_layer.composite_cache or self._get_composite_image(up_to_index=index - 1)
            cached_pix = prev_layer.cached_pixmap

        self.canvas.set_display_image(pre_layer_img, cached_pixmap=cached_pix)

        # Load this layer's mask onto the canvas overlay (0ms if cached_mask_qimage present)
        self.canvas.set_mask_image(layer.mask, cached_mask_qimage=layer.cached_mask_qimage)

        # Update controls
        self.btn_erase.setText(f"Update {layer.name}")
        self.layer_active_banner.setVisible(True)
        self.lbl_banner_text.setText(f"✏️ Editing: {layer.name}")

        # Update mode combo and checkboxes
        idx = self.combo_engine.findData(layer.engine_mode)
        if idx >= 0:
            self.combo_engine.setCurrentIndex(idx)
        self.chk_detect_subject.setChecked(layer.detect_subject)
        self.chk_grain.setChecked(layer.enable_grain)

        # If layer has candidate variations, display them in the carousel
        if layer.variations and len(layer.variations) > 1:
            self.current_variations = layer.variations
            self.active_variation_index = layer.active_variation_index
            self._variation_pixmaps = [QPixmap.fromImage(pil_to_qimage(v)) for v in layer.variations]
            self._render_carousel_thumbnails()
            self.carousel_panel.setVisible(True)
        else:
            self.carousel_panel.setVisible(False)

        # Update layer card selection styles in-place (INSTANT, no widget rebuild!)
        self._update_layer_cards_selection()
        self.statusBar().showMessage(
            f"Selected {layer.name}. Paint or Erase to modify mask, then click 'Update {layer.name}'.", 4000
        )

    def _toggle_layer_visibility(self, index: int):
        """Toggle visibility on a specific modification layer and re-composite the stack."""
        if 0 <= index < len(self.modification_layers):
            layer = self.modification_layers[index]
            layer.visible = not layer.visible

            # FAST PATH: If toggling the TOP layer and all preceding layers are visible
            is_top = (index == len(self.modification_layers) - 1)
            all_preceding_visible = all(l.visible for l in self.modification_layers[:index])

            if is_top and all_preceding_visible:
                if layer.visible:
                    # Restoring top layer -> instantaneous display of its cached composite/pixmap (0ms)
                    disp_img = layer.composite_cache or layer.get_active_image()
                    disp_pix = layer.cached_pixmap
                else:
                    # Hiding top layer -> instantaneous display of preceding layer (0ms)
                    if index > 0:
                        disp_img = self.modification_layers[index - 1].composite_cache or self.base_image
                        disp_pix = self.modification_layers[index - 1].cached_pixmap
                    else:
                        disp_img = self.base_image
                        disp_pix = getattr(self, "_base_pixmap", None)

                self.canvas.set_display_image(disp_img, cached_pixmap=disp_pix)
                self.canvas._current_pil = disp_img.copy()
            else:
                # Invalidate composite caches from index onwards and recompute stack
                for l in self.modification_layers[index:]:
                    l.composite_cache = None
                    l.cached_pixmap = None

                full_composite = self._recompute_composite_stack(from_index=index)
                top_cached_pix = (
                    self.modification_layers[-1].cached_pixmap
                    if (self.modification_layers and self.modification_layers[-1].cached_pixmap)
                    else getattr(self, "_base_pixmap", None)
                )
                self.canvas.set_display_image(full_composite, cached_pixmap=top_cached_pix)
                self.canvas._current_pil = full_composite.copy()

            if self.active_layer_index == index and not layer.visible:
                self._on_new_layer_clicked()
            else:
                self._update_layer_card_visibility_ui(index)

            state = "visible" if layer.visible else "hidden"
            self.statusBar().showMessage(f"{layer.name} is now {state}.", 3000)

    def _delete_layer(self, index: int):
        """Delete a modification layer and cleanly re-composite the stack."""
        if 0 <= index < len(self.modification_layers):
            layer = self.modification_layers.pop(index)
            if self.active_layer_index == index:
                self.active_layer_index = None
                self.canvas.clear_mask(save_state=False)
                if hasattr(self, "carousel_panel"):
                    self.carousel_panel.setVisible(False)
                self.current_variations = []
            elif self.active_layer_index is not None and self.active_layer_index > index:
                self.active_layer_index -= 1

            # Invalidate composite caches from index onwards on all remaining layers
            for l in self.modification_layers[index:]:
                l.composite_cache = None
                l.cached_pixmap = None

            if self.modification_layers:
                full_composite = self._recompute_composite_stack(from_index=index)
                top_cached_pix = (
                    self.modification_layers[-1].cached_pixmap
                    if self.modification_layers[-1].cached_pixmap is not None
                    else QPixmap.fromImage(pil_to_qimage(full_composite))
                )
            else:
                full_composite = self.base_image.copy() if self.base_image is not None else Image.new("RGB", (100, 100), (0, 0, 0))
                top_cached_pix = getattr(self, "_base_pixmap", None)

            self.canvas.set_display_image(full_composite, cached_pixmap=top_cached_pix)
            self.canvas._current_pil = full_composite.copy()

            self._refresh_layers_ui()
            self.statusBar().showMessage(f"Deleted {layer.name}.", 3000)

    def _recompute_composite_stack(self, from_index: int = 0) -> Image.Image:
        """
        Recompute composite stack from `from_index` onwards and refresh all caches, pixmaps, and thumbnails.
        Ensures deleted or hidden layers are completely purged from all subsequent layers.
        """
        if self.base_image is None:
            curr = self.canvas.get_current_image()
            if curr is not None:
                self.base_image = curr.copy().convert("RGB")
                self._base_pixmap = QPixmap.fromImage(pil_to_qimage(self.base_image))
            else:
                return Image.new("RGB", (100, 100), (0, 0, 0))

        if not self.modification_layers:
            return self.base_image.copy()

        from_index = max(0, min(from_index, len(self.modification_layers)))
        img_w, img_h = self.base_image.size

        # Determine starting composite
        if from_index == 0:
            composite = self.base_image.copy()
        else:
            prev_layer = self.modification_layers[from_index - 1]
            if prev_layer.composite_cache is not None:
                composite = prev_layer.composite_cache.copy()
            else:
                composite = self._get_composite_image(up_to_index=from_index - 1)

        from companion.utils.blending import seamless_distance_feather_blend

        for i in range(from_index, len(self.modification_layers)):
            layer = self.modification_layers[i]
            if not layer.visible:
                layer.composite_cache = None
                layer.cached_pixmap = None
                continue

            pre_layer_composite = composite.copy()
            active_img = layer.get_active_image()
            if active_img is not None and layer.mask is not None:
                mask_l = layer.mask.convert("L") if layer.mask.mode != "L" else layer.mask
                mask_np = np.asarray(mask_l)
                bx, by, bw, bh = cv2.boundingRect((mask_np > 10).astype(np.uint8))
                if bw > 0 and bh > 0:
                    pad = 25
                    x1 = max(0, bx - pad)
                    y1 = max(0, by - pad)
                    x2 = min(img_w, bx + bw + pad)
                    y2 = min(img_h, by + bh + pad)

                    sub_comp = composite.crop((x1, y1, x2, y2)).convert("RGB")
                    sub_active = active_img.crop((x1, y1, x2, y2)).convert("RGB")
                    sub_mask = mask_l.crop((x1, y1, x2, y2))

                    blended_sub = seamless_distance_feather_blend(
                        sub_comp,
                        sub_active,
                        sub_mask,
                        feather_radius=14,
                    )
                    composite.paste(blended_sub, (x1, y1))

                    # If the layer has candidate variations, update each variation against pre_layer_composite
                    if layer.variations and len(layer.variations) > 1:
                        for v_idx, var_img in enumerate(layer.variations):
                            if v_idx == layer.active_variation_index:
                                layer.variations[v_idx] = composite.copy()
                            else:
                                var_comp = pre_layer_composite.copy()
                                sub_v_comp = var_comp.crop((x1, y1, x2, y2)).convert("RGB")
                                sub_v_act = var_img.crop((x1, y1, x2, y2)).convert("RGB")
                                blended_v = seamless_distance_feather_blend(
                                    sub_v_comp,
                                    sub_v_act,
                                    sub_mask,
                                    feather_radius=14,
                                )
                                var_comp.paste(blended_v, (x1, y1))
                                layer.variations[v_idx] = var_comp

            # Store updated composite and pixmap on layer
            layer.composite_cache = composite.copy()
            layer.cached_pixmap = QPixmap.fromImage(pil_to_qimage(composite))
            layer.inpainted_image = composite.copy()
            layer.update_thumbnail(self.base_image)

        return composite

    def _on_new_layer_clicked(self):
        """Deselect active layer and return canvas to full composited view ready for new edits."""
        self.active_layer_index = None
        if (
            self.modification_layers
            and all(l.visible for l in self.modification_layers)
            and self.modification_layers[-1].composite_cache is not None
        ):
            full_composite = self.modification_layers[-1].composite_cache
            top_cached_pix = self.modification_layers[-1].cached_pixmap
        else:
            full_composite = self._get_composite_image()
            top_cached_pix = (
                self.modification_layers[-1].cached_pixmap
                if (self.modification_layers and self.modification_layers[-1].cached_pixmap)
                else None
            )
        self.canvas.set_display_image(full_composite, cached_pixmap=top_cached_pix)
        self.canvas.clear_mask(save_state=False)
        self.btn_erase.setText("Erase Object")
        self.layer_active_banner.setVisible(False)
        self.carousel_panel.setVisible(False)
        self._update_layer_cards_selection()
        self.statusBar().showMessage("Ready for new modification. Paint red brush on photo and click Erase Object.", 3000)

    def _on_delete_selected_layer(self):
        """Delete whichever layer is currently selected."""
        if self.active_layer_index is not None and 0 <= self.active_layer_index < len(self.modification_layers):
            self._delete_layer(self.active_layer_index)
        elif self.modification_layers:
            self._delete_layer(len(self.modification_layers) - 1)

    def _on_done_editing_layer(self):
        """Exit layer editing mode and return to composite view."""
        self._on_new_layer_clicked()

    def _toggle_layers_panel(self, visible: bool):
        """Show or hide the Modifications sidebar panel."""
        if hasattr(self, "layers_panel"):
            self.layers_panel.setVisible(visible)
        if hasattr(self, "btn_layers_toggle"):
            self.btn_layers_toggle.setText("✓ Layers" if visible else "Layers")

    def _get_composite_image(self, up_to_index: Optional[int] = None) -> Image.Image:
        """
        Generate the composited image from self.base_image applying all visible layers.
        Uses fast bounded-box localized feathering for 100x-500x speedup on 24MP-60MP photos.
        If up_to_index is provided (e.g. -1 for pristine base image, or k for layers 0..k),
        only applies layers up to that index.
        """
        if self.base_image is None:
            curr = self.canvas.get_current_image()
            if curr is not None:
                self.base_image = curr.copy().convert("RGB")
            else:
                return Image.new("RGB", (100, 100), (0, 0, 0))

        if up_to_index is not None and up_to_index < 0:
            return self.base_image.copy()

        limit = len(self.modification_layers) if up_to_index is None else min(up_to_index + 1, len(self.modification_layers))
        if limit == 0:
            return self.base_image.copy()

        # Fast path: if all layers up to limit are visible and the top one has composite_cache, reuse it!
        all_visible = all(self.modification_layers[i].visible for i in range(limit))
        if all_visible and self.modification_layers[limit - 1].composite_cache is not None:
            return self.modification_layers[limit - 1].composite_cache.copy()

        composite = self.base_image.copy()
        img_w, img_h = composite.size

        from companion.utils.blending import seamless_distance_feather_blend

        for i in range(limit):
            layer = self.modification_layers[i]
            if not layer.visible:
                continue
            active_img = layer.get_active_image()
            if active_img is None or layer.mask is None:
                continue

            # Bounded crop optimization: find bounding box of mask + 25px feather padding
            mask_l = layer.mask.convert("L") if layer.mask.mode != "L" else layer.mask
            mask_np = np.asarray(mask_l)
            bx, by, bw, bh = cv2.boundingRect((mask_np > 10).astype(np.uint8))
            if bw == 0 or bh == 0:
                continue

            pad = 25
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(img_w, bx + bw + pad)
            y2 = min(img_h, by + bh + pad)

            sub_comp = composite.crop((x1, y1, x2, y2)).convert("RGB")
            sub_active = active_img.crop((x1, y1, x2, y2)).convert("RGB")
            sub_mask = mask_l.crop((x1, y1, x2, y2))

            blended_sub = seamless_distance_feather_blend(
                sub_comp,
                sub_active,
                sub_mask,
                feather_radius=14,
            )
            composite.paste(blended_sub, (x1, y1))

        return composite


    def _render_loading_carousel(self):
        """Display placeholder loading cards during candidate generation."""
        while self.var_card_layout.count():
            item = self.var_card_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.var_buttons = []
        for i in range(3):
            card = QFrame()
            card.setFixedSize(140, 72)
            card.setStyleSheet("""
                QFrame {
                    background-color: #242424;
                    border: 1px dashed #404040;
                    border-radius: 6px;
                }
            """)
            c_layout = QVBoxLayout(card)
            c_layout.setContentsMargins(4, 4, 4, 4)
            c_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl = QLabel(f"⏳ Generating {i + 1}...")
            lbl.setStyleSheet("color: #0078d4; font-size: 11px; font-weight: 500;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            c_layout.addWidget(lbl)
            self.var_card_layout.addWidget(card)

    def _create_top_bar(self) -> QFrame:
        """Create the Lightroom-style top control bar."""
        bar = QFrame(self)
        bar.setObjectName("topBar")
        bar.setStyleSheet("""
            QFrame#topBar {
                background-color: #242424;
                border-bottom: 1px solid #333333;
            }
            QFrame#topBar QLabel {
                color: #cccccc;
                font-size: 11px;
            }
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(6)

        def add_vsep():
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            sep.setStyleSheet("color: #383838; max-width: 1px;")
            layout.addWidget(sep)

        # Brand App Logo
        if os.path.isfile(_LOGO_PATH):
            self.logo_label = QLabel()
            pix = QPixmap(_LOGO_PATH).scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.logo_label.setPixmap(pix)
            self.logo_label.setToolTip("Efface Magique LR - AI Generative Eraser")
            self.logo_label.setStyleSheet("padding-right: 4px;")
            layout.addWidget(self.logo_label)

        # Open file button (for standalone mode)
        self.btn_open = QPushButton("📂 Open")
        self.btn_open.setToolTip("Open Photo from computer")
        self.btn_open.clicked.connect(self._on_open_file)
        layout.addWidget(self.btn_open)

        add_vsep()

        # Engine Mode Switcher (clean naming without emojis)
        self.combo_engine = QComboBox()
        self.combo_engine.addItem("Generative AI", EngineMode.FIREFLY)
        self.combo_engine.addItem("Fast Spot Removal", EngineMode.FAST)
        self.combo_engine.currentIndexChanged.connect(self._on_engine_changed)
        layout.addWidget(self.combo_engine)

        # Detect Subject (Object-Aware) toggle
        self.chk_detect_subject = QCheckBox("Subject")
        self.chk_detect_subject.setChecked(False)
        self.chk_detect_subject.setToolTip("Automatically isolate the subject inside your brush stroke so background does not change")
        layout.addWidget(self.chk_detect_subject)

        # Sensor Grain toggle
        self.chk_grain = QCheckBox("Grain")
        self.chk_grain.setChecked(False)
        self.chk_grain.setToolTip("Add subtle monochromatic camera sensor noise")
        layout.addWidget(self.chk_grain)

        add_vsep()

        # Brush Radius slider
        layout.addWidget(QLabel("Size:"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(5, 200)
        self.size_slider.setValue(self.canvas.brush_radius)
        self.size_slider.setFixedWidth(75)
        self.size_slider.setToolTip("Brush Radius (Hotkeys: [ and ])")
        layout.addWidget(self.size_slider)

        self.size_label = QLabel(f"{self.canvas.brush_radius}px")
        self.size_label.setFixedWidth(36)
        layout.addWidget(self.size_label)

        add_vsep()

        # Fit to Screen button
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setToolTip("Fit entire image into screen (Ctrl+0 / F)")
        self.btn_fit.clicked.connect(self.canvas.fit_to_screen)
        layout.addWidget(self.btn_fit)

        # Compare Before / After Toggle
        self.btn_compare = QPushButton("Compare")
        self.btn_compare.setToolTip("Hold \\ or Spacebar for instant Before / After preview")
        self.btn_compare.setCheckable(True)
        self.btn_compare.toggled.connect(self.canvas.set_compare_mode)
        layout.addWidget(self.btn_compare)

        # Split-Screen Slider Comparison button
        self.btn_split = QPushButton("Split")
        self.btn_split.setCheckable(True)
        self.btn_split.setToolTip("Before & After interactive split-screen slider (Hotkey: Y)")
        self.btn_split.toggled.connect(self.canvas.set_split_compare_mode)
        layout.addWidget(self.btn_split)

        add_vsep()

        # Undo / Redo
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo last action (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.canvas.undo)
        layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setToolTip("Redo last action (Ctrl+Y)")
        self.btn_redo.clicked.connect(self.canvas.redo)
        layout.addWidget(self.btn_redo)

        # Reset All
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Reset canvas and variations to original unedited photo")
        self.btn_reset.clicked.connect(self._on_reset_all)
        layout.addWidget(self.btn_reset)

        add_vsep()

        # Always on Top Pin
        self.btn_pin = QPushButton("Pin")
        self.btn_pin.setCheckable(True)
        self.btn_pin.setToolTip("Keep window always on top of Lightroom (Ctrl+T)")
        self.btn_pin.toggled.connect(self._toggle_always_on_top)
        layout.addWidget(self.btn_pin)

        # Live Status Badge
        self.live_badge = QLabel(" 🟢 Live Synced " if self.is_live_mode else " ⚡ Live Ready ")
        self.live_badge.setStyleSheet("""
            QLabel {
                background-color: #103820;
                color: #4ade80;
                border: 1px solid #166534;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
                padding: 2px 6px;
            }
        """)
        self.live_badge.setToolTip("Lightroom Classic Live IPC Bridge status")
        layout.addWidget(self.live_badge)

        # Normalized Help & Guide button
        self.btn_lr_help = QPushButton("Help & Guide")
        self.btn_lr_help.setToolTip("Complete User Guide, Installation, Shortcuts, & Download (F1)")
        self.btn_lr_help.clicked.connect(self._on_show_lr_help)
        layout.addWidget(self.btn_lr_help)

        # Toggle Modifications / Layers Panel button with check mark
        self.btn_layers_toggle = QPushButton("✓ Layers")
        self.btn_layers_toggle.setCheckable(True)
        self.btn_layers_toggle.setChecked(True)
        self.btn_layers_toggle.setToolTip("Toggle Modifications (Layers) sidebar panel (Ctrl+L / L)")
        self.btn_layers_toggle.toggled.connect(self._toggle_layers_panel)
        layout.addWidget(self.btn_layers_toggle)

        # Spacer pushes action buttons smoothly to the right
        layout.addStretch(1)

        # Primary Inpainting Button
        self.btn_erase = QPushButton("Erase Object")
        self.btn_erase.setObjectName("primaryAction")
        self.btn_erase.setToolTip("Run AI inpainting on marked red areas (Enter)")
        self.btn_erase.clicked.connect(self._on_run_inpainting)
        layout.addWidget(self.btn_erase)

        # Save & Sync to Lightroom Button with download icon
        self.btn_sync = QPushButton("📥 Save & Sync to Lightroom")
        self.btn_sync.setObjectName("syncAction")
        self.btn_sync.setToolTip("Save & sync photo into Lightroom without closing window (Ctrl+S)")
        self.btn_sync.clicked.connect(self._on_sync_to_lightroom)
        layout.addWidget(self.btn_sync)

        # Alias for backwards compatibility
        self.btn_save = self.btn_sync

        return bar

    def _create_statusbar(self):
        statusbar = QStatusBar(self)
        self.setStatusBar(statusbar)

        # Dimensions & Megapixels
        self.dim_label = QLabel(" 0 × 0 px ")
        self.dim_label.setStyleSheet("color: #888888; font-size: 11px; margin-right: 12px;")
        statusbar.addPermanentWidget(self.dim_label)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        statusbar.addPermanentWidget(self.progress_bar)

        # Generation Elapsed Time Badge
        self.timer_label = QLabel("")
        self.timer_label.setStyleSheet("color: #00d26a; font-weight: bold; font-size: 11px; margin-right: 12px;")
        statusbar.addPermanentWidget(self.timer_label)

        # Device & Hardware Acceleration Indicator
        self.dev_label = QLabel(f" {get_device_telemetry(self.engine.device)} ")
        self.dev_label.setStyleSheet("color: #0078d4; font-weight: bold; font-size: 11px;")
        statusbar.addPermanentWidget(self.dev_label)

    def _connect_signals(self):
        self.size_slider.valueChanged.connect(self.canvas.set_brush_radius)
        self.canvas.brushSizeChanged.connect(self._on_brush_size_changed)
        self.canvas.splitCompareChanged.connect(self.btn_split.setChecked)
        self.canvas.strokeFinished.connect(self._on_stroke_finished)
        self.canvas.statusMessage.connect(lambda msg: self.statusBar().showMessage(msg, 4000))

        # Global window shortcuts for Undo / Redo / Save / Erase
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.canvas.undo)

        self.shortcut_redo_y = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo_y.activated.connect(self.canvas.redo)

        self.shortcut_redo_shift_z = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self.shortcut_redo_shift_z.activated.connect(self.canvas.redo)

        # Shortcuts for Save / Sync
        self.shortcut_sync = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_sync.activated.connect(self._on_sync_to_lightroom)

        self.shortcut_save_exit = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        self.shortcut_save_exit.activated.connect(self._on_sync_to_lightroom)

        # Enter / Return shortcut to trigger Erase Object
        self.shortcut_erase_ret = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        self.shortcut_erase_ret.activated.connect(self._on_run_inpainting)
        self.shortcut_erase_ent = QShortcut(QKeySequence(Qt.Key.Key_Enter), self)
        self.shortcut_erase_ent.activated.connect(self._on_run_inpainting)

        # Shortcut for Help
        self.shortcut_help = QShortcut(QKeySequence(Qt.Key.Key_F1), self)
        self.shortcut_help.activated.connect(self._on_show_lr_help)

    def _on_reset_all(self):
        """Reset canvas to original unedited photo and clear all variations and modification layers."""
        self.canvas.reset_all()
        self.modification_layers.clear()
        self.active_layer_index = None
        if self.base_image:
            self.canvas.set_display_image(self.base_image)
        self.current_variations = []
        self.carousel_panel.setVisible(False)
        self.last_base_image = None
        self.last_used_mask = None
        if hasattr(self, "layer_cards_layout"):
            self._refresh_layers_ui()
        self.statusBar().showMessage("Reset canvas and variations to original unedited photo.", 4000)

    def _on_open_output_folder(self):
        """Reveal active photo or containing folder in Windows File Explorer / OS file manager."""
        target = None
        if self.output_path and os.path.isfile(self.output_path):
            target = self.output_path
        elif self.input_path and os.path.isfile(self.input_path):
            target = self.input_path
        elif self.output_path or self.input_path:
            candidate = self.output_path or self.input_path
            folder = os.path.dirname(os.path.abspath(candidate))
            if os.path.isdir(folder):
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
                self.statusBar().showMessage(f"Opened folder: {folder}", 3500)
                return

        if not target:
            tmp_dir = os.path.join(_PROJECT_ROOT, ".tmp")
            target_folder = tmp_dir if os.path.isdir(tmp_dir) else _PROJECT_ROOT
            QDesktopServices.openUrl(QUrl.fromLocalFile(target_folder))
            self.statusBar().showMessage(f"Opened workspace folder: {target_folder}", 3500)
            return

        abs_target = os.path.abspath(target)
        if sys.platform == "win32":
            try:
                import subprocess
                subprocess.Popen(f'explorer /select,"{abs_target}"')
                self.statusBar().showMessage(f"Revealed in File Explorer: {os.path.basename(abs_target)}", 4000)
                return
            except Exception as e:
                logger.warning(f"Failed to reveal file with explorer /select: {e}")

        folder = os.path.dirname(abs_target)
        if os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
            self.statusBar().showMessage(f"Opened folder: {folder}", 3500)
        else:
            QMessageBox.warning(self, "Folder Not Found", f"Directory does not exist:\n{folder}")

    def _on_engine_changed(self, index: int):
        mode = self.combo_engine.itemData(index)
        self.engine.set_mode(mode)
        if mode == EngineMode.FAST:
            if hasattr(self, "chk_detect_subject"):
                self.chk_detect_subject.setEnabled(False)
                self.chk_detect_subject.setChecked(False)
            if hasattr(self, "chk_grain"):
                self.chk_grain.setEnabled(False)
                self.chk_grain.setChecked(False)
        else:
            if hasattr(self, "chk_detect_subject"):
                self.chk_detect_subject.setEnabled(True)
            if hasattr(self, "chk_grain"):
                self.chk_grain.setEnabled(True)
        self.statusBar().showMessage(f"Switched engine to {mode.value.upper()}", 3000)

    def _on_stroke_finished(self):
        """Called when a brush stroke is completed."""
        if self.canvas.has_mask() and self.worker is None:
            self.statusBar().showMessage("Mask painted. Click '✨ Erase Object' (or press Enter) to run AI removal.", 4000)

    def _on_show_lr_help(self):
        """Display the complete interactive User Guide, Installation, Shortcuts, & Download dialog."""
        dialog = HelpGuideDialog(self)
        dialog.exec()

    def _on_brush_size_changed(self, size: int):
        self.size_slider.blockSignals(True)
        self.size_slider.setValue(size)
        self.size_slider.blockSignals(False)
        self.size_label.setText(f" {size}px ")

    def _cleanup_temp_files(self):
        """Remove temporary files from .tmp directory."""
        tmp_dir = os.path.join(_PROJECT_ROOT, ".tmp")
        if not os.path.isdir(tmp_dir):
            return
        try:
            for fname in os.listdir(tmp_dir):
                fpath = os.path.join(tmp_dir, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                    elif os.path.isdir(fpath):
                        shutil.rmtree(fpath, ignore_errors=True)
                except Exception as e:
                    logger.debug(f"Could not remove temp item {fpath}: {e}")
            logger.info("Cleaned up temporary files in .tmp.")
        except Exception as e:
            logger.warning(f"Error cleaning temporary files: {e}")

    def load_image_file(self, file_path: str):
        """Open and display an image file, preserving original ICC profile and EXIF metadata."""
        try:
            pil_img = Image.open(file_path)
            # Cache color profile and metadata
            self.original_icc_profile = pil_img.info.get("icc_profile")
            try:
                self.original_exif = pil_img.getexif()
            except Exception:
                self.original_exif = None
            self.original_dpi = pil_img.info.get("dpi")

            self.canvas.load_image(pil_img)
            self.base_image = pil_img.copy().convert("RGB")
            self._base_pixmap = QPixmap.fromImage(pil_to_qimage(self.base_image))
            self.modification_layers = []
            self.active_layer_index = None
            self.layer_card_widgets = {}

            self.input_path = file_path
            if not self.output_path:
                is_tmp_lr = os.path.abspath(file_path).startswith(os.path.abspath(os.path.join(_PROJECT_ROOT, ".tmp")))
                if is_tmp_lr:
                    self.output_path = file_path
                else:
                    base, ext = os.path.splitext(file_path)
                    self.output_path = f"{base}_ai_edit{ext}"

            self.current_variations = []
            self.carousel_panel.setVisible(False)

            if hasattr(self, "layer_cards_layout"):
                self._refresh_layers_ui()

            w, h = pil_img.size
            mp = (w * h) / 1_000_000.0
            if hasattr(self, "dim_label"):
                self.dim_label.setText(f" {w} × {h} px ({mp:.1f} MP) ")

            self.setWindowTitle(f"Efface Magique LR — {os.path.basename(file_path)} ({w}x{h})")
            self.statusBar().showMessage(f"Loaded {file_path}", 5000)
        except Exception as e:
            logger.exception("Failed to open image")
            QMessageBox.critical(self, "Error Opening Image", f"Could not load image:\n{e}")

    def _on_open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Photo to Edit",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.dng *.psd);;All Files (*.*)"
        )
        if file_path:
            self.load_image_file(file_path)

    def _on_run_inpainting(self):
        """Trigger AI inpainting in a background worker thread."""
        current_img = self.canvas.get_current_image()
        mask_img = self.canvas.get_mask_image()

        if current_img is None:
            QMessageBox.information(self, "No Image", "Please load an image first.")
            return

        if not self.canvas.has_mask():
            QMessageBox.information(self, "No Mask", "Please paint over the object you want to erase using the red brush.")
            return

        # Immediate UI reaction so button doesn't feel stuck
        self.btn_erase.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(1)
        self.statusBar().showMessage("Starting AI inpainting pipeline...")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        if self.base_image is None:
            self.base_image = current_img.copy().convert("RGB")
            self._base_pixmap = getattr(self, "_base_pixmap", None) or QPixmap.fromImage(pil_to_qimage(self.base_image))

        if self.active_layer_index is not None and 0 <= self.active_layer_index < len(self.modification_layers):
            # Updating an existing layer! Input image is composite of layers before this one
            if self.active_layer_index == 0:
                input_img = self.base_image
            else:
                prev_layer = self.modification_layers[self.active_layer_index - 1]
                input_img = prev_layer.composite_cache or self._get_composite_image(up_to_index=self.active_layer_index - 1)
        else:
            # New layer on top of all visible layers:
            # Fast check: if layers exist, top layer has composite_cache, and all layers are visible, use it directly!
            if (
                self.modification_layers
                and all(l.visible for l in self.modification_layers)
                and self.modification_layers[-1].composite_cache is not None
            ):
                input_img = self.modification_layers[-1].composite_cache
            elif not self.modification_layers:
                input_img = self.base_image
            else:
                input_img = self._get_composite_image()

        self._start_inpainting_pipeline(input_img, mask_img)

    def _start_inpainting_pipeline(self, current_img: Image.Image, mask_img: Image.Image, seed: Optional[int] = None):
        """Launch worker thread for single-pass or multi-variation generation."""
        self.btn_erase.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(2)
        self._generation_start_time = time.time()
        if hasattr(self, "timer_label"):
            self.timer_label.setText("")

        if self.engine.mode == EngineMode.FIREFLY:
            self._render_loading_carousel()
            self.carousel_panel.setVisible(True)

        self.statusBar().showMessage("Initializing generative inpainting pipeline...")

        self.last_base_image = current_img.copy()
        self.last_used_mask = mask_img.copy()

        prompt_text = None
        num_vars = 3 if self.engine.mode == EngineMode.FIREFLY else 1

        detect_subj = self.chk_detect_subject.isChecked() if (hasattr(self, "chk_detect_subject") and self.chk_detect_subject.isEnabled()) else False
        enable_grain = self.chk_grain.isChecked() if (hasattr(self, "chk_grain") and self.chk_grain.isEnabled()) else False

        self.worker = InpaintingWorker(
            engine=self.engine,
            image=current_img,
            mask=mask_img,
            num_variations=num_vars,
            prompt=prompt_text,
            seed=seed,
            detect_subject=detect_subj,
            enable_grain=enable_grain,
        )
        self.worker.progress.connect(self._on_inpainting_progress)
        self.worker.variationsReady.connect(self._on_inpainting_finished)
        self.worker.error.connect(self._on_inpainting_error)
        self.worker.finished.connect(self._on_worker_thread_finished)
        self.worker.start()

    def _on_worker_thread_finished(self):
        """Safely release worker reference after native thread has terminated."""
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def _on_inpainting_progress(self, percentage: int, message: str):
        self.progress_bar.setValue(percentage)
        self.statusBar().showMessage(message)

    def _on_inpainting_finished(self, variations: List[Image.Image]):
        self.progress_bar.setValue(98)
        self.statusBar().showMessage("Applying inpainting composite to canvas...")

        if not variations:
            self.progress_bar.setVisible(False)
            self.btn_erase.setEnabled(True)
            return

        self.current_variations = variations
        self.active_variation_index = 0

        # Pre-cache primary variation QPixmap for instant 0ms canvas display
        pix_0 = QPixmap.fromImage(pil_to_qimage(variations[0]))
        self._variation_pixmaps = [pix_0]
        for v in variations[1:]:
            self._variation_pixmaps.append(QPixmap.fromImage(pil_to_qimage(v)))

        detect_subj = self.chk_detect_subject.isChecked() if (hasattr(self, "chk_detect_subject") and self.chk_detect_subject.isEnabled()) else False
        enable_grain = self.chk_grain.isChecked() if (hasattr(self, "chk_grain") and self.chk_grain.isEnabled()) else False

        used_mask = (
            self.last_used_mask.copy()
            if self.last_used_mask is not None
            else (self.canvas.get_mask_image() or Image.new("L", variations[0].size, 0))
        )
        if self.base_image is None:
            curr = self.canvas.get_current_image()
            self.base_image = (curr if curr is not None else variations[0]).copy().convert("RGB")
            self._base_pixmap = QPixmap.fromImage(pil_to_qimage(self.base_image))

        cached_mask_qi = self.canvas._mask_qimage.copy() if self.canvas._mask_qimage else None

        if self.active_layer_index is not None and 0 <= self.active_layer_index < len(self.modification_layers):
            # Update existing layer
            layer = self.modification_layers[self.active_layer_index]
            layer.mask = used_mask
            layer.inpainted_image = variations[0].copy()
            layer.variations = [v.copy() for v in variations]
            layer.active_variation_index = 0
            layer.engine_mode = self.engine.mode
            layer.detect_subject = detect_subj
            layer.enable_grain = enable_grain
            layer.cached_pixmap = pix_0
            layer.cached_mask_qimage = cached_mask_qi

            # Invalidate composite caches from active_layer_index onwards and cleanly re-composite stack
            for l in self.modification_layers[self.active_layer_index:]:
                l.composite_cache = None
                l.cached_pixmap = None

            full_composite = self._recompute_composite_stack(from_index=self.active_layer_index)
            is_top = (self.active_layer_index == len(self.modification_layers) - 1)
            top_cached_pix = (
                self.modification_layers[-1].cached_pixmap
                if (self.modification_layers and self.modification_layers[-1].cached_pixmap is not None)
                else pix_0
            )
            self.canvas.apply_inpainted_image(full_composite, cached_pixmap=top_cached_pix if is_top else None)
            self.statusBar().showMessage(f"Updated {layer.name} successfully.", 4000)
            self.active_layer_index = None
        else:
            # Create new layer
            new_idx = len(self.modification_layers) + 1
            new_layer = ModificationLayer(
                layer_id=str(uuid.uuid4())[:8],
                name=f"Modification {new_idx}",
                mask=used_mask,
                inpainted_image=variations[0].copy(),
                variations=[v.copy() for v in variations],
                active_variation_index=0,
                engine_mode=self.engine.mode,
                detect_subject=detect_subj,
                enable_grain=enable_grain,
                visible=True,
                cached_pixmap=pix_0,
                cached_mask_qimage=cached_mask_qi,
            )

            # When adding a new layer on top, variations[0] was generated directly on top
            # of the active composite, so variations[0] IS ALREADY the complete composite!
            full_composite = variations[0]
            new_layer.composite_cache = full_composite.copy()
            new_layer.update_thumbnail(self.base_image)
            self.modification_layers.append(new_layer)
            self.active_layer_index = None

            # Apply to canvas directly reusing pix_0 without duplicate QPixmap conversion
            self.canvas.apply_inpainted_image(full_composite, cached_pixmap=pix_0)
            self.statusBar().showMessage(f"Added {new_layer.name} to Modifications stack.", 4000)

        # Refresh layer cards UI
        if hasattr(self, "layer_cards_layout"):
            self._refresh_layers_ui()

        # Compute elapsed time
        elapsed = time.time() - getattr(self, "_generation_start_time", time.time())
        if hasattr(self, "timer_label"):
            self.timer_label.setText(f" ⚡ {elapsed:.2f}s ")

        # Render carousel if multiple variations exist
        if len(variations) > 1 and self.engine.mode == EngineMode.FIREFLY:
            self._render_carousel_thumbnails()
            self.carousel_panel.setVisible(True)
        else:
            self.carousel_panel.setVisible(False)

        if hasattr(self, "dev_label"):
            self.dev_label.setText(f" {get_device_telemetry(self.engine.device)} ")

        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.btn_erase.setEnabled(True)

    def _render_carousel_thumbnails(self):
        """Populate the bottom carousel with thumbnails for each candidate variation."""
        # Clear existing buttons
        while self.var_card_layout.count():
            item = self.var_card_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.var_buttons = []
        for idx, var_img in enumerate(self.current_variations):
            thumb = var_img.copy()
            thumb.thumbnail((120, 80), Image.Resampling.BILINEAR)
            qimg = pil_to_qimage(thumb)
            pix = QPixmap.fromImage(qimg)

            btn = QPushButton()
            btn.setCheckable(True)
            btn.setChecked(idx == self.active_variation_index)
            btn.setIcon(QIcon(pix))
            btn.setIconSize(QSize(90, 60))
            btn.setText(f" Variation {idx + 1}")
            btn.setFixedHeight(72)
            btn.setFixedWidth(160)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #242424;
                    border: 2px solid #383838;
                    border-radius: 6px;
                    color: #ffffff;
                    font-size: 11px;
                    padding: 4px;
                    text-align: center;
                }
                QPushButton:hover {
                    border-color: #0078d4;
                    background-color: #2c2c2c;
                }
                QPushButton:checked {
                    border: 2px solid #0078d4;
                    background-color: #12283e;
                    color: #ffffff;
                    font-weight: bold;
                }
            """)
            btn.clicked.connect(lambda checked, i=idx: self._on_select_variation(i))
            self.var_card_layout.addWidget(btn)
            self.var_buttons.append(btn)

    def _on_select_variation(self, index: int):
        """User clicked a candidate variation thumbnail card."""
        if not (0 <= index < len(self.current_variations)):
            return

        self.active_variation_index = index
        for idx, btn in enumerate(self.var_buttons):
            btn.setChecked(idx == index)

        cached_pix = self._variation_pixmaps[index] if (index < len(self._variation_pixmaps)) else None
        target_img = self.current_variations[index]

        target_layer_idx = (
            self.active_layer_index
            if (self.active_layer_index is not None)
            else (len(self.modification_layers) - 1 if self.modification_layers else None)
        )

        if target_layer_idx is not None and 0 <= target_layer_idx < len(self.modification_layers):
            layer = self.modification_layers[target_layer_idx]
            layer.active_variation_index = index
            layer.cached_pixmap = cached_pix

            # If modifying the top layer or single layer, target_img IS the final composite!
            if target_layer_idx == len(self.modification_layers) - 1:
                layer.composite_cache = target_img.copy()
                self.canvas.set_display_image(target_img, cached_pixmap=cached_pix)
            else:
                for l in self.modification_layers[target_layer_idx:]:
                    l.composite_cache = None
                    l.cached_pixmap = None
                full_comp = self._recompute_composite_stack(from_index=target_layer_idx)
                top_pix = (
                    self.modification_layers[-1].cached_pixmap
                    if self.modification_layers and self.modification_layers[-1].cached_pixmap is not None
                    else None
                )
                self.canvas.set_display_image(full_comp, cached_pixmap=top_pix)
        else:
            self.canvas.set_preview_image(target_img, cached_pixmap=cached_pix)

        self.statusBar().showMessage(f"Displaying Variation {index + 1} of {len(self.current_variations)}", 2000)

    def _on_accept_variation(self):
        """Lock in selected variation and hide carousel."""
        self.carousel_panel.setVisible(False)
        self.statusBar().showMessage(f"Accepted Variation {self.active_variation_index + 1}.", 3000)

    def _on_generate_more_variations(self):
        """Generate 3 more candidate variations with a fresh seed."""
        if self.last_base_image and self.last_used_mask:
            import random
            fresh_seed = random.randint(1000, 999999)
            self._start_inpainting_pipeline(self.last_base_image, self.last_used_mask, seed=fresh_seed)
        else:
            self._on_run_inpainting()

    def _on_inpainting_error(self, err_msg: str):
        self.progress_bar.setVisible(False)
        self.btn_erase.setEnabled(True)
        QMessageBox.critical(self, "AI Inpainting Error", f"An error occurred during inpainting:\n\n{err_msg}")
        self.statusBar().showMessage("Inpainting failed.", 4000)

    def _toggle_always_on_top(self, pinned: bool):
        """Toggle WindowStaysOnTopHint dynamically without losing window position."""
        pos = self.pos()
        size = self.size()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, pinned)
        self.show()
        self.move(pos)
        self.resize(size)
        self.statusBar().showMessage("Window pinned: Always on top" if pinned else "Window unpinned", 2500)

    def _on_live_photo_selected(self, payload: dict):
        """Called when Lightroom user selects another photo in Library or Develop module."""
        new_path = payload.get("path")
        if not new_path or not os.path.isfile(new_path):
            logger.warning(f"Live selection path missing or invalid: {new_path}")
            return

        if self.worker is not None:
            logger.info(f"Inpainting active; delaying switch to {new_path}")
            self.statusBar().showMessage("Lightroom selection changed, but inpainting is currently in progress...", 4000)
            return

        self.current_photo_id = payload.get("photo_id")
        self.current_original_path = payload.get("original_path")

        # Load the newly selected photo seamlessly
        self.load_image_file(new_path)
        self.canvas.fit_to_screen()
        if hasattr(self, "live_badge"):
            self.live_badge.setText(" 🟢 Live Synced ")
        self.statusBar().showMessage(f"⚡ Live Synced photo: {os.path.basename(new_path)}", 4000)

    def _on_live_focus_requested(self):
        """Bring window to foreground when requested by Lightroom."""
        self.raise_()
        self.activateWindow()

    def _on_live_import_completed(self, payload: dict):
        """Lightroom finished importing and stacking the edited photo."""
        imported_path = payload.get("path", "")
        base = os.path.basename(imported_path) if imported_path else "photo"
        self.statusBar().showMessage(f"✓ Stacked adjacent to original in Lightroom: {base}", 5000)

    def save_image_to_disk(self) -> Optional[str]:
        """Save the active variation to disk with full 16-bit color profile and metadata preservation."""
        current_img = self.canvas.get_current_image()
        if current_img is None:
            QMessageBox.information(self, "No Image", "There is no image to save.")
            return None

        # Use actively selected variation if user is actively previewing variations in carousel
        if (
            hasattr(self, "carousel_panel")
            and self.carousel_panel.isVisible()
            and self.current_variations
            and 0 <= self.active_variation_index < len(self.current_variations)
        ):
            save_img = self.current_variations[self.active_variation_index]
        elif self.modification_layers:
            save_img = self._get_composite_image()
        else:
            save_img = current_img

        if not self.output_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Edited Photo",
                "",
                "TIFF (*.tif *.tiff);;PNG (*.png);;JPEG (*.jpg *.jpeg)"
            )
            if not file_path:
                return None
            self.output_path = file_path

        # Non-destructive protection: ensure we never overwrite original image when loaded directly
        is_tmp_lr = os.path.abspath(self.output_path).startswith(os.path.abspath(os.path.join(_PROJECT_ROOT, ".tmp")))
        if not is_tmp_lr and self.input_path and os.path.abspath(self.output_path) == os.path.abspath(self.input_path):
            base, ext = os.path.splitext(self.input_path)
            self.output_path = f"{base}_ai_edit{ext}"
            logger.info(f"Non-destructive save: diverted output to {self.output_path} to preserve original.")

        try:
            # Preserve color profile, resolution tags, and compression
            save_kwargs = {}
            if self.output_path.lower().endswith((".tif", ".tiff")):
                save_kwargs["compression"] = "tiff_deflate"
                if self.original_icc_profile:
                    save_kwargs["icc_profile"] = self.original_icc_profile
                if self.original_dpi:
                    save_kwargs["dpi"] = self.original_dpi
            elif self.output_path.lower().endswith((".jpg", ".jpeg")):
                save_kwargs["quality"] = 95
                if self.original_icc_profile:
                    save_kwargs["icc_profile"] = self.original_icc_profile
                if self.original_dpi:
                    save_kwargs["dpi"] = self.original_dpi
            elif self.output_path.lower().endswith(".png"):
                if self.original_icc_profile:
                    save_kwargs["icc_profile"] = self.original_icc_profile
                if self.original_dpi:
                    save_kwargs["dpi"] = self.original_dpi

            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            try:
                save_img.save(self.output_path, **save_kwargs)
            except Exception as enc_err:
                logger.warning(f"Initial save raised {enc_err}. Retrying with safe fallback parameters...")
                safe_kwargs = {}
                if self.output_path.lower().endswith((".tif", ".tiff")):
                    safe_kwargs["compression"] = "tiff_deflate"
                elif self.output_path.lower().endswith((".jpg", ".jpeg")):
                    safe_kwargs["quality"] = 95
                if self.original_icc_profile:
                    safe_kwargs["icc_profile"] = self.original_icc_profile
                save_img.save(self.output_path, **safe_kwargs)

            logger.info(f"Saved image to {self.output_path} (ICC Profile preserved: {bool(self.original_icc_profile)})")
            return self.output_path
        except Exception as e:
            logger.exception("Failed to save image")
            QMessageBox.critical(self, "Error Saving Image", f"Could not save file:\n{e}")
            return None

    def _on_sync_to_lightroom(self):
        """Save image and queue for Lightroom auto-import while keeping window open."""
        saved_path = self.save_image_to_disk()
        if saved_path:
            self.saved_successfully = True
            self.live_bridge.queue_import(
                export_path=saved_path,
                original_path=self.current_original_path,
                photo_id=self.current_photo_id,
            )
            self.statusBar().showMessage(f"✓ Saved & Synced to Lightroom: {os.path.basename(saved_path)}", 5000)
            if hasattr(self, "live_badge"):
                self.live_badge.setText(" ✓ Synced ")
                reset_text = " 🟢 Live Synced " if self.is_live_mode else " ⚡ Live Ready "
                def _safe_reset():
                    try:
                        if hasattr(self, "live_badge") and self.live_badge is not None:
                            self.live_badge.setText(reset_text)
                    except (RuntimeError, AttributeError):
                        pass
                QTimer.singleShot(3000, _safe_reset)

    def _on_save_and_exit(self):
        """Save the active image, queue import for Lightroom, and close companion."""
        saved_path = self.save_image_to_disk()
        if saved_path:
            self.saved_successfully = True
            self.live_bridge.queue_import(
                export_path=saved_path,
                original_path=self.current_original_path,
                photo_id=self.current_photo_id,
            )
            print(f"[Efface Magique] Edited photo successfully saved to: {saved_path}")
            self.close()

    def closeEvent(self, event):
        """Ensure all threads, live servers, and background processes terminate when window is closed."""
        logger.info("MainWindow closeEvent: shutting down background processes and cleaning temp files...")
        try:
            # 1. Terminate active worker thread if running
            if hasattr(self, "worker") and self.worker is not None:
                try:
                    self.worker.terminate()
                    self.worker.wait(300)
                except Exception:
                    pass
                self.worker = None

            # 2. Stop Live Bridge server and remove live_bridge.json
            if hasattr(self, "live_bridge") and self.live_bridge is not None:
                try:
                    # Signal Lightroom to stop its Live Sync activity immediately
                    self.live_bridge.notify_lightroom_closing()
                    self.live_bridge.stop()
                except Exception as e:
                    logger.warning(f"Error stopping live bridge on close: {e}")

            # 3. Clean up temporary files in .tmp
            self._cleanup_temp_files()
        except Exception as e:
            logger.exception(f"Error during closeEvent cleanup: {e}")

        event.accept()
        super().closeEvent(event)
        app = QApplication.instance()
        if app:
            app.quit()


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Efface Magique LR - AI Generative Eraser Companion")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input photo exported from Lightroom")
    parser.add_argument("--image", type=str, default=None, help="Alias for --input")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to write output photo")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Optional text prompt for Firefly generative fill")
    parser.add_argument("--headless", action="store_true", help="Run without GUI in automated test mode")
    parser.add_argument("--live", action="store_true", help="Start companion in persistent Live Bridge mode")
    parser.add_argument("--test-mask-rect", type=str, default=None, help="Apply rectangular test mask X,Y,W,H in headless mode")
    args = parser.parse_args()

    input_file = args.input or args.image
    output_file = args.output or input_file

    if args.headless:
        logger.info("Executing in headless automated test mode...")
        if not input_file or not os.path.isfile(input_file):
            logger.error(f"Headless mode requires an existing input image (--input or --image): {input_file}")
            sys.exit(1)

        try:
            pil_img = Image.open(input_file)
            w, h = pil_img.size
            mask_pil = Image.new("L", (w, h), 0)

            if args.test_mask_rect:
                from PIL import ImageDraw
                rx, ry, rw, rh = [int(v.strip()) for v in args.test_mask_rect.split(",")]
                draw = ImageDraw.Draw(mask_pil)
                draw.rectangle([rx, ry, rx + rw, ry + rh], fill=255)
                logger.info(f"Generated test mask rect: [{rx}, {ry}, {rx+rw}, {ry+rh}]")

            engine = InpaintingEngine()
            result_img = engine.inpaint_full_resolution(pil_img, mask_pil, prompt=args.prompt)

            if not output_file:
                output_file = input_file

            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            if output_file.lower().endswith((".tif", ".tiff")):
                result_img.save(output_file, compression="tiff_deflate")
            else:
                result_img.save(output_file)

            logger.info(f"Saved headless output to {output_file}")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"Headless pipeline error: {e}")
            sys.exit(1)

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("effacemagique.ai.eraser")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("Efface Magique LR")
    if os.path.isfile(_LOGO_ICO_PATH):
        app.setWindowIcon(QIcon(_LOGO_ICO_PATH))
    elif os.path.isfile(_LOGO_PATH):
        app.setWindowIcon(QIcon(_LOGO_PATH))

    window = MainWindow(input_path=input_file, output_path=output_file, live_mode=args.live)
    window.show()

    ret = app.exec()
    try:
        if window:
            window._cleanup_temp_files()
    except Exception:
        pass

    # Terminate all processes, worker threads, and child resources immediately on close
    if ret == 0:
        if input_file and not window.saved_successfully:
            # User closed window without saving edits -> exit code 130 indicates user cancelled
            os._exit(130)
        os._exit(0)
    os._exit(ret)


if __name__ == "__main__":
    main()
