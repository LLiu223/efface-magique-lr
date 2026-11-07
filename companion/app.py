"""
app.py
Efface Magique LR - Adobe Lightroom Classic AI Companion Application
Dual-Mode Inpainting Interface (LaMa Variations & Fast)

Key Capabilities:
- Model dropdown filled from companion.models.registry (default LaMa), saved in QSettings
- Dual Inpainting Engine (pipeline mode, independent of the model):
  * ✨ LaMa Variations mode (model inpainting with slight seeded variations, 3 candidates;
    the optional prompt only adds a deterministic colour tint of a few levels)
  * ⚡ Fast Mode (single model pass for spots, wires, and quick fixes)
- Variations Carousel Panel with real-time viewport updates & "More Variations" regeneration
- Camera Sensor Grain & Noise Matching
- Split-Screen & Hold-to-Compare (\\ and Space keys)
- Color profile (ProPhoto RGB / Adobe RGB / sRGB) & DPI preservation
- 16-bit TIFFs: untouched pixels keep full 16-bit precision; the erased area is computed in
  8-bit and merged back; preview is 8-bit
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

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, qInstallMessageHandler, QtMsgType, QUrl, QSettings
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

from companion.ui import (
    _ASSETS_DIR,
    _CHECKMARK_PATH,
    _LOGO_PATH,
    _LOGO_ICO_PATH,
    DARK_STYLE,
    HelpGuideDialog,
    LoadingSpinnerOverlay,
)
from companion.canvas import ImageCanvas, pil_to_qimage
from companion.device import (
    DeviceType,
    DeviceInfo,
    get_optimal_device,
    get_device_telemetry,
    configure_cpu_threading,
    setup_device_optimizations,
    free_device_memory,
)
from companion.model_engine import (
    EngineMode,
    InpaintingEngine,
    InpaintingWorker,
)
from companion.models import registry as model_registry, DEFAULT_MODEL
from companion.layers import ModificationLayer, create_layer_thumbnail
from companion import __version__
from companion.diagnostics import build_diagnostics
from companion.live_bridge import LiveBridgeServer
from companion.config import CONFIG
from companion import recovery
from companion.utils.tiff16 import read_rgb16, write_tiff16, merge_into_16bit
from companion.config import CONFIG
from companion import agent as agent_pkg
from companion.agent.harness import AgentWorker
from companion.agent.llm import PROVIDERS as AGENT_PROVIDERS, is_local_url
from companion.agent.tools import AgentSession

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

# Outside .tmp/ on purpose: _cleanup_temp_files wipes .tmp on every close
_RECOVERY_ROOT = os.path.join(_PROJECT_ROOT, ".recovery")


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


MODEL_SETTINGS_KEY = "model/name"


def _open_settings() -> QSettings:
    return QSettings("EffaceMagique", "EffaceMagiqueLR")


def restore_model_name(settings: QSettings) -> str:
    """Saved model name, or DEFAULT_MODEL (with a warning) if it is unknown or unavailable."""
    name = settings.value(MODEL_SETTINGS_KEY, DEFAULT_MODEL, type=str)
    try:
        if model_registry.get(name).is_available() or name == DEFAULT_MODEL:
            return name
        logger.warning(f"Saved model {name!r} is not available; falling back to {DEFAULT_MODEL!r}.")
    except KeyError:
        logger.warning(f"Saved model {name!r} is unknown; falling back to {DEFAULT_MODEL!r}.")
    return DEFAULT_MODEL


# -----------------------------------------------------------------------------
# Background Worker Thread for Non-Blocking Image Loading
# -----------------------------------------------------------------------------

class ImageLoaderWorker(QThread):
    """Background worker for loading high-resolution images without GUI freezing."""
    loaded = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            pil_img = Image.open(self.file_path)
            pil_img.load()
            self.loaded.emit(pil_img, self.file_path)
        except Exception as e:
            self.error.emit(str(e))


# -----------------------------------------------------------------------------
# Main Application Window
# -----------------------------------------------------------------------------


class MainWindow(QMainWindow):
    # Emitted from the prewarm thread; AutoConnection queues it onto the GUI thread
    modelLoadFailed = pyqtSignal(str)
    _prewarm_finished = pyqtSignal()  # emitted from the warm-up daemon thread, delivered queued on the GUI thread

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
        # Per-photo: True once the current layers were synced to Lightroom
        self._layers_synced = False
        # Latest Lightroom selection received while inpainting was running
        self._pending_select: Optional[dict] = None
        # True while the discard prompt is open (its nested event loop keeps delivering selections)
        self._confirming_switch = False
        # Last user-visible error message, shown in Copy diagnostics
        self.last_error: Optional[str] = None

        # Metadata preservation
        self.original_icc_profile = None
        self.original_exif = None
        self.original_dpi = None
        # Lossless uint16 RGB pixels of a 16-bit TIFF input (None for every other input)
        self.base_image16: Optional[np.ndarray] = None

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
        # Layer-op history: (op, index, layer) with op in {"add", "delete", "toggle"}; no image copies
        self._layer_undo: List[tuple] = []
        self._layer_redo: List[tuple] = []
        # True when layers changed since the last recovery autosave
        self._autosave_dirty = False

        # Optional assistant: pending proposal (nothing is committed until Apply)
        self._agent_proposal = None
        self._agent_prev_display: Optional[Image.Image] = None
        self._agent_state_key = None
        self._usage_store = None

        self.setWindowTitle("Efface Magique LR - LaMa Variations Eraser")
        if os.path.isfile(_LOGO_PATH):
            self.setWindowIcon(QIcon(_LOGO_PATH))
        self.resize(1400, 920)
        self.setStyleSheet(DARK_STYLE)

        # Inpainting Engine instance & Hardware Detection
        device = get_optimal_device()
        setup_device_optimizations(device)
        self.engine = InpaintingEngine(device=device, mode=EngineMode.FIREFLY)
        self.worker: Optional[InpaintingWorker] = None
        # Restore the model before the warm-up thread starts so it loads the right one
        self._settings = _open_settings()
        self.engine.set_model(restore_model_name(self._settings))

        self._pending_model: Optional[str] = None  # model picked while warm-up held the load lock
        self._prewarm_finished.connect(self._apply_pending_model)
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
        self._update_history_buttons()

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(CONFIG.AUTOSAVE_INTERVAL_MS)
        self._autosave_timer.timeout.connect(self._on_autosave_timer)
        self._autosave_timer.start()

        # Asynchronously pre-warm inpainting models in background daemon thread (after the UI exists)
        import threading
        self.modelLoadFailed.connect(self._on_model_load_failed)
        self._prewarm_thread = threading.Thread(target=self._prewarm_engine, daemon=True)
        self._prewarm_thread.start()

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
            if self.engine.load_error is None:
                self.engine.load_error = str(e)
        finally:
            try:
                if self.engine.load_error is not None:  # before the deferred switch, so it reports the warmed model
                    self.modelLoadFailed.emit(self.engine.load_error)
                self._prewarm_finished.emit()
            except RuntimeError:  # window already deleted
                pass

    def _on_model_load_failed(self, err: str):
        """GUI thread: non-blocking, persistent status bar warning (no modal)."""
        self.last_error = f"LaMa model failed to load: {err}"
        msg = ("⚠ LaMa model failed to load: weights missing or install broken. "
               "Erasing will not work in Fast mode.")
        self.model_warning_label.setText(msg)
        self.model_warning_label.setToolTip(err)
        self.model_warning_label.setVisible(True)
        self.statusBar().showMessage(msg)

    def diagnostics_info(self) -> dict:
        """Collect values for build_diagnostics (versions, device, bridge, errors only)."""
        import platform
        import PIL
        import torch
        from PyQt6.QtCore import PYQT_VERSION_STR
        dev = self.engine.device
        gpu_name = None
        if dev.type == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(dev)
            except Exception:
                pass
        if not self.engine.is_loaded:
            lama = "not yet loaded"
        elif self.engine._model is not None:
            lama = "loaded"
        else:
            lama = f"failed ({self.engine.load_error})" if self.engine.load_error else "failed"
        bridge = getattr(self, "live_bridge", None)
        return {
            "app_version": __version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pillow": PIL.__version__,
            "pyqt6": PYQT_VERSION_STR,
            "device": dev.type,
            "gpu_name": gpu_name,
            "lama_model": lama,
            "bridge_port": getattr(self, "live_bridge_port", None),
            "bridge_connected": bridge.is_connected if bridge is not None else None,
            "last_error": self.last_error,
        }

    def diagnostics_text(self) -> str:
        return build_diagnostics(self.diagnostics_info())

    def _create_carousel_panel(self) -> QFrame:
        """Create the bottom carousel panel for LaMa candidate variations."""
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

        lbl = QLabel("<b>LaMa Variations:</b>")
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
        header_btn_style = """
            QPushButton {
                background-color: #2b2b2b;
                border: 1px solid #444444;
                border-radius: 4px;
                color: #ffffff;
                font-size: 11px;
                padding: 2px 8px;
            }
            QPushButton:hover:enabled {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            QPushButton:disabled {
                background-color: #242424;
                color: #555555;
                border-color: #303030;
            }
        """
        self.btn_new_layer.clicked.connect(self._on_new_layer_clicked)

        self.btn_layer_undo = QPushButton("Undo")
        self.btn_layer_undo.setToolTip("Undo last layer add/delete/visibility change (Ctrl+Z)")
        self.btn_layer_undo.clicked.connect(self._undo_layer_op)
        self.btn_layer_redo = QPushButton("Redo")
        self.btn_layer_redo.setToolTip("Redo layer change (Ctrl+Y / Ctrl+Shift+Z)")
        self.btn_layer_redo.clicked.connect(self._redo_layer_op)
        for btn in (self.btn_layer_undo, self.btn_layer_redo, self.btn_new_layer):
            btn.setFixedHeight(24)
            btn.setStyleSheet(header_btn_style)
            header_layout.addWidget(btn)

        # Delete layer button (retained on window instance for compatibility, hidden from UI)
        self.btn_delete_layer = QPushButton("Delete")
        self.btn_delete_layer.setVisible(False)

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

        mode_str = "Fast Spot" if layer.engine_mode == EngineMode.FAST else "LaMa Variations"
        detail_lbl = QLabel(f"{mode_str}")
        detail_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        info_layout.addWidget(detail_lbl)

        layout.addLayout(info_layout, stretch=1)

        # 4. Per-Layer Delete Button
        btn_delete = QPushButton("✕")
        btn_delete.setFixedSize(24, 24)
        btn_delete.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_delete.setToolTip(f"Delete {layer.name}")
        btn_delete.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                color: #888888;
                font-size: 11px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
                border-color: #f87171;
                color: #ffffff;
            }
        """)
        btn_delete.clicked.connect(lambda _, i=idx: self._delete_layer(i))
        layout.addWidget(btn_delete)
        card.btn_delete = btn_delete

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
            self._push_layer_op("toggle", index, layer)

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
            self._autosave()

    def _delete_layer(self, index: int):
        """Delete a modification layer and cleanly re-composite the stack."""
        if 0 <= index < len(self.modification_layers):
            layer = self.modification_layers.pop(index)
            self._push_layer_op("delete", index, layer)
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
            self._autosave()

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

                    # Layer results are already feathered by the pipeline: paste them through the
                    # binary mask. Re-blending would fade the edges on every recomposite (not idempotent).
                    box = (x1, y1, x2, y2)
                    sub_mask = Image.fromarray(((mask_np[y1:y2, x1:x2] > 10) * 255).astype(np.uint8))
                    composite.paste(active_img.crop(box).convert("RGB"), (x1, y1), sub_mask)

                    # If the layer has candidate variations, update each variation against pre_layer_composite
                    if layer.variations and len(layer.variations) > 1:
                        for v_idx, var_img in enumerate(layer.variations):
                            if v_idx == layer.active_variation_index:
                                layer.variations[v_idx] = composite.copy()
                            else:
                                var_comp = pre_layer_composite.copy()
                                var_comp.paste(var_img.crop(box).convert("RGB"), (x1, y1), sub_mask)
                                layer.variations[v_idx] = var_comp

            # Store updated composite and pixmap on layer
            layer.composite_cache = composite.copy()
            layer.cached_pixmap = QPixmap.fromImage(pil_to_qimage(composite))
            layer.inpainted_image = composite.copy()
            layer.update_thumbnail(self.base_image)

        return composite

    # -------------------------------------------------------------------------
    # Layer Undo / Redo (add, delete, toggle visibility)
    # -------------------------------------------------------------------------

    def _push_layer_op(self, op: str, index: int, layer: ModificationLayer):
        """Record a new undoable layer op; any new op drops the redo history."""
        self._layer_undo.append((op, index, layer))
        # Dropping the oldest entry releases its layer reference (e.g. a deleted layer)
        del self._layer_undo[:-CONFIG.LAYER_HISTORY_DEPTH]
        self._layer_redo.clear()
        self._layers_synced = False
        self._update_history_buttons()

    def _clear_layer_history(self):
        self._layer_undo.clear()
        self._layer_redo.clear()
        self._update_history_buttons()

    def _update_history_buttons(self):
        if hasattr(self, "btn_layer_undo"):
            self.btn_layer_undo.setEnabled(bool(self._layer_undo))
            self.btn_layer_redo.setEnabled(bool(self._layer_redo))

    def _on_undo(self):
        """Ctrl+Z: unconsumed brush strokes first, then the last layer op."""
        if self.canvas.has_mask() and len(self.canvas._mask_history) > 1:
            self.canvas.undo()
        else:
            self._undo_layer_op()

    def _on_redo(self):
        """Ctrl+Y / Ctrl+Shift+Z: brush strokes first, then layer ops."""
        if self.canvas._mask_redo_stack:
            self.canvas.redo()
        else:
            self._redo_layer_op()

    def _undo_layer_op(self):
        if not self._layer_undo:
            self.statusBar().showMessage("Nothing to undo.", 3000)
            return
        entry = self._layer_undo.pop()
        self._layer_redo.append(entry)
        self._apply_layer_op(*entry, undo=True)

    def _redo_layer_op(self):
        if not self._layer_redo:
            self.statusBar().showMessage("Nothing to redo.", 3000)
            return
        entry = self._layer_redo.pop()
        self._layer_undo.append(entry)
        self._apply_layer_op(*entry, undo=False)

    def _apply_layer_op(self, op: str, index: int, layer: ModificationLayer, undo: bool):
        """Apply (undo=False) or revert (undo=True) one history entry, then recomposite from index."""
        layers = self.modification_layers
        if op == "reset":  # layer is (layers before the reset, their active layer index)
            saved, active = layer
            if undo:
                layers[:] = saved
            else:
                self._reset_layers_state()
        elif op == "toggle":
            layer.visible = not layer.visible
        elif (op == "add") == undo:  # undo add / redo delete
            del layers[index]
        else:  # undo delete / redo add: the same object goes back at its old index
            layers.insert(index, layer)
        self._layers_synced = False

        if self.active_layer_index is not None:
            self.active_layer_index = None
            self.canvas.clear_mask(save_state=False)
        self.carousel_panel.setVisible(False)
        self.current_variations = []

        for l in layers[index:]:
            l.composite_cache = None
            l.cached_pixmap = None
        full = self._recompute_composite_stack(from_index=index)
        top_pix = layers[-1].cached_pixmap if layers else self._base_pixmap
        self.canvas.set_display_image(full, cached_pixmap=top_pix)

        self._refresh_layers_ui()
        self._update_history_buttons()
        if op == "reset":
            if undo and active is not None:
                self._select_layer(active)
            what = "Reset All"
        else:
            what = f"{op} of {layer.name}"
        self.statusBar().showMessage(f"{'Undid' if undo else 'Redid'} {what}.", 3000)
        self._autosave()

    # -------------------------------------------------------------------------
    # Crash-recovery autosave
    # -------------------------------------------------------------------------

    def _recovery_key(self) -> Optional[str]:
        if self.current_photo_id:
            return str(self.current_photo_id)
        path = self.current_original_path or self.input_path
        return os.path.abspath(path) if path else None

    def _recovery_path(self) -> Optional[str]:
        key = self._recovery_key()
        return recovery.recovery_dir(_RECOVERY_ROOT, key) if key else None

    def _autosave(self):
        """Write the layer stack to the recovery folder. Errors are logged and shown, never raised."""
        self._autosave_dirty = False
        path = self._recovery_path()
        if path is None or self.base_image is None:
            return
        try:
            if self.modification_layers:
                recovery.save(path, self._recovery_key(), self.base_image.size, self.modification_layers)
            else:
                recovery.clear(path)  # nothing left to recover
        except Exception as e:
            self._autosave_dirty = True  # retry on the next timer tick
            logger.warning(f"Recovery autosave failed: {e}")
            self.statusBar().showMessage(f"Autosave failed: {e}", 6000)

    def _on_autosave_timer(self):
        if self._autosave_dirty and self.modification_layers and not self._layers_synced:
            self._autosave()

    def _clear_recovery(self):
        path = self._recovery_path()
        if path:
            recovery.clear(path)

    def _offer_recovery(self):
        """After a load: offer back layers autosaved for this photo by a previous session."""
        path = self._recovery_path()
        if path is None or self.base_image is None:
            return
        try:
            entries = recovery.load(path, self._recovery_key(), self.base_image.size)
        except ValueError as e:
            logger.warning(f"Discarding invalid recovery data: {e}")
            recovery.clear(path)
            return
        if entries is None:
            return
        if not entries:
            recovery.clear(path)
            return

        n = len(entries)
        self._confirming_switch = True  # Lightroom selections arriving now go to _pending_select
        try:
            answer = QMessageBox.question(
                self,
                "Recover Layers",
                f"Restore {n} unsaved layer{'' if n == 1 else 's'} from a previous session?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
        finally:
            self._confirming_switch = False
        pending, self._pending_select = self._pending_select, None
        if pending is not None:
            # Apply the newest selection after this load returns (normal switch rules, no stacking)
            QTimer.singleShot(0, lambda: self._on_live_photo_selected(pending))

        if answer != QMessageBox.StandardButton.Yes:
            recovery.clear(path)
            return

        for e in entries:
            x1, y1 = e["box"][:2]
            img = self.base_image.copy()
            img.paste(e["patch"], (x1, y1))
            mask = Image.new("L", self.base_image.size, 0)
            mask.paste(e["mask"], (x1, y1))
            try:
                mode = EngineMode(e["engine_mode"])
            except ValueError:
                mode = EngineMode.FIREFLY
            self.modification_layers.append(ModificationLayer(
                layer_id=e["id"], name=e["name"], mask=mask, inpainted_image=img,
                prompt=e.get("prompt"), engine_mode=mode, seed=e.get("seed"),
                detect_subject=e["detect_subject"], enable_grain=e["enable_grain"], visible=e["visible"],
            ))
        full = self._recompute_composite_stack(from_index=0)
        self.canvas.set_display_image(full, cached_pixmap=self.modification_layers[-1].cached_pixmap)
        self._layers_synced = False
        if hasattr(self, "layer_cards_layout"):
            self._refresh_layers_ui()
        self.statusBar().showMessage(f"Restored {n} layer{'' if n == 1 else 's'} from a previous session.", 5000)

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

            # Same binary-mask paste as _recompute_composite_stack (results are already feathered)
            sub_mask = Image.fromarray(((mask_np[y1:y2, x1:x2] > 10) * 255).astype(np.uint8))
            composite.paste(active_img.crop((x1, y1, x2, y2)).convert("RGB"), (x1, y1), sub_mask)

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
            self.logo_label.setToolTip(f"Efface Magique LR v{__version__} - LaMa Variations Eraser")
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
        self.combo_engine.addItem("LaMa Variations", EngineMode.FIREFLY)
        self.combo_engine.addItem("Fast Spot Removal", EngineMode.FAST)
        self.combo_engine.currentIndexChanged.connect(self._on_engine_changed)
        layout.addWidget(self.combo_engine)

        # Model selector (which inpainting network; independent of the Fast/Generative mode)
        self.combo_model = QComboBox()
        self.combo_model.setToolTip("Inpainting model")
        for cls in model_registry.all():
            self.combo_model.addItem(cls.label, cls.name)
            if not cls.is_available():
                i = self.combo_model.count() - 1
                item = self.combo_model.model().item(i)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                self.combo_model.setItemData(i, cls.unavailable_reason(), Qt.ItemDataRole.ToolTipRole)
        self.combo_model.setCurrentIndex(max(0, self.combo_model.findData(self.engine.model_name)))
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        layout.addWidget(self.combo_model)

        # Prompt for models with uses_prompt (Stable Diffusion); disabled for the others
        self.txt_prompt = QLineEdit()
        self.txt_prompt.setPlaceholderText("Prompt (optional)")
        self.txt_prompt.setAccessibleName("Prompt")
        self.txt_prompt.setToolTip(
            "Text prompt. Only used by Stable Diffusion; greyed out for the other models because they ignore prompts."
        )
        self.txt_prompt.setMaximumWidth(220)
        self._update_prompt_enabled()
        layout.addWidget(self.txt_prompt)

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
        self.btn_undo.clicked.connect(self._on_undo)
        layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setToolTip("Redo last action (Ctrl+Y)")
        self.btn_redo.clicked.connect(self._on_redo)
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

        # Normalized Help/Guide button
        self.btn_lr_help = QPushButton("Help/Guide")
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

        add_vsep()
        self._create_assistant_widgets(layout)

        # Spacer pushes action buttons smoothly to the right
        layout.addStretch(1)

        # Primary Inpainting Button
        self.btn_erase = QPushButton("Erase Object")
        self.btn_erase.setObjectName("primaryAction")
        self.btn_erase.setToolTip("Run AI inpainting on marked red areas (Enter)")
        self.btn_erase.clicked.connect(self._on_run_inpainting)
        layout.addWidget(self.btn_erase)

        # Save to Lightroom Button with download icon
        self.btn_sync = QPushButton("📥 Save to Lightroom")
        self.btn_sync.setObjectName("syncAction")
        self.btn_sync.setToolTip("Save photo into Lightroom without closing window (Ctrl+S)")
        self.btn_sync.clicked.connect(self._on_sync_to_lightroom)
        layout.addWidget(self.btn_sync)

        # Alias for backwards compatibility
        self.btn_save = self.btn_sync

        return bar

    def _create_assistant_widgets(self, layout: QHBoxLayout):
        """Assistant input + provider/model fields; disabled when requirements-agent.txt is missing."""
        self.agent_input = QLineEdit()
        self.agent_input.setPlaceholderText("Assistant: e.g. remove the person on the left")
        self.agent_input.setMinimumWidth(200)
        self.agent_input.returnPressed.connect(self._on_agent_ask)
        self.combo_agent_provider = QComboBox()
        self.combo_agent_provider.addItems(AGENT_PROVIDERS)
        saved = self._settings.value(agent_pkg.KEY_PROVIDER, agent_pkg.DEFAULT_PROVIDER, type=str)
        self.combo_agent_provider.setCurrentIndex(max(0, self.combo_agent_provider.findText(saved)))
        self.combo_agent_provider.currentTextChanged.connect(lambda t: self._settings.setValue(agent_pkg.KEY_PROVIDER, t))
        self.ed_agent_model = QLineEdit(self._settings.value(agent_pkg.KEY_MODEL, agent_pkg.DEFAULT_AGENT_MODEL, type=str))
        self.ed_agent_model.setFixedWidth(110)
        self.ed_agent_model.setToolTip("Assistant LLM model name (e.g. llama3.1, claude-sonnet-4-5, gpt-4o-mini, gemini-2.5-flash)")
        self.ed_agent_model.textChanged.connect(lambda t: self._settings.setValue(agent_pkg.KEY_MODEL, t.strip()))
        self.btn_agent_ask = QPushButton("Ask")
        self.btn_agent_ask.clicked.connect(self._on_agent_ask)
        self.btn_agent_cancel = QPushButton("Stop")
        self.btn_agent_cancel.setToolTip("Cancel the assistant request")
        self.btn_agent_cancel.clicked.connect(self._on_agent_cancel)
        self.btn_agent_apply = QPushButton("Apply")
        self.btn_agent_apply.setObjectName("primaryAction")
        self.btn_agent_apply.setToolTip("Commit the assistant's proposal as a layer")
        self.btn_agent_apply.clicked.connect(self._on_agent_apply)
        self.btn_agent_discard = QPushButton("Discard")
        self.btn_agent_discard.clicked.connect(self._on_agent_discard)
        self.btn_agent_usage = QPushButton("Usage")
        self.btn_agent_usage.setToolTip("Assistant usage: tokens, cost, settings")
        self.btn_agent_usage.clicked.connect(self._on_show_agent_usage)
        for w in (self.agent_input, self.combo_agent_provider, self.ed_agent_model, self.btn_agent_ask,
                  self.btn_agent_cancel, self.btn_agent_apply, self.btn_agent_discard, self.btn_agent_usage):
            layout.addWidget(w)
        for w in (self.btn_agent_cancel, self.btn_agent_apply, self.btn_agent_discard):
            w.setVisible(False)
        self.agent_enabled = agent_pkg.agent_available()
        if not self.agent_enabled:
            for w in (self.agent_input, self.combo_agent_provider, self.ed_agent_model, self.btn_agent_ask):
                w.setEnabled(False)
                w.setToolTip(agent_pkg.MISSING_TOOLTIP)

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

        # Persistent LaMa load-failure warning (hidden unless the prewarm fails)
        self.model_warning_label = QLabel("")
        self.model_warning_label.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 11px; margin-left: 8px;")
        self.model_warning_label.setVisible(False)
        statusbar.addPermanentWidget(self.model_warning_label)

    def _connect_signals(self):
        self.size_slider.valueChanged.connect(self.canvas.set_brush_radius)
        self.canvas.brushSizeChanged.connect(self._on_brush_size_changed)
        self.canvas.splitCompareChanged.connect(self.btn_split.setChecked)
        self.canvas.strokeFinished.connect(self._on_stroke_finished)
        self.canvas.statusMessage.connect(lambda msg: self.statusBar().showMessage(msg, 4000))

        # Global window shortcuts for Undo / Redo / Save / Erase
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self._on_undo)

        self.shortcut_redo_y = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo_y.activated.connect(self._on_redo)

        self.shortcut_redo_shift_z = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self.shortcut_redo_shift_z.activated.connect(self._on_redo)

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
        """Reset canvas to original unedited photo and clear all variations and modification layers (undoable)."""
        if self.modification_layers:
            self._push_layer_op("reset", 0, (list(self.modification_layers), self.active_layer_index))
        self._reset_layers_state()
        if hasattr(self, "layer_cards_layout"):
            self._refresh_layers_ui()
        self.statusBar().showMessage("Reset canvas and variations to original unedited photo.", 4000)
        # No layers left: this drops the on-disk copy like deleting the last layer; undo writes it back
        self._autosave()

    def _reset_layers_state(self):
        """Back to the unedited photo: no layers, no variations, no mask. History is left alone."""
        self._drop_agent_proposal()
        self.canvas.reset_all()
        self.modification_layers.clear()
        self.active_layer_index = None
        if self.base_image:
            self.canvas.set_display_image(self.base_image)
        self.current_variations = []
        self.carousel_panel.setVisible(False)
        self.last_base_image = None
        self.last_used_mask = None

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

    def _on_model_changed(self, index: int):
        name = self.combo_model.itemData(index)
        self._settings.setValue(MODEL_SETTINGS_KEY, name)
        if self.engine.set_model(name, blocking=False):
            self._pending_model = None
            self._show_model_status(index)
            self._update_prompt_enabled(self.combo_model.isEnabled())  # stays locked if a run is in progress
            return
        # Warm-up holds the load lock: never wait on it from the GUI thread; apply once it ends.
        # Erase stays locked meanwhile so nothing runs on the old model.
        self._pending_model = name
        self.btn_erase.setEnabled(False)
        self._update_prompt_enabled(self.combo_model.isEnabled())
        self.statusBar().showMessage(f"Switching to {self.combo_model.itemText(index)} after warm-up...")

    def _apply_pending_model(self):
        """GUI thread, after warm-up released the load lock: apply the last model picked meanwhile."""
        name, self._pending_model = self._pending_model, None
        if name is None:
            return
        try:
            self.engine.set_model(name)
            self._show_model_status(self.combo_model.findData(name))
        except Exception as e:
            logger.exception(f"Deferred switch to model {name!r} failed")
            self.statusBar().showMessage(f"Model switch to {name} failed: {e}", 8000)
        finally:
            if self.worker is None:
                self._set_run_controls_enabled(True)

    def _model_switch_pending(self) -> bool:
        """True (with a status hint) while a deferred switch waits: no erase may start on the old model."""
        if self._pending_model is None:
            return False
        self.statusBar().showMessage("Waiting for model switch...", 3000)
        return True

    def _show_model_status(self, index: int):
        msg = f"Inpainting model: {self.combo_model.itemText(index)}"
        if self.engine.last_unload_error:
            self.statusBar().showMessage(f"{msg} ({self.engine.last_unload_error})", 8000)
        else:
            self.statusBar().showMessage(msg, 3000)

    def _active_model_uses_prompt(self) -> bool:
        # A deferred switch counts already, so the field matches the dropdown during warm-up.
        return model_registry.get(self._pending_model or self.engine.model_name).uses_prompt

    def _update_prompt_enabled(self, enabled: bool = True):
        self.txt_prompt.setEnabled(enabled and self._active_model_uses_prompt())

    def _set_run_controls_enabled(self, enabled: bool):
        """Erase button, Model dropdown and prompt field are locked together for the whole worker run."""
        self.btn_erase.setEnabled(enabled)
        self.combo_model.setEnabled(enabled)
        self._update_prompt_enabled(enabled)

    def _apply_bridge_model(self, payload: dict):
        """Apply the optional (bridge-validated) 'model' of a selection payload; absent keeps the current one."""
        name = payload.get("model")
        if name is not None:
            idx = self.combo_model.findData(name)
            if idx >= 0:
                self.combo_model.setCurrentIndex(idx)

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

    def _on_show_agent_usage(self):
        from companion.agent.usage_dialog import UsageDialog
        UsageDialog(self._agent_usage_store(), self._settings, self).exec()

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

    def load_image_file(self, file_path: str, offer_recovery: bool = True):
        """Open and display an image file, preserving original ICC profile and EXIF metadata."""
        self.base_image16 = None
        self._drop_agent_proposal()  # a proposal belongs to the previous photo
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
            tag = getattr(pil_img, "tag_v2", None)
            if tag is not None and tuple(tag.get(258, ())) == (16, 16, 16):
                arr16 = read_rgb16(file_path)
                if arr16 is not None and arr16.shape[:2] == (self.base_image.height, self.base_image.width):
                    self.base_image16 = arr16
                else:
                    logger.warning("16-bit decode unavailable or size differs from 8-bit decode; saving at 8 bits.")
            self.modification_layers = []
            self.active_layer_index = None
            self.layer_card_widgets = {}
            self._clear_layer_history()

            # A different photo needs its own output path; keep an explicit one for the initial load
            if self.input_path is None or os.path.abspath(file_path) != os.path.abspath(self.input_path):
                self.output_path = None
            self.input_path = file_path
            self._layers_synced = False
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
            if offer_recovery:
                self._offer_recovery()
        except Exception as e:
            logger.exception("Failed to open image")
            self.last_error = f"Could not load image: {e}"
            QMessageBox.critical(self, "Error Opening Image", f"Could not load image:\n{e}")

    def _on_open_file(self):
        if self.worker is not None:
            # The running erase would land its result on the newly opened photo
            self.statusBar().showMessage("Erase in progress; open another photo once it finishes.", 5000)
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Photo to Edit",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.dng *.psd);;All Files (*.*)"
        )
        if file_path and self._confirm_discard_layers():
            self.load_image_file(file_path, offer_recovery=False)
            if self.input_path == file_path:
                # Not a Lightroom selection: Sync must not stack this edit onto the previous LR photo
                self.current_photo_id = None
                self.current_original_path = None
                self._offer_recovery()  # only now is the recovery key this file, not the previous LR photo
        # Selections that arrived during the prompt are superseded by this explicit choice
        self._pending_select = None

    def _confirm_discard_layers(self) -> bool:
        """Return True if the current photo may be replaced (no unsynced layers, or user agreed)."""
        if not self.modification_layers or self._layers_synced:
            return True
        n = len(self.modification_layers)
        # Companion may sit behind Lightroom; bring it forward so the prompt is seen
        self.raise_()
        self.activateWindow()
        self._confirming_switch = True
        try:
            answer = QMessageBox.question(
                self,
                "Unsaved Layers",
                f"Discard {n} unsaved layer{'' if n == 1 else 's'}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
        finally:
            self._confirming_switch = False
        if answer == QMessageBox.StandardButton.Yes:
            self._clear_recovery()
            return True
        self.statusBar().showMessage("Kept current photo; switch cancelled. Sync still targets this photo.", 6000)
        return False

    def _on_run_inpainting(self):
        """Trigger AI inpainting in a background worker thread."""
        if self._model_switch_pending():  # Return/Enter shortcuts bypass the disabled Erase button
            return
        if self.agent_input.hasFocus():
            # Enter in the assistant box is a question, not an erase
            self._on_agent_ask()
            return
        if self.worker is not None:
            return  # single worker slot (inpainting or assistant)
        if self._agent_proposal is not None:
            self._on_agent_discard()
        current_img = self.canvas.get_current_image()
        mask_img = self.canvas.get_mask_image()

        if current_img is None:
            QMessageBox.information(self, "No Image", "Please load an image first.")
            return

        if not self.canvas.has_mask():
            QMessageBox.information(self, "No Mask", "Please paint over the object you want to erase using the red brush.")
            return

        # Immediate UI reaction so button doesn't feel stuck
        self._set_run_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(1)
        self.statusBar().showMessage("Starting AI inpainting pipeline...")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        if self.base_image is None:
            self.base_image = current_img.copy().convert("RGB")
            self._base_pixmap = getattr(self, "_base_pixmap", None) or QPixmap.fromImage(pil_to_qimage(self.base_image))

        self._start_inpainting_pipeline(self._erase_input_image(), mask_img)

    def _erase_input_image(self) -> Image.Image:
        """Image an erase works on: composite below the active layer, or of all visible layers for a new one."""
        if self.active_layer_index is not None and 0 <= self.active_layer_index < len(self.modification_layers):
            # Updating an existing layer! Input image is composite of layers before this one
            if self.active_layer_index == 0:
                return self.base_image
            prev_layer = self.modification_layers[self.active_layer_index - 1]
            return prev_layer.composite_cache or self._get_composite_image(up_to_index=self.active_layer_index - 1)
        # New layer on top of all visible layers:
        # Fast check: if layers exist, top layer has composite_cache, and all layers are visible, use it directly!
        if (
            self.modification_layers
            and all(l.visible for l in self.modification_layers)
            and self.modification_layers[-1].composite_cache is not None
        ):
            return self.modification_layers[-1].composite_cache
        if not self.modification_layers:
            return self.base_image
        return self._get_composite_image()

    # -------------------------------------------------------------------------
    # Optional assistant (companion/agent): ask -> proposal preview -> Apply/Discard
    # -------------------------------------------------------------------------

    def _layer_state_key(self):
        """Everything a proposal's input image depends on; Apply is refused if it changed."""
        return (
            id(self.base_image),
            self.active_layer_index,
            tuple((l.layer_id, l.visible, id(l.inpainted_image)) for l in self.modification_layers),
        )

    def _agent_usage_store(self):
        if self._usage_store is None:
            from companion.agent.usage import UsageStore
            self._usage_store = UsageStore()
        return self._usage_store

    def _on_agent_ask(self):
        if not self.agent_enabled:
            return
        text = self.agent_input.text().strip()
        if not text:
            return
        if self.worker is not None:
            self.statusBar().showMessage("Busy; ask again once the current job finishes.", 4000)
            return
        if self.base_image is None:
            QMessageBox.information(self, "Assistant", "Load a photo first.")
            return
        if self._agent_proposal is not None:
            self._on_agent_discard()

        from companion.agent.usage import make_usage_callback

        provider = self.combo_agent_provider.currentText()
        model = self.ed_agent_model.text().strip()
        url = self._settings.value(agent_pkg.KEY_OLLAMA_URL, "", type=str).strip() or None
        budget = self._settings.value(agent_pkg.KEY_TOKEN_BUDGET, CONFIG.AGENT_TOKEN_BUDGET, type=int)
        store_prompts = self._settings.value(agent_pkg.KEY_STORE_PROMPTS, False, type=bool)
        store = self._agent_usage_store()
        if provider == "ollama" and url and not is_local_url(url):
            self.statusBar().showMessage("Warning: the Ollama host is not localhost; your request leaves this computer.", 6000)

        # Only the request text and region labels reach the model: no image, file name or EXIF
        # The prompt field seeds the session; a set_prompt from the assistant flows back in _on_agent_result
        session = AgentSession(engine=self.engine, image=self._erase_input_image().copy(),
                               prompt=self.txt_prompt.text().strip() or None)
        self._agent_state_key = self._layer_state_key()
        self.worker = AgentWorker(
            self._agent_llm_factory(provider, model, url),
            lambda: make_usage_callback(store, provider, model, store_prompts=store_prompts),
            session, text, token_budget=budget,
        )
        self.worker.progress.connect(lambda msg: self.statusBar().showMessage(f"Assistant: {msg}"))
        self.worker.resultReady.connect(lambda result: self._on_agent_result(result, session))
        self.worker.finished.connect(self._on_agent_thread_finished)
        self.worker.finished.connect(self._on_worker_thread_finished)
        self._set_run_controls_enabled(False)
        self.btn_agent_ask.setEnabled(False)
        self.btn_agent_cancel.setVisible(True)
        self.statusBar().showMessage("Assistant: Thinking...")
        self.worker.start()

    def _agent_llm_factory(self, provider: str, model: str, url: Optional[str]):
        """Called on the worker thread (LangChain imports are slow). Tests replace this."""
        from companion.agent.llm import make_llm
        return lambda: make_llm(provider, model, base_url=url, timeout=CONFIG.AGENT_TIMEOUT_S)

    def _on_agent_cancel(self):
        if isinstance(self.worker, AgentWorker):
            self.worker.cancel()
            self.statusBar().showMessage("Assistant: cancelling after the current step...")

    def _on_agent_thread_finished(self):
        self._set_run_controls_enabled(True)
        self.btn_agent_ask.setEnabled(self.agent_enabled)
        self.btn_agent_cancel.setVisible(False)

    def _sync_model_combo(self):
        """The agent may have switched the engine model; mirror it without overwriting the saved choice."""
        idx = self.combo_model.findData(self.engine.model_name)
        if idx >= 0 and idx != self.combo_model.currentIndex():
            self.combo_model.blockSignals(True)
            self.combo_model.setCurrentIndex(idx)
            self.combo_model.blockSignals(False)

    def _on_agent_result(self, result, session=None):
        self._sync_model_combo()
        if session is not None and "set_prompt" in session.calls:
            self.txt_prompt.setText(session.prompt or "")
        if not result.ok:
            self.statusBar().showMessage(f"Assistant stopped: {result.reason}", 8000)
            QMessageBox.information(self, "Assistant", result.reason)
            return
        proposal = result.proposal
        if proposal is None or not proposal.variations:
            msg = result.text or "Nothing matching was found. Only VOC objects (person, car, bottle, dog...) can be located."
            self.statusBar().showMessage(f"Assistant: {msg}", 8000)
            QMessageBox.information(self, "Assistant", msg)
            return
        if self._layer_state_key() != self._agent_state_key:
            self.statusBar().showMessage("Assistant proposal dropped: the photo or layers changed meanwhile.", 6000)
            return
        self._agent_proposal = proposal
        self._agent_prev_display = self.canvas.get_current_image()
        self.canvas.set_preview_image(proposal.variations[0])
        self.btn_agent_apply.setVisible(True)
        self.btn_agent_discard.setVisible(True)
        note = f" {result.text}" if result.text else ""
        self.statusBar().showMessage(f"Assistant proposal (preview): {proposal.describe()}. Click Apply to commit or Discard.{note}")

    def _drop_agent_proposal(self):
        self._agent_proposal = None
        self._agent_prev_display = None
        self.btn_agent_apply.setVisible(False)
        self.btn_agent_discard.setVisible(False)

    def _on_agent_discard(self):
        prev = self._agent_prev_display
        if prev is not None and self._layer_state_key() == self._agent_state_key:
            self.canvas.set_display_image(prev)
        self._drop_agent_proposal()
        self.statusBar().showMessage("Assistant proposal discarded.", 3000)

    def _on_agent_apply(self):
        """Commit the proposal through the same layer path as Erase (_on_inpainting_finished)."""
        p = self._agent_proposal
        if p is None or self.worker is not None:
            return
        if self._layer_state_key() != self._agent_state_key:
            self._on_agent_discard()
            QMessageBox.information(self, "Assistant", "The photo or layers changed since the proposal; ask again.")
            return
        if self._agent_prev_display is not None:
            self.canvas.set_display_image(self._agent_prev_display)
        self._drop_agent_proposal()
        self.canvas.set_mask_image(p.mask)
        self.last_base_image = p.base_image.copy()
        self.last_used_mask = p.mask.copy()
        self._generation_start_time = time.time()
        self._on_inpainting_finished(p.variations)

    def _start_inpainting_pipeline(self, current_img: Image.Image, mask_img: Image.Image, seed: Optional[int] = None):
        """Launch worker thread for single-pass or multi-variation generation."""
        if self._model_switch_pending():  # covers Generate More, which skips _on_run_inpainting
            return
        self._set_run_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(2)
        self._generation_start_time = time.time()
        if hasattr(self, "timer_label"):
            self.timer_label.setText("")

        if self.engine.mode == EngineMode.FIREFLY:
            self._render_loading_carousel()
            self.carousel_panel.setVisible(True)

        self.statusBar().showMessage("Initializing LaMa inpainting pipeline...")

        self.last_base_image = current_img.copy()
        self.last_used_mask = mask_img.copy()

        prompt_text = (self.txt_prompt.text().strip() or None) if self._active_model_uses_prompt() else None
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
        # Clear before applying: the switch may open a modal prompt (re-entrancy)
        pending, self._pending_select = self._pending_select, None
        if pending is not None:
            self._on_live_photo_selected(pending)

    def _on_inpainting_progress(self, percentage: int, message: str):
        self.progress_bar.setValue(percentage)
        self.statusBar().showMessage(message)

    def _on_inpainting_finished(self, variations: List[Image.Image]):
        free_device_memory(self.engine.device)
        self.progress_bar.setValue(98)
        self.statusBar().showMessage("Applying inpainting composite to canvas...")

        if not variations:
            self.progress_bar.setVisible(False)
            self._set_run_controls_enabled(True)
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
            # In-place re-erase is not undoable; a stale redo must not reapply over it
            self._layer_redo.clear()
            self._update_history_buttons()
            self._layers_synced = False
            self._autosave_dirty = True
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
            self._push_layer_op("add", len(self.modification_layers) - 1, new_layer)
            self.active_layer_index = None

            # Apply to canvas directly reusing pix_0 without duplicate QPixmap conversion
            self.canvas.apply_inpainted_image(full_composite, cached_pixmap=pix_0)
            self.statusBar().showMessage(f"Added {new_layer.name} to Modifications stack.", 4000)
            self._autosave()

        # Layer history replaces the canvas image history; drop its full-res copies
        del self.canvas._image_history[:-1]

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
        self._set_run_controls_enabled(True)

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
            # Variation picks are not undoable; a stale redo must not reapply over them
            self._layer_redo.clear()
            self._update_history_buttons()
            self._autosave_dirty = True

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
        if self.worker is not None:
            return  # single worker slot (inpainting or assistant)
        if self.last_base_image and self.last_used_mask:
            import random
            fresh_seed = random.randint(1000, 999999)
            self._start_inpainting_pipeline(self.last_base_image, self.last_used_mask, seed=fresh_seed)
        else:
            self._on_run_inpainting()

    def _on_inpainting_error(self, err_msg: str):
        self.last_error = f"Inpainting error: {err_msg}"
        free_device_memory(self.engine.device)
        self.progress_bar.setVisible(False)
        self._set_run_controls_enabled(True)
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

        if self._confirming_switch:
            # Prompt already open: remember only the newest selection, never stack dialogs
            self._pending_select = payload
            return

        if self.worker is not None:
            logger.info(f"Inpainting active; delaying switch to {new_path}")
            self._pending_select = payload
            self.statusBar().showMessage("Lightroom selection changed; switching once inpainting finishes...", 4000)
            return

        self._apply_bridge_model(payload)

        # Re-selecting the loaded photo changes nothing; don't prompt or throw away its layers
        if self.modification_layers and self.input_path and os.path.abspath(new_path) == os.path.abspath(self.input_path):
            return

        confirmed = self._confirm_discard_layers()
        newer, self._pending_select = self._pending_select, None
        if not confirmed:
            return
        if newer is not None:
            # User agreed to switch; follow Lightroom's latest selection, not the stale one
            payload = newer
            new_path = payload.get("path")
            self._apply_bridge_model(payload)

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
        """
        Save the active variation (or layer composite) with its ICC profile and DPI.

        16-bit TIFF input saved as .tif/.tiff: untouched pixels keep full 16-bit precision;
        the erased area is computed in 8-bit and merged back (orig16 + (out8 - base8) * 257,
        clipped) and written as a 16-bit TIFF via a temp file + os.replace. Preview is 8-bit.
        Every other input or output format is saved through Pillow at 8 bits per channel.
        """
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
            if self.base_image16 is not None and self.output_path.lower().endswith((".tif", ".tiff")):
                merged = merge_into_16bit(self.base_image16, self.base_image, save_img)
                write_tiff16(self.output_path, merged, self.original_icc_profile, self.original_dpi)
                del merged
                logger.info(f"Saved 16-bit image to {self.output_path} (ICC Profile preserved: {bool(self.original_icc_profile)})")
                return self.output_path
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
            self.last_error = f"Could not save file: {e}"
            QMessageBox.critical(self, "Error Saving Image", f"Could not save file:\n{e}")
            return None

    def _on_sync_to_lightroom(self):
        """Save image and queue for Lightroom auto-import while keeping window open."""
        saved_path = self.save_image_to_disk()
        if saved_path:
            self.saved_successfully = True
            self._layers_synced = True
            self._autosave_dirty = False
            self._clear_recovery()
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

    def closeEvent(self, event):
        """Ensure all threads, live servers, and background processes terminate when window is closed."""
        logger.info("MainWindow closeEvent: shutting down background processes and cleaning temp files...")
        try:
            # Flush pending recovery changes; unsynced recovery data stays on disk after a clean close
            self._on_autosave_timer()

            # 1. Give an active worker thread a moment to finish (os._exit in main() kills it otherwise)
            if hasattr(self, "worker") and self.worker is not None:
                try:
                    if isinstance(self.worker, AgentWorker):
                        self.worker.cancel()
                    self.worker.wait(300)
                    # Keep the reference while running so Python never GCs a live QThread
                    if not self.worker.isRunning():
                        self.worker = None
                except Exception:
                    pass

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
    parser = argparse.ArgumentParser(description="Efface Magique LR - LaMa AI Eraser Companion")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input photo exported from Lightroom")
    parser.add_argument("--image", type=str, default=None, help="Alias for --input")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to write output photo")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Optional text; only adds a deterministic colour tint of a few levels (no prompt-driven generation)")
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
