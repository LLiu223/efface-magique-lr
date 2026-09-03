"""
canvas.py
Efface Magique LR - Interactive High-Performance Image Canvas

PyQt6 Canvas supporting:
- Smooth zoom & pan on 24MP-60MP images
- Red overlay mask painting with dynamic brush cursor
- Eraser mode & brush radius hotkeys ([ and ])
- Non-destructive Before/After comparison
- Full Undo / Redo history for brush strokes and inpainting steps
"""

from typing import Optional, List, Tuple
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QCursor,
    QWheelEvent,
    QMouseEvent,
    QKeyEvent,
    QPainterPath,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QWidget,
)
from PIL import Image

try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    HAS_OPENGL = True
except Exception:
    HAS_OPENGL = False


def pil_to_qimage(pil_img: Image.Image) -> QImage:
    """Convert PIL RGB Image to PyQt6 QImage (Format_RGB888)."""
    if pil_img.mode == "RGB":
        rgb_img = pil_img
    else:
        rgb_img = pil_img.convert("RGB")
    width, height = rgb_img.size
    bytes_per_line = 3 * width
    raw_bytes = rgb_img.tobytes("raw", "RGB")
    qimage = QImage(raw_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return qimage.copy()


def qimage_to_pil(qimage: QImage) -> Image.Image:
    """Convert PyQt6 QImage to PIL RGB Image."""
    qimage_rgba = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
    width = qimage_rgba.width()
    height = qimage_rgba.height()
    bytes_per_line = qimage_rgba.bytesPerLine()
    ptr = qimage_rgba.bits()
    ptr.setsize(qimage_rgba.sizeInBytes())
    pil_img = Image.frombuffer("RGBA", (width, height), ptr, "raw", "RGBA", bytes_per_line, 1)
    return pil_img.copy().convert("RGB")


class SplitBeforeItem(QGraphicsItem):
    """Zero-copy hardware-clipped QGraphicsItem displaying the Before image left of the split divider."""

    def __init__(self, orig_pixmap: QPixmap, parent: Optional[QGraphicsItem] = None):
        super().__init__(parent)
        self._pixmap: QPixmap = orig_pixmap
        self._w: float = float(orig_pixmap.width())
        self._h: float = float(orig_pixmap.height())
        self._split_x: float = self._w * 0.5
        self.setZValue(0.5)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._w, self._h)

    def set_split_x(self, x: float):
        new_x = max(0.0, min(self._w, float(x)))
        if abs(new_x - self._split_x) > 0.2:
            self._split_x = new_x
            self.update()

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._w = float(pixmap.width())
        self._h = float(pixmap.height())
        self.prepareGeometryChange()
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        if self._pixmap.isNull() or self._split_x <= 0:
            return
        painter.save()
        # Hardware clip directly to left half of split divider in scene coordinates
        painter.setClipRect(QRectF(0, 0, self._split_x, self._h))
        painter.drawPixmap(0, 0, self._pixmap)
        painter.restore()


class ImageCanvas(QGraphicsView):
    """
    Interactive Graphics View for drawing inpainting masks and viewing photos.
    Features:
    - 60 FPS hardware-accelerated canvas pan & zoom
    - Live dual concentric vector brush rings with soft feathering
    - Translucent on-canvas HUD overlay
    - Before / After Split-Screen slider view
    """

    maskChanged = pyqtSignal()
    brushSizeChanged = pyqtSignal(int)
    brushFeatherChanged = pyqtSignal(float)
    statusMessage = pyqtSignal(str)
    strokeFinished = pyqtSignal()
    splitCompareChanged = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Hardware-accelerated OpenGL viewport for 60 FPS rendering
        if HAS_OPENGL:
            try:
                self.setViewport(QOpenGLWidget())
            except Exception:
                pass

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Rendering options for smooth rendering & high performance
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)

        # Background color (Lightroom dark grey #181818)
        self.setBackgroundBrush(QBrush(QColor("#181818")))

        # Image state
        self._orig_pil: Optional[Image.Image] = None
        self._current_pil: Optional[Image.Image] = None
        self._orig_pixmap: QPixmap = QPixmap()
        self._curr_pixmap: QPixmap = QPixmap()
        self._image_item: Optional[QGraphicsPixmapItem] = None
        self._mask_item: Optional[QGraphicsPixmapItem] = None
        self._split_orig_item: Optional[SplitBeforeItem] = None

        # Mask QImage (stores alpha/red mask at full resolution)
        self._mask_qimage: Optional[QImage] = None

        # View and zoom state
        self._user_has_manually_zoomed: bool = False

        # Brush state
        self.brush_radius: int = 30  # radius in image pixels
        self.min_brush_radius: int = 3
        self.max_brush_radius: int = 300
        self.brush_feather: float = 0.25  # 0.0 = hard edge, 1.0 = maximum softness
        self.brush_opacity: float = 0.65
        self.is_eraser: bool = False
        self._is_drawing: bool = False
        self._last_draw_pos: Optional[QPointF] = None

        # Feather interactive drag adjustment (Shift + mouse drag)
        self._is_adjusting_feather: bool = False
        self._feather_drag_start_pos: QPoint = QPoint()
        self._feather_start_val: float = 0.25

        # Pan & Compare state
        self._is_panning: bool = False
        self._pan_start_pos: QPoint = QPoint()
        self._space_pressed: bool = False
        self._compare_mode: bool = False

        # Split-Screen Comparison Mode
        self._split_compare_mode: bool = False
        self._split_pos: float = 0.5  # 0.0 to 1.0
        self._is_dragging_split: bool = False

        # On-Canvas HUD Overlay Badge
        self._hud_text: str = ""
        self._hud_opacity: float = 0.0
        self._hud_fade_step: int = 0
        self._hud_timer = QTimer(self)
        self._hud_timer.timeout.connect(self._on_hud_timer_tick)

        # Undo / Redo history
        self._image_history: List[Image.Image] = []
        self._image_redo_stack: List[Image.Image] = []
        self._mask_history: List[QImage] = []
        self._mask_redo_stack: List[QImage] = []

        # Vector brush overlay rings (outer boundary + inner feather core)
        self._brush_ring_item: Optional[QGraphicsEllipseItem] = None
        self._brush_inner_ring_item: Optional[QGraphicsEllipseItem] = None
        self._init_brush_ring()

        self._update_cursor()

    # -------------------------------------------------------------------------
    # Image Loading and State Management
    # -------------------------------------------------------------------------

    def _init_brush_ring(self):
        """Create high-precision vector brush cursor overlay in scene coordinates."""
        if self._brush_ring_item is not None:
            try:
                self.scene.removeItem(self._brush_ring_item)
            except Exception:
                pass
        if self._brush_inner_ring_item is not None:
            try:
                self.scene.removeItem(self._brush_inner_ring_item)
            except Exception:
                pass

        # Outer boundary ring
        self._brush_ring_item = QGraphicsEllipseItem()
        ring_pen = QPen(QColor(255, 60, 60, 240), 1.5)
        ring_pen.setCosmetic(True)
        self._brush_ring_item.setPen(ring_pen)
        self._brush_ring_item.setBrush(QBrush(Qt.GlobalColor.transparent))
        self._brush_ring_item.setZValue(9999)
        self.scene.addItem(self._brush_ring_item)
        self._brush_ring_item.setVisible(False)

        # Inner core ring (shows solid core boundary when feathering is active)
        self._brush_inner_ring_item = QGraphicsEllipseItem()
        inner_pen = QPen(QColor(255, 140, 140, 190), 1.0, Qt.PenStyle.DashLine)
        inner_pen.setCosmetic(True)
        self._brush_inner_ring_item.setPen(inner_pen)
        self._brush_inner_ring_item.setBrush(QBrush(Qt.GlobalColor.transparent))
        self._brush_inner_ring_item.setZValue(9998)
        self.scene.addItem(self._brush_inner_ring_item)
        self._brush_inner_ring_item.setVisible(False)

    def load_image(self, pil_img: Image.Image):
        """Set a new base image and initialize scene and masks with minimal memory overhead."""
        if pil_img.mode == "RGB":
            rgb_img = pil_img.copy()
        else:
            rgb_img = pil_img.convert("RGB")

        self._orig_pil = rgb_img
        self._current_pil = rgb_img.copy()

        # Clear histories
        self._image_history = [self._current_pil]
        self._image_redo_stack.clear()
        self._mask_history.clear()
        self._mask_redo_stack.clear()

        # Build QImage mask with same dimensions
        w, h = pil_img.size
        self._mask_qimage = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        self._mask_qimage.fill(QColor(0, 0, 0, 0))
        self._save_mask_state()

        # Update scene
        self.scene.clear()

        base_qimg = pil_to_qimage(self._current_pil)
        self._orig_pixmap = QPixmap.fromImage(base_qimg)
        self._curr_pixmap = self._orig_pixmap
        self._image_item = self.scene.addPixmap(self._curr_pixmap)
        self._image_item.setZValue(0)

        mask_pixmap = QPixmap.fromImage(self._mask_qimage)
        self._mask_item = self.scene.addPixmap(mask_pixmap)
        self._mask_item.setZValue(1)

        self._split_orig_item = None
        self._brush_ring_item = None
        self._brush_inner_ring_item = None
        self._init_brush_ring()

        self.scene.setSceneRect(0, 0, w, h)
        self.fit_to_screen()
        self.statusMessage.emit(f"Loaded image {w}x{h} px")

    def fit_to_screen(self):
        """Scale and center the image to comfortably fit the current canvas viewport."""
        if self._current_pil is None or self.scene.sceneRect().isEmpty():
            return
        if self.viewport().width() > 10 and self.viewport().height() > 10:
            self.resetTransform()
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._user_has_manually_zoomed = False
            self._update_cursor()
            self._show_hud("Fit to Screen (Ctrl+0)")

    def zoom_to_actual_size(self):
        """Scale to 100% 1:1 pixel view (one image pixel = one screen pixel)."""
        if self._current_pil is None or self.scene.sceneRect().isEmpty():
            return
        self.resetTransform()
        self._user_has_manually_zoomed = True
        self._update_cursor()
        self.statusMessage.emit("Zoom: 100% (1:1 Pixel View)")
        self._show_hud("Zoom: 100% 1:1 (Ctrl+1)")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Automatically fit image to view when window resizes unless user actively zoomed in
        if not self._user_has_manually_zoomed and self._current_pil is not None:
            self.fit_to_screen()

    def get_current_image(self) -> Optional[Image.Image]:
        """Return the current active PIL image."""
        return self._current_pil

    def set_preview_image(self, pil_img: Image.Image):
        """Update the displayed image without altering history (for Firefly variations)."""
        self._current_pil = pil_img.copy().convert("RGB")
        base_qimg = pil_to_qimage(self._current_pil)
        self._curr_pixmap = QPixmap.fromImage(base_qimg)
        if self._image_item:
            self._image_item.setPixmap(self._curr_pixmap)
        if self._split_compare_mode:
            self._update_split_display()

    def get_original_image(self) -> Optional[Image.Image]:
        """Return the original unmodified PIL image."""
        return self._orig_pil

    def get_mask_image(self) -> Optional[Image.Image]:
        """Extract the drawn mask as a binary grayscale PIL Image (0 = background, 255 = masked)."""
        if self._mask_qimage is None:
            return None
        alpha_qimage = self._mask_qimage.convertToFormat(QImage.Format.Format_Alpha8)
        w = alpha_qimage.width()
        h = alpha_qimage.height()
        bytes_per_line = alpha_qimage.bytesPerLine()
        ptr = alpha_qimage.bits()
        ptr.setsize(alpha_qimage.sizeInBytes())
        raw_mask = Image.frombuffer("L", (w, h), ptr, "raw", "L", bytes_per_line, 1)
        mask_np = (np.asarray(raw_mask) > 10).astype(np.uint8) * 255
        return Image.fromarray(mask_np, mode="L")

    def has_mask(self) -> bool:
        """Check if any area has been painted on the mask."""
        if self._mask_qimage is None:
            return False
        mask = self.get_mask_image()
        if mask is None:
            return False
        return bool(np.any(np.asarray(mask) > 10))

    def apply_inpainted_image(self, new_pil_img: Image.Image):
        """Update the canvas with new inpainted image, reset mask, and save to undo history."""
        self._current_pil = new_pil_img.copy().convert("RGB")
        self._image_history.append(self._current_pil.copy())
        self._image_redo_stack.clear()

        # Update base image pixmap
        base_qimg = pil_to_qimage(self._current_pil)
        self._curr_pixmap = QPixmap.fromImage(base_qimg)
        if self._image_item:
            self._image_item.setPixmap(self._curr_pixmap)
        if self._split_compare_mode:
            self._update_split_display()

        # Clear mask and reset mask history so Ctrl+Z undoes the AI image, not the consumed brush strokes
        self.clear_mask(save_state=False)
        self._mask_history = [self._mask_qimage.copy()]
        self._mask_redo_stack.clear()

    # -------------------------------------------------------------------------
    # Undo / Redo Actions
    # -------------------------------------------------------------------------

    def _save_mask_state(self):
        """Push current mask snapshot onto mask history stack."""
        if self._mask_qimage is not None:
            self._mask_history.append(self._mask_qimage.copy())
            if len(self._mask_history) > 30:
                self._mask_history.pop(0)
            self._mask_redo_stack.clear()

    def undo(self) -> bool:
        """Undo last active mask stroke or last AI inpainting step."""
        # 1. If user has active drawn brush strokes that haven't been inpainted yet, undo the stroke
        if self.has_mask() and len(self._mask_history) > 1:
            self._mask_redo_stack.append(self._mask_history.pop())
            previous_mask = self._mask_history[-1].copy()
            self._mask_qimage = previous_mask
            if self._mask_item:
                self._mask_item.setPixmap(QPixmap.fromImage(self._mask_qimage))
            self.maskChanged.emit()
            self.statusMessage.emit("Undid brush stroke.")
            return True

        # 2. Otherwise, undo AI inpainting step directly and NEVER show old brush strokes
        if len(self._image_history) > 1:
            self._image_redo_stack.append(self._image_history.pop())
            self._current_pil = self._image_history[-1].copy()
            base_qimg = pil_to_qimage(self._current_pil)
            self._curr_pixmap = QPixmap.fromImage(base_qimg)
            if self._image_item:
                self._image_item.setPixmap(self._curr_pixmap)
            if self._split_compare_mode:
                self._update_split_display()
            # Keep mask completely clean - do not show old brush strokes
            self.clear_mask(save_state=False)
            self._mask_history = [self._mask_qimage.copy()]
            self._mask_redo_stack.clear()
            self.statusMessage.emit("Undid AI inpainting step.")
            return True

        self.statusMessage.emit("Nothing to undo.")
        return False

    def redo(self) -> bool:
        """Redo next inpainting step or brush stroke."""
        # 1. If there is an inpainting step on the redo stack, redo the AI inpainting
        if self._image_redo_stack:
            next_img = self._image_redo_stack.pop()
            self._image_history.append(next_img.copy())
            self._current_pil = next_img.copy()
            base_qimg = pil_to_qimage(self._current_pil)
            self._curr_pixmap = QPixmap.fromImage(base_qimg)
            if self._image_item:
                self._image_item.setPixmap(self._curr_pixmap)
            if self._split_compare_mode:
                self._update_split_display()
            self.clear_mask(save_state=False)
            self._mask_history = [self._mask_qimage.copy()]
            self._mask_redo_stack.clear()
            self.statusMessage.emit("Redid AI inpainting step.")
            return True

        # 2. Otherwise redo brush strokes if available
        if self._mask_redo_stack:
            next_mask = self._mask_redo_stack.pop()
            self._mask_history.append(next_mask.copy())
            self._mask_qimage = next_mask.copy()
            if self._mask_item:
                self._mask_item.setPixmap(QPixmap.fromImage(self._mask_qimage))
            self.maskChanged.emit()
            self.statusMessage.emit("Redid brush stroke.")
            return True

        self.statusMessage.emit("Nothing to redo.")
        return False

    def reset_all(self):
        """Reset canvas to original unedited photo and clear all history."""
        if self._orig_pil is not None:
            self._current_pil = self._orig_pil.copy()
            self._image_history = [self._current_pil.copy()]
            self._image_redo_stack.clear()
            self._curr_pixmap = self._orig_pixmap
            if self._image_item and not self._curr_pixmap.isNull():
                self._image_item.setPixmap(self._curr_pixmap)
            if self._split_orig_item:
                self._split_orig_item.setVisible(False)
            if self._split_compare_mode:
                self.set_split_compare_mode(False)
            self.clear_mask(save_state=False)
            self._mask_history = [self._mask_qimage.copy()]
            self._mask_redo_stack.clear()
            self.statusMessage.emit("Reset to original photo.")

    def clear_mask(self, save_state: bool = True):
        """Clear the current mask without altering the image."""
        if self._mask_qimage is not None:
            self._mask_qimage.fill(QColor(0, 0, 0, 0))
            if self._mask_item:
                self._mask_item.setPixmap(QPixmap.fromImage(self._mask_qimage))
            if save_state:
                self._save_mask_state()
            self.maskChanged.emit()
            self.statusMessage.emit("Mask cleared.")

    # -------------------------------------------------------------------------
    # Comparison Modes (Before / After and Split-Screen Slider)
    # -------------------------------------------------------------------------

    def set_compare_mode(self, enabled: bool):
        """Toggle Before/After comparison view."""
        self._compare_mode = enabled
        if self._image_item is None:
            return

        target_pixmap = self._orig_pixmap if enabled else self._curr_pixmap
        if not target_pixmap.isNull():
            self._image_item.setPixmap(target_pixmap)

        # Hide mask and brush rings during compare
        visible = not enabled
        if self._mask_item:
            self._mask_item.setVisible(visible)
        if self._brush_ring_item:
            self._brush_ring_item.setVisible(visible)
        if self._brush_inner_ring_item:
            self._brush_inner_ring_item.setVisible(visible)

        label = "Showing Original (Before)" if enabled else "Showing Edited (After)"
        self.statusMessage.emit(label)
        self.viewport().update()

    def set_split_compare_mode(self, enabled: bool):
        """Toggle Before / After Split-Screen slider view with zero-copy hardware blitting."""
        self._split_compare_mode = enabled
        if self._split_compare_mode:
            # Ensure SplitBeforeItem overlay is created on top of base image
            if self._split_orig_item is None and not self._orig_pixmap.isNull():
                self._split_orig_item = SplitBeforeItem(self._orig_pixmap)
                self.scene.addItem(self._split_orig_item)
            elif self._split_orig_item is not None and not self._orig_pixmap.isNull():
                self._split_orig_item.set_pixmap(self._orig_pixmap)
                self._split_orig_item.setVisible(True)

            self._update_split_display()
            self._show_hud("Split-Screen View: Active")
            self.statusMessage.emit("Split-Screen: Drag vertical divider to compare Before & After.")
        else:
            if self._split_orig_item is not None:
                self._split_orig_item.setVisible(False)
            if self._image_item and not self._curr_pixmap.isNull():
                self._image_item.setPixmap(self._curr_pixmap)
            self.statusMessage.emit("Split-Screen: Deactivated.")

        # Hide mask and brush rings during split comparison
        visible = not enabled
        if self._mask_item:
            self._mask_item.setVisible(visible)
        if self._brush_ring_item:
            self._brush_ring_item.setVisible(visible)
        if self._brush_inner_ring_item:
            self._brush_inner_ring_item.setVisible(visible)

        self.splitCompareChanged.emit(enabled)
        self.viewport().update()

    def _update_split_display(self):
        """Update split divider position with zero CPU copies (120+ FPS hardware speed)."""
        if not self._split_compare_mode or self._orig_pixmap.isNull() or self.scene is None:
            return

        w = self.scene.sceneRect().width()
        split_scene_x = w * self._split_pos

        if self._split_orig_item is not None:
            self._split_orig_item.set_split_x(split_scene_x)
            self._split_orig_item.setVisible(True)

        if self._image_item and not self._curr_pixmap.isNull():
            self._image_item.setPixmap(self._curr_pixmap)

        self.viewport().update()

    # -------------------------------------------------------------------------
    # Brush, Feather & Tool Controls
    # -------------------------------------------------------------------------

    def set_brush_radius(self, radius: int):
        self.brush_radius = max(self.min_brush_radius, min(radius, self.max_brush_radius))
        self._update_brush_ring()
        self._show_hud(f"Brush: {self.brush_radius}px  |  Feather: {int(self.brush_feather * 100)}%")
        self.brushSizeChanged.emit(self.brush_radius)

    def set_brush_feather(self, feather: float):
        self.brush_feather = max(0.0, min(1.0, float(feather)))
        self._update_brush_ring()
        self._show_hud(f"Brush: {self.brush_radius}px  |  Feather: {int(self.brush_feather * 100)}%")
        self.brushFeatherChanged.emit(self.brush_feather)

    def set_eraser_mode(self, enabled: bool):
        self.is_eraser = enabled
        self._update_brush_ring()
        mode_str = "Eraser" if enabled else "Mask Brush"
        self.statusMessage.emit(f"Active Tool: {mode_str}")

    def _update_brush_ring(self, scene_pos: Optional[QPointF] = None):
        """Update position, radius, and appearance of dual concentric vector brush rings."""
        if self._brush_ring_item is None or self.scene is None:
            return
        if self._compare_mode or self._split_compare_mode or self._is_panning or self._space_pressed or self._current_pil is None:
            self._brush_ring_item.setVisible(False)
            if self._brush_inner_ring_item:
                self._brush_inner_ring_item.setVisible(False)
            return

        if scene_pos is None:
            view_pt = self.mapFromGlobal(QCursor.pos())
            scene_pos = self.mapToScene(view_pt)

        rect = self.scene.sceneRect()
        if not rect.contains(scene_pos) and not self._is_drawing:
            self._brush_ring_item.setVisible(False)
            if self._brush_inner_ring_item:
                self._brush_inner_ring_item.setVisible(False)
            return

        r = float(self.brush_radius)
        self._brush_ring_item.setRect(QRectF(scene_pos.x() - r, scene_pos.y() - r, r * 2.0, r * 2.0))

        if self.is_eraser:
            pen = QPen(QColor(255, 255, 255, 240), 1.5)
        else:
            pen = QPen(QColor(255, 60, 60, 240), 1.5)
        pen.setCosmetic(True)
        self._brush_ring_item.setPen(pen)
        self._brush_ring_item.setVisible(True)

        # Inner core ring (shows solid core boundary when feather > 0)
        if self._brush_inner_ring_item is not None:
            if self.brush_feather > 0.05:
                r_inner = max(1.0, r * (1.0 - self.brush_feather))
                self._brush_inner_ring_item.setRect(QRectF(scene_pos.x() - r_inner, scene_pos.y() - r_inner, r_inner * 2.0, r_inner * 2.0))
                inner_pen = QPen(QColor(255, 160, 160, 180), 1.0, Qt.PenStyle.DashLine)
                inner_pen.setCosmetic(True)
                self._brush_inner_ring_item.setPen(inner_pen)
                self._brush_inner_ring_item.setVisible(True)
            else:
                self._brush_inner_ring_item.setVisible(False)

    def _update_cursor(self):
        """Configure viewport cursor mode and update vector rings."""
        if self._compare_mode:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if self._brush_ring_item:
                self._brush_ring_item.setVisible(False)
            if self._brush_inner_ring_item:
                self._brush_inner_ring_item.setVisible(False)
            return

        if self._is_panning:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            if self._brush_ring_item:
                self._brush_ring_item.setVisible(False)
            if self._brush_inner_ring_item:
                self._brush_inner_ring_item.setVisible(False)
            return

        if self._space_pressed:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            if self._brush_ring_item:
                self._brush_ring_item.setVisible(False)
            if self._brush_inner_ring_item:
                self._brush_inner_ring_item.setVisible(False)
            return

        if self._split_compare_mode:
            self.setCursor(Qt.CursorShape.SplitHCursor)
            if self._brush_ring_item:
                self._brush_ring_item.setVisible(False)
            if self._brush_inner_ring_item:
                self._brush_inner_ring_item.setVisible(False)
            return

        # Accurate crosshair at mouse tip + vector ring overlay on canvas
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._update_brush_ring()

    def enterEvent(self, event):
        super().enterEvent(event)
        self._update_brush_ring()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if self._brush_ring_item:
            self._brush_ring_item.setVisible(False)
        if self._brush_inner_ring_item:
            self._brush_inner_ring_item.setVisible(False)

    # -------------------------------------------------------------------------
    # On-Canvas HUD Overlay & Foreground Rendering
    # -------------------------------------------------------------------------

    def _show_hud(self, text: str):
        """Trigger floating translucent HUD badge on canvas viewport."""
        self._hud_text = text
        self._hud_opacity = 1.0
        self._hud_fade_step = 0
        self._hud_timer.stop()
        self._hud_timer.start(25)  # 25ms tick for smooth fade
        self.viewport().update()

    def _on_hud_timer_tick(self):
        self._hud_fade_step += 1
        # Hold full opacity for ~750ms (30 ticks), then smoothly fade over ~375ms (15 ticks)
        if self._hud_fade_step > 30:
            self._hud_opacity = max(0.0, 1.0 - (self._hud_fade_step - 30) / 15.0)
            if self._hud_opacity <= 0.0:
                self._hud_timer.stop()
        self.viewport().update()

    def drawForeground(self, painter: QPainter, rect: QRectF):
        """Draw viewport-level HUD overlays: Split divider, pills, and brush badge."""
        super().drawForeground(painter, rect)

        # 1. Render Split-Screen Curtain Divider & Pills
        if self._split_compare_mode and self.scene is not None and not self.scene.sceneRect().isEmpty():
            painter.save()
            painter.resetTransform()

            split_scene_x = self.scene.sceneRect().width() * self._split_pos
            pt_top = self.mapFromScene(QPointF(split_scene_x, 0))
            vx = int(pt_top.x())
            vh = self.viewport().height()

            # Vertical divider line
            pen = QPen(QColor(0, 120, 212, 230), 2.0)
            painter.setPen(pen)
            painter.drawLine(vx, 0, vx, vh)

            # Center grab handle
            cy = vh // 2
            painter.setBrush(QBrush(QColor(0, 120, 212, 240)))
            painter.setPen(QPen(QColor(255, 255, 255, 255), 1.5))
            painter.drawEllipse(QPoint(vx, cy), 14, 14)

            # Left/right grab arrows
            arrow_pen = QPen(QColor(255, 255, 255, 255), 2.0)
            painter.setPen(arrow_pen)
            painter.drawLine(vx - 5, cy, vx - 2, cy - 3)
            painter.drawLine(vx - 5, cy, vx - 2, cy + 3)
            painter.drawLine(vx + 5, cy, vx + 2, cy - 3)
            painter.drawLine(vx + 5, cy, vx + 2, cy + 3)

            # "BEFORE" badge (Left)
            painter.setBrush(QBrush(QColor(28, 28, 28, 220)))
            painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
            painter.drawRoundedRect(vx - 90, 24, 76, 26, 6, 6)
            painter.setPen(QColor(230, 230, 230))
            f = painter.font()
            f.setPointSize(9)
            f.setBold(True)
            painter.setFont(f)
            painter.drawText(QRect(vx - 90, 24, 76, 26), Qt.AlignmentFlag.AlignCenter, "BEFORE")

            # "AFTER" badge (Right)
            painter.setBrush(QBrush(QColor(0, 120, 212, 230)))
            painter.setPen(QPen(QColor(255, 255, 255, 90), 1))
            painter.drawRoundedRect(vx + 14, 24, 76, 26, 6, 6)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QRect(vx + 14, 24, 76, 26), Qt.AlignmentFlag.AlignCenter, "AFTER")

            painter.restore()

        # 2. Render On-Canvas Translucent HUD Overlay Badge
        if self._hud_opacity > 0.0:
            painter.save()
            painter.resetTransform()

            vw = self.viewport().width()
            vh = self.viewport().height()

            badge_w = 260
            badge_h = 36
            badge_x = (vw - badge_w) // 2
            badge_y = vh - 60

            bg_col = QColor(24, 24, 24, int(225 * self._hud_opacity))
            border_col = QColor(0, 120, 212, int(230 * self._hud_opacity))
            text_col = QColor(255, 255, 255, int(255 * self._hud_opacity))

            painter.setBrush(QBrush(bg_col))
            painter.setPen(QPen(border_col, 1.5))
            painter.drawRoundedRect(badge_x, badge_y, badge_w, badge_h, 8, 8)

            f = painter.font()
            f.setPointSize(10)
            f.setBold(True)
            painter.setFont(f)
            painter.setPen(text_col)
            painter.drawText(QRect(badge_x, badge_y, badge_w, badge_h), Qt.AlignmentFlag.AlignCenter, self._hud_text)

            painter.restore()

    def _paint_stroke(self, p1: QPointF, p2: QPointF):
        """Draw or erase a stroke on the mask QImage between two points."""
        if self._mask_qimage is None:
            return

        painter = QPainter(self._mask_qimage)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pt1 = p1.toPoint()
        pt2 = p2.toPoint()

        if self.is_eraser:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            pen = QPen(QColor(0, 0, 0, 255), self.brush_radius * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 255)))
            painter.drawLine(pt1, pt2)
            painter.drawEllipse(pt1, self.brush_radius, self.brush_radius)
            painter.drawEllipse(pt2, self.brush_radius, self.brush_radius)
        else:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            mask_color = QColor(255, 30, 30, int(self.brush_opacity * 255))

            if self.brush_feather > 0.05:
                # Radial gradient stamped along stroke for smooth feathered falloff
                grad = QRadialGradient(QPointF(pt2), float(self.brush_radius))
                stop_solid = max(0.0, min(0.95, 1.0 - self.brush_feather))
                grad.setColorAt(0.0, mask_color)
                grad.setColorAt(stop_solid, mask_color)
                grad.setColorAt(1.0, QColor(255, 30, 30, 0))

                pen = QPen(mask_color, self.brush_radius * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(pt1, pt2)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(grad))
                painter.drawEllipse(pt2, self.brush_radius, self.brush_radius)
            else:
                pen = QPen(mask_color, self.brush_radius * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(QBrush(mask_color))
                painter.drawLine(pt1, pt2)
                painter.drawEllipse(pt1, self.brush_radius, self.brush_radius)
                painter.drawEllipse(pt2, self.brush_radius, self.brush_radius)

        painter.end()

        # Update display pixmap
        if self._mask_item:
            self._mask_item.setPixmap(QPixmap.fromImage(self._mask_qimage))

    # -------------------------------------------------------------------------
    # Mouse & Keyboard Event Handlers
    # -------------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent):
        """Smooth mouse-wheel zoom centered on cursor."""
        self._user_has_manually_zoomed = True
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)
        self._update_cursor()
        self.viewport().update()

    def mousePressEvent(self, event: QMouseEvent):
        # 1. Split-Screen slider drag initiation
        if self._split_compare_mode and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            split_scene_x = self.scene.sceneRect().width() * self._split_pos
            pt_top = self.mapFromScene(QPointF(split_scene_x, 0))
            if abs(event.pos().x() - pt_top.x()) < 30 or abs(scene_pos.x() - split_scene_x) < 30:
                self._is_dragging_split = True
                self.setCursor(Qt.CursorShape.SplitHCursor)
                event.accept()
                return

        # 2. Middle click or Spacebar+Left click starts hardware pan
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and self._space_pressed):
            self._is_panning = True
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # 3. Shift + Left Click starts interactive feather adjustment
        if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) and event.button() == Qt.MouseButton.LeftButton:
            self._is_adjusting_feather = True
            self._feather_drag_start_pos = event.pos()
            self._feather_start_val = self.brush_feather
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            event.accept()
            return

        # 4. Left click starts mask stroke
        if event.button() == Qt.MouseButton.LeftButton and not self._compare_mode and not self._split_compare_mode:
            scene_pos = self.mapToScene(event.pos())
            if self.scene.sceneRect().contains(scene_pos):
                self._is_drawing = True
                self._last_draw_pos = scene_pos
                self._paint_stroke(scene_pos, scene_pos)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        # Handle split slider dragging
        if self._is_dragging_split:
            scene_pos = self.mapToScene(event.pos())
            w = self.scene.sceneRect().width()
            if w > 0:
                self._split_pos = max(0.01, min(0.99, scene_pos.x() / w))
                self._update_split_display()
            event.accept()
            return

        # Show split cursor when hovering over divider in split compare mode
        if self._split_compare_mode and not self._is_panning and not self._is_adjusting_feather:
            split_scene_x = self.scene.sceneRect().width() * self._split_pos
            pt_top = self.mapFromScene(QPointF(split_scene_x, 0))
            if abs(event.pos().x() - pt_top.x()) < 25:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self._update_cursor()

        # Handle active panning
        if self._is_panning:
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        # Handle interactive feather adjustment
        if self._is_adjusting_feather:
            delta_y = self._feather_drag_start_pos.y() - event.pos().y()
            delta_feather = delta_y / 180.0
            new_feather = max(0.0, min(1.0, self._feather_start_val + delta_feather))
            self.set_brush_feather(new_feather)
            event.accept()
            return

        scene_pos = self.mapToScene(event.pos())

        # Handle active stroke painting
        if self._is_drawing and self._last_draw_pos is not None:
            self._paint_stroke(self._last_draw_pos, scene_pos)
            self._last_draw_pos = scene_pos
            self._update_brush_ring(scene_pos)
            event.accept()
            return

        self._update_brush_ring(scene_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._is_dragging_split and event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging_split = False
            self._update_cursor()
            event.accept()
            return

        if self._is_adjusting_feather and event.button() == Qt.MouseButton.LeftButton:
            self._is_adjusting_feather = False
            self._update_cursor()
            event.accept()
            return

        if event.button() == Qt.MouseButton.MiddleButton or (self._is_panning and event.button() == Qt.MouseButton.LeftButton):
            self._is_panning = False
            self._update_cursor()
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self._is_drawing:
            self._is_drawing = False
            self._last_draw_pos = None
            self._save_mask_state()
            self.maskChanged.emit()
            self.strokeFinished.emit()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()

        # Spacebar or Backslash (\) hold for instant Before/After comparison
        if (key == Qt.Key.Key_Space or key == Qt.Key.Key_Backslash) and not event.isAutoRepeat():
            self._space_pressed = True
            if not self._is_panning:
                self.set_compare_mode(True)
                if key == Qt.Key.Key_Space:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return

        # Hotkeys [ and ] for brush radius
        if key == Qt.Key.Key_BracketLeft:
            self.set_brush_radius(self.brush_radius - 5)
            event.accept()
            return
        elif key == Qt.Key.Key_BracketRight:
            self.set_brush_radius(self.brush_radius + 5)
            event.accept()
            return

        # Quick zoom shortcuts
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_0:
                self.fit_to_screen()
                event.accept()
                return
            elif key == Qt.Key.Key_1:
                self.zoom_to_actual_size()
                event.accept()
                return
            elif key == Qt.Key.Key_Z:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.redo()
                else:
                    self.undo()
                event.accept()
                return
            elif key == Qt.Key.Key_Y:
                self.redo()
                event.accept()
                return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        key = event.key()
        if (key == Qt.Key.Key_Space or key == Qt.Key.Key_Backslash) and not event.isAutoRepeat():
            self._space_pressed = False
            self.set_compare_mode(False)
            self._update_cursor()
            event.accept()
            return

        super().keyReleaseEvent(event)

