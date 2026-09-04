"""
layers.py
Efface Magique LR - Non-Destructive Modification Layer System
Models individual inpainting and erase operations as independent, re-editable layers.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image

from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QBrush, QPen
from PyQt6.QtCore import Qt

from companion.inpainting_engine import EngineMode


@dataclass
class ModificationLayer:
    """Represents an individual erase or inpainting modification in the non-destructive stack."""
    layer_id: str
    name: str
    mask: Image.Image                                    # Grayscale 'L' mode mask (0 = untouched, 255 = modified)
    inpainted_image: Image.Image                         # Full-resolution result of this layer
    variations: List[Image.Image] = field(default_factory=list) # Candidate variations if Firefly mode
    active_variation_index: int = 0
    prompt: Optional[str] = None
    engine_mode: EngineMode = EngineMode.FAST
    seed: Optional[int] = None
    detect_subject: bool = False
    enable_grain: bool = False
    visible: bool = True
    created_at: float = field(default_factory=time.time)
    thumbnail: Optional[QPixmap] = None
    cached_pixmap: Optional[QPixmap] = None
    cached_mask_qimage: Optional[QImage] = None
    composite_cache: Optional[Image.Image] = None

    def get_active_image(self) -> Image.Image:
        """Return the active variation image or primary inpainted image."""
        if self.variations and 0 <= self.active_variation_index < len(self.variations):
            return self.variations[self.active_variation_index]
        return self.inpainted_image

    def update_thumbnail(self, base_image: Image.Image):
        """Update the preview thumbnail for this layer."""
        display_img = self.composite_cache if self.composite_cache is not None else self.get_active_image()
        self.thumbnail = create_layer_thumbnail(base_image, self.mask, display_img)


def create_layer_thumbnail(
    base_image: Image.Image,
    mask: Image.Image,
    result_image: Optional[Image.Image] = None,
    target_size: Tuple[int, int] = (56, 42),
) -> QPixmap:
    """
    Generate an informative, high-contrast thumbnail card for a modification layer.
    Shows the affected region on the photo with a subtle red overlay highlighting the erased area.
    """
    display_img = result_image if result_image is not None else base_image
    w, h = display_img.size
    tw, th = target_size

    scale = min(tw / w, th / h)
    real_tw = max(1, int(w * scale))
    real_th = max(1, int(h * scale))

    # Resize directly without making expensive full-resolution image copies
    thumb_pil = display_img.resize((real_tw, real_th), Image.Resampling.BILINEAR)
    if thumb_pil.mode != "RGB":
        thumb_pil = thumb_pil.convert("RGB")

    mask_thumb = mask.resize((real_tw, real_th), Image.Resampling.BILINEAR)
    if mask_thumb.mode != "L":
        mask_thumb = mask_thumb.convert("L")
    mask_np = np.asarray(mask_thumb)

    # Base QImage
    thumb_rgb = np.asarray(thumb_pil).copy()
    # Overlay bright red on masked region in thumbnail for scannability
    mask_zone = mask_np > 30
    thumb_rgb[mask_zone, 0] = np.clip(thumb_rgb[mask_zone, 0].astype(int) + 100, 0, 255).astype(np.uint8)
    thumb_rgb[mask_zone, 1] = (thumb_rgb[mask_zone, 1] * 0.4).astype(np.uint8)
    thumb_rgb[mask_zone, 2] = (thumb_rgb[mask_zone, 2] * 0.4).astype(np.uint8)

    qimg = QImage(thumb_rgb.data, real_tw, real_th, real_tw * 3, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg.copy())

    # Draw a rounded border on a fixed-size canvas
    final_pixmap = QPixmap(tw, th)
    final_pixmap.fill(QColor(24, 24, 24))
    painter = QPainter(final_pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Center image
    ox = (tw - real_tw) // 2
    oy = (th - real_th) // 2
    painter.drawPixmap(ox, oy, pix)

    # Subtle outline border
    painter.setPen(QPen(QColor(60, 60, 60), 1))
    painter.setBrush(Qt.GlobalColor.transparent)
    painter.drawRect(0, 0, tw - 1, th - 1)
    painter.end()

    return final_pixmap
