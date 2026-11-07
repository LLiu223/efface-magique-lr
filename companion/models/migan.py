"""MI-GAN (Picsart AI Research, ICCV 2023): small, fast, prompt-free inpainting via the official ONNX pipeline."""

import hashlib
import importlib.util
import logging
import os
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

from companion.config import CONFIG
from companion.models.base import InpaintModel
from companion.models.registry import register

logger = logging.getLogger("EffaceMagiqueModelEngine")

# Pinned in install.bat too: huggingface.co/andraniksargsyan/migan @ 406830d0fa60666da0071c342ad2fbc8f30c5c64
WEIGHT_FILE = "migan_pipeline_v2.onnx"
WEIGHT_SHA256 = "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@register
class MiganModel(InpaintModel):
    name = "migan"
    label = "MI-GAN"
    uses_prompt = False

    def __init__(self):
        self._session = None
        self._device: Optional[torch.device] = None

    @staticmethod
    def weight_path() -> str:
        return os.path.join(CONFIG.MODELS_DIR, WEIGHT_FILE)

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("onnxruntime") is not None and os.path.isfile(cls.weight_path())

    @classmethod
    def unavailable_reason(cls) -> str:
        if importlib.util.find_spec("onnxruntime") is None:
            return "MI-GAN needs the onnxruntime package (pip install -r requirements-extra.txt)."
        if not os.path.isfile(cls.weight_path()):
            return f"MI-GAN weights missing: {cls.weight_path()} (re-run install.bat and answer y to MI-GAN)."
        return ""

    def load(self, device: torch.device, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("MI-GAN needs the onnxruntime package (pip install -r requirements-extra.txt).") from e
        path = self.weight_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"MI-GAN weights missing: {path} (re-run install.bat).")
        if _sha256(path) != WEIGHT_SHA256:  # 28 MB: cheap enough to verify on every load
            raise ValueError(f"MI-GAN weights failed the SHA256 check: {path} (delete it and re-run install.bat).")
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self._session = ort.InferenceSession(path, providers=providers)
        self._device = device
        logger.info(f"MI-GAN ONNX pipeline loaded ({providers[0]}).")

    def unload(self) -> None:
        self._session = None

    def inpaint(self, crop: Image.Image, mask: Image.Image, seed: int = 0, prompt: Optional[str] = None) -> Image.Image:
        """The ONNX pipeline accepts any size (it crops around the hole and runs at 512 internally);
        the crop is only capped at max_dim to bound memory. seed and prompt are ignored."""
        crop_rgb = crop.convert("RGB")
        w, h = crop_rgb.size
        max_dim = self.max_dim_gpu if (self._device is not None and self._device.type == "cuda") else self.max_dim_cpu
        scale = min(1.0, max_dim / max(w, h))
        iw, ih = max(1, int(w * scale)), max(1, int(h * scale))
        img = crop_rgb.resize((iw, ih), Image.Resampling.LANCZOS) if scale < 1.0 else crop_rgb
        m = mask.convert("L")
        m = m.resize((iw, ih), Image.Resampling.NEAREST) if scale < 1.0 else m

        hole = np.asarray(m) > 10
        img_in = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)[np.newaxis]
        mask_in = np.where(hole, 0, 255).astype(np.uint8)[np.newaxis, np.newaxis]  # 255 = known, 0 = fill
        out = self._session.run(None, {"image": img_in, "mask": mask_in})[0][0].transpose(1, 2, 0)
        result = Image.fromarray(np.ascontiguousarray(out, dtype=np.uint8), "RGB")
        if result.size != (w, h):
            result = result.resize((w, h), Image.Resampling.LANCZOS)

        # Outside the mask the input crop comes back untouched (the pipeline blurs the seam edge)
        full_hole = (np.asarray(mask.convert("L")) > 10)[:, :, np.newaxis]
        return Image.fromarray(np.where(full_hole, np.asarray(result), np.asarray(crop_rgb)))
