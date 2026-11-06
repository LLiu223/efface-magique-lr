"""LaMa (simple-lama-inpainting, TorchScript) - the default inpainting model."""

import importlib.util
import logging
import sys
from typing import Callable, Optional

import torch
from PIL import Image

from companion.device import free_device_memory
from companion.models.base import InpaintModel
from companion.models.registry import register

logger = logging.getLogger("EffaceMagiqueModelEngine")


@register
class LamaModel(InpaintModel):
    name = "lama"
    label = "LaMa"
    uses_prompt = False

    def __init__(self):
        self._model = None
        self._device: Optional[torch.device] = None

    @classmethod
    def is_available(cls) -> bool:
        # find_spec only locates the package; it imports nothing and loads no weights
        return "simple_lama_inpainting" in sys.modules or importlib.util.find_spec("simple_lama_inpainting") is not None

    @classmethod
    def unavailable_reason(cls) -> str:
        return "simple-lama-inpainting is not installed (pip install -r requirements.txt)."

    def load(self, device: torch.device, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        from simple_lama_inpainting import SimpleLama  # lazy: keeps startup fast and lets tests mock it
        self._device = device
        self._model = SimpleLama(device=device)
        logger.info(f"Fast Simple-LaMa inpainting model loaded on {device}.")

        # Warmup pass on GPU to compile CUDA kernels ahead of user brush strokes
        if device.type == "cuda":
            try:
                if progress_callback:
                    progress_callback("Warming up GPU Tensor Cores & compiling kernels...")
                dummy_img = Image.new("RGB", (256, 256), (128, 128, 128))
                dummy_mask = Image.new("L", (256, 256), 255)
                with torch.inference_mode():
                    self._model(dummy_img, dummy_mask)
                logger.info("GPU inference pipeline pre-warmed successfully.")
            except Exception as e:
                logger.warning(f"GPU warmup non-fatal exception: {e}")
            finally:
                # Always free dummy tensors so they don't linger until next GC cycle
                free_device_memory(device)

    def unload(self) -> None:
        self._model = None

    def inpaint(self, crop: Image.Image, mask: Image.Image, seed: int = 0, prompt: Optional[str] = None) -> Image.Image:
        """
        Single unified high-resolution LaMa pass with LANCZOS resampling (no tile seams).
        seed and prompt are ignored: LaMa is deterministic and unconditioned.
        """
        crop_w, crop_h = crop.size
        is_gpu = self._device is not None and self._device.type == "cuda"
        max_dim = self.max_dim_gpu if is_gpu else self.max_dim_cpu

        scale = min(1.0, max_dim / max(crop_w, crop_h))

        if scale < 1.0:
            infer_w = int(crop_w * scale)
            infer_h = int(crop_h * scale)
            infer_img = crop.resize((infer_w, infer_h), Image.Resampling.LANCZOS)
            infer_mask = mask.resize((infer_w, infer_h), Image.Resampling.NEAREST)
        else:
            infer_img = crop
            infer_mask = mask

        with torch.inference_mode():
            raw_result = self._model(infer_img, infer_mask)

        if scale < 1.0 or raw_result.size != (crop_w, crop_h):
            raw_result = raw_result.resize((crop_w, crop_h), Image.Resampling.LANCZOS)

        return raw_result
