"""Stable Diffusion 1.5 inpainting (diffusers): prompt-driven, slow, optional install."""

import importlib.util
import logging
import os
from typing import Callable, List, Optional

import numpy as np
import torch
from PIL import Image

from companion.config import CONFIG
from companion.models.base import InpaintModel
from companion.models.registry import register

logger = logging.getLogger("EffaceMagiqueModelEngine")

PACKAGES = ("diffusers", "transformers", "accelerate")
WEIGHT_SUBDIR = "sd_inpaint"
# huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting @ 8a4288a76071f7280aedbdb3253bdb9e9d5d84bb,
# fp16 safetensors only (install.bat pins the same revision and checks every file's SHA256)
WEIGHT_FILES = (
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)
CPU_WARNING = "Stable Diffusion on CPU: this can take several minutes per variation."


def _missing_packages() -> List[str]:
    return [p for p in PACKAGES if importlib.util.find_spec(p) is None]


@register
class SdInpaintModel(InpaintModel):
    name = "sd_inpaint"
    label = "Stable Diffusion"
    uses_prompt = True
    steps = 20
    max_dim_gpu = 768  # SD 1.5 is trained at 512; far above that it starts repeating content
    max_dim_cpu = 512

    def __init__(self):
        self._pipe = None
        self._device: Optional[torch.device] = None

    @staticmethod
    def weight_dir() -> str:
        return os.path.join(CONFIG.MODELS_DIR, WEIGHT_SUBDIR)

    @classmethod
    def _missing_files(cls) -> List[str]:
        return [f for f in WEIGHT_FILES if not os.path.isfile(os.path.join(cls.weight_dir(), f))]

    @classmethod
    def is_available(cls) -> bool:
        return not _missing_packages() and not cls._missing_files()

    @classmethod
    def unavailable_reason(cls) -> str:
        missing = _missing_packages()
        if missing:
            return f"Stable Diffusion needs {', '.join(missing)} (pip install -r requirements-sd.txt)."
        files = cls._missing_files()
        if files:
            return (f"Stable Diffusion weights missing in {cls.weight_dir()}: {', '.join(files)} "
                    "(re-run install.bat and answer y to Stable Diffusion).")
        return ""

    def load(self, device: torch.device, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        on_cuda = device.type == "cuda"
        if progress_callback and not on_cuda:
            progress_callback(CPU_WARNING)
        missing = _missing_packages()
        if missing:
            raise ImportError(f"Stable Diffusion needs {', '.join(missing)} (pip install -r requirements-sd.txt).")
        files = self._missing_files()
        if files:
            raise FileNotFoundError(f"Stable Diffusion weights missing in {self.weight_dir()}: {', '.join(files)} "
                                    "(re-run install.bat).")
        from diffusers import StableDiffusionInpaintPipeline  # lazy: optional heavy dependency

        # safety_checker=None: decision recorded in factory job 014 build.md (fills small photo
        # regions; a false positive would paste a black square into the user's photo)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.weight_dir(),
            torch_dtype=torch.float16 if on_cuda else torch.float32,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(device)
        if on_cuda:
            pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        self._device = device
        logger.info(f"Stable Diffusion inpainting loaded on {device}.")

    def unload(self) -> None:
        self._pipe = None

    def inpaint(self, crop: Image.Image, mask: Image.Image, seed: int = 0, prompt: Optional[str] = None) -> Image.Image:
        crop_rgb = crop.convert("RGB")
        w, h = crop_rgb.size
        max_dim = self.max_dim_gpu if (self._device is not None and self._device.type == "cuda") else self.max_dim_cpu
        scale = min(1.0, max_dim / max(w, h))
        # floor to a multiple of 8 (the VAE downsamples by 8) so the result never exceeds max_dim
        iw, ih = max(8, int(w * scale) // 8 * 8), max(8, int(h * scale) // 8 * 8)
        img = crop_rgb.resize((iw, ih), Image.Resampling.LANCZOS)
        m = mask.convert("L").resize((iw, ih), Image.Resampling.NEAREST)

        steps = max(1, min(50, int(20 if self.steps is None else self.steps)))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.inference_mode():
            out = self._pipe(
                prompt=prompt or "",
                image=img,
                mask_image=m,
                height=ih,
                width=iw,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]
        result = out.convert("RGB")
        if result.size != (w, h):
            result = result.resize((w, h), Image.Resampling.LANCZOS)

        # SD re-encodes the whole crop through the VAE; keep everything outside the mask byte-identical
        hole = (np.asarray(mask.convert("L")) > 10)[:, :, np.newaxis]
        return Image.fromarray(np.where(hole, np.asarray(result), np.asarray(crop_rgb)))
