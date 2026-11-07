"""InpaintModel: the interface every inpainting model in the registry implements."""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from PIL import Image

from companion.config import CONFIG


class InpaintModel(ABC):
    """One inpainting backend. Instances are created and loaded lazily by InpaintingEngine."""

    name: str = ""            # stable id used by settings, the bridge and the registry
    label: str = ""           # UI text
    uses_prompt: bool = False
    max_dim_gpu: int = CONFIG.GPU_MAX_INFERENCE_DIM
    max_dim_cpu: int = CONFIG.CPU_MAX_INFERENCE_DIM
    steps: Optional[int] = None

    @classmethod
    def is_available(cls) -> bool:
        """Cheap check that deps/weights exist. Must never load the model."""
        return True

    @classmethod
    def unavailable_reason(cls) -> str:
        """Tooltip text shown in the UI when is_available() is False."""
        return f"{cls.label or cls.name} is not installed."

    @abstractmethod
    def load(self, device: torch.device, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Load weights onto `device`. Raise on failure."""

    @abstractmethod
    def unload(self) -> None:
        """Drop every reference to weights/tensors so free_device_memory can reclaim them."""

    @abstractmethod
    def inpaint(self, crop: Image.Image, mask: Image.Image, seed: int, prompt: Optional[str]) -> Image.Image:
        """Fill `mask` (L, 255 = fill) in `crop`; return an RGB image of crop.size."""
