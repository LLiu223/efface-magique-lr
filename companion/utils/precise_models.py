"""Precise selection models, local files only (install.bat's optional "Precise selection" step, pinned and SHA256-checked):

- Grounding DINO  IDEA-Research/grounding-dino-tiny      (Apache-2.0)  query -> one box per instance
- SAM 2.1         facebook/sam2.1-hiera-small            (Apache-2.0)  box (+points) -> outline
- ViTMatte        hustvl/vitmatte-small-composition-1k   (Apache-2.0)  trimap -> soft hair/fur alpha

Same contract as clipseg_detector: never downloads (local_files_only=True), loads lazily on first use, cached;
a load failure is logged once and cached. CUDA when available.
"""

import importlib.util
import logging
import os
import threading
from typing import Callable, Optional, Tuple

import torch

from companion.config import CONFIG

logger = logging.getLogger("EffaceMagiquePrecise")


class LocalModel:
    def __init__(self, name: str, subdir: str, files: tuple, loader: Callable, fp16: bool):
        self.name, self.WEIGHT_SUBDIR, self.WEIGHT_FILES = name, subdir, files
        self._loader, self.fp16 = loader, fp16
        self._model: Optional[Tuple[object, torch.nn.Module]] = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    def weight_dir(self) -> str:
        return os.path.join(CONFIG.MODELS_DIR, self.WEIGHT_SUBDIR)

    def is_available(self) -> bool:
        if importlib.util.find_spec("transformers") is None:
            return False
        return all(os.path.isfile(os.path.join(self.weight_dir(), f)) for f in self.WEIGHT_FILES)

    def get(self) -> Optional[Tuple[object, torch.nn.Module]]:
        """(processor, model) or None when not installed or the load failed."""
        with self._lock:
            if self._model is not None or self._error is not None:
                return self._model
            if not self.is_available():
                return None
            cuda = torch.cuda.is_available()
            dtype = torch.float16 if cuda and self.fp16 else torch.float32
            try:
                processor, model = self._loader(self.weight_dir())
                self._model = processor, model.to(torch.device("cuda" if cuda else "cpu"), dtype).eval()
                logger.info(f"{self.name} loaded on {'cuda' if cuda else 'cpu'} ({dtype}).")
            except Exception as e:
                self._error = f"{type(e).__name__}: {e}"[:200]
                logger.warning(f"Could not load {self.name} from {self.weight_dir()}: {self._error}. Falling back.")
            return self._model

    def failed(self) -> bool:
        return self._error is not None

    def status(self) -> str:
        """For Copy diagnostics: not installed | installed | loaded | failed (...)."""
        if self._model is not None:
            return "loaded"
        if self._error is not None:
            return f"failed ({self._error})"
        return "installed" if self.is_available() else "not installed"


def _gdino(path):
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
    return (GroundingDinoProcessor.from_pretrained(path, local_files_only=True),
            GroundingDinoForObjectDetection.from_pretrained(path, local_files_only=True))


def _sam2(path):
    from transformers import Sam2Model, Sam2Processor
    # the repo's config is the video model; the image model loads its subset of the weights
    return (Sam2Processor.from_pretrained(path, local_files_only=True),
            Sam2Model.from_pretrained(path, local_files_only=True))


def _vitmatte(path):
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    return (VitMatteImageProcessor.from_pretrained(path, local_files_only=True),
            VitMatteForImageMatting.from_pretrained(path, local_files_only=True))


# fp16 only for SAM 2, whose reference predictor runs in half-precision autocast. Grounding DINO (text-image fusion,
# 0.25 text threshold) and ViTMatte (fine alpha) stay fp32: their fp16 output is unverified and they fit (~0.8 GB).
GROUNDING_DINO = LocalModel("Grounding DINO", "grounding_dino",
                            ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                             "special_tokens_map.json", "vocab.txt", "added_tokens.json", "model.safetensors"),
                            _gdino, fp16=False)
SAM2 = LocalModel("SAM 2.1", "sam2", ("config.json", "preprocessor_config.json", "processor_config.json",
                                      "model.safetensors"), _sam2, fp16=True)
VITMATTE = LocalModel("ViTMatte", "vitmatte", ("config.json", "preprocessor_config.json", "model.safetensors"),
                      _vitmatte, fp16=False)
STAGES = {"grounding_dino": GROUNDING_DINO, "sam2": SAM2, "vitmatte": VITMATTE}
