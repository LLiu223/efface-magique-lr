"""Open-vocabulary region detection: CLIPSeg (CIDAS/clipseg-rd64-refined, Apache-2.0) from local files only.

install.bat's optional "find any named object" step puts the files in models/clipseg/ (pinned revision,
SHA256-checked). This module never downloads: from_pretrained gets local_files_only=True.
Loaded lazily on the first query, cached; a load failure is logged once and cached too (no retry per call).
"""

import importlib.util
import logging
import os
import threading
from typing import Optional, Tuple

import torch

from companion.config import CONFIG

logger = logging.getLogger("EffaceMagiqueClipSeg")

WEIGHT_SUBDIR = "clipseg"
WEIGHT_FILES = ("config.json", "preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json",
                "vocab.json", "merges.txt", "model.safetensors")

_CLIPSEG: Optional[Tuple[object, torch.nn.Module]] = None
_LOAD_ERROR: Optional[str] = None
_LOCK = threading.Lock()


def weight_dir() -> str:
    return os.path.join(CONFIG.MODELS_DIR, WEIGHT_SUBDIR)


def is_available() -> bool:
    """transformers importable and every weight file present (same pattern as sd_inpaint)."""
    if importlib.util.find_spec("transformers") is None:
        return False
    return all(os.path.isfile(os.path.join(weight_dir(), f)) for f in WEIGHT_FILES)


def _load(path: str, device: torch.device):
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    processor = CLIPSegProcessor.from_pretrained(path, local_files_only=True)
    model = CLIPSegForImageSegmentation.from_pretrained(path, local_files_only=True)
    return processor, model.to(device).eval()


def get_clipseg() -> Optional[Tuple[object, torch.nn.Module]]:
    """(processor, model) on CUDA when available, else CPU; None when not installed or the load failed."""
    global _CLIPSEG, _LOAD_ERROR
    with _LOCK:
        if _CLIPSEG is not None or _LOAD_ERROR is not None:
            return _CLIPSEG
        if not is_available():
            return None
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        try:
            _CLIPSEG = _load(weight_dir(), device)
            logger.info(f"CLIPSeg open-vocabulary detector loaded on {device}.")
        except Exception as e:
            _LOAD_ERROR = f"{type(e).__name__}: {e}"[:200]
            logger.warning(f"Could not load CLIPSeg from {weight_dir()}: {_LOAD_ERROR}. Falling back to VOC classes.")
        return _CLIPSEG


def unavailable_note() -> str:
    """Tool-result note when a query was given but CLIPSeg could not be used."""
    if _LOAD_ERROR is not None:
        return ("The open-vocabulary detector failed to load (see the log); only VOC classes were searched. "
                "Tell the user.")
    return ("Open-vocabulary detection is not installed: re-run install.bat and answer y to the find-any-object "
            "step. Only VOC classes were searched. Tell the user.")


def status() -> str:
    """For Copy diagnostics: not installed | installed | loaded | failed (...)."""
    if _CLIPSEG is not None:
        return "loaded"
    if _LOAD_ERROR is not None:
        return f"failed ({_LOAD_ERROR})"
    return "installed" if is_available() else "not installed"
