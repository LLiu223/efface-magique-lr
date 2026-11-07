"""Fine-tuned LaMa models (job 016): one registry entry per models/finetuned/<name>/ folder.

The folders are produced offline by tools/finetune (never imported here). Each holds a TorchScript
model.pt with big-lama.pt's contract plus meta.json. Only this local folder is scanned: a .pt is code-like.
"""

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Type

import torch
from PIL import Image

from companion.models.lama import LamaModel
from companion.models.registry import register

logger = logging.getLogger("EffaceMagiqueModelEngine")

# shortcut: resolved from the repo/app root, frozen builds are not handled (out of scope for 016)
FINETUNED_DIR = Path(__file__).resolve().parents[2] / "models" / "finetuned"
# Folder names become registry names and HTTP-selectable ids.
_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
# Must match tools/finetune/export.py META_KEYS (checked by tests/test_tuned_models.py).
META_KEYS = ("name", "date", "base_ckpt", "manifest_hash", "steps", "lr", "metrics")


class _TorchScriptLama:
    """SimpleLama.__call__ for an explicit file: SimpleLama.__init__ only reads the LAMA_MODEL env var."""

    def __init__(self, path: Path, device: torch.device):
        self.model = torch.jit.load(str(path), map_location=device)
        self.model.eval()
        self.device = device

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        import numpy as np
        from simple_lama_inpainting.utils.util import prepare_img_and_mask
        image_t, mask_t = prepare_img_and_mask(image, mask, self.device)
        with torch.inference_mode():
            out = self.model(image_t, mask_t)[0].permute(1, 2, 0).detach().cpu().numpy()
        return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


class TunedLamaModel(LamaModel):
    """LamaModel with a fine-tuned TorchScript file; inpaint() is LamaModel's."""

    model_dir: Optional[Path] = None
    _load_error: Optional[str] = None  # set when load() failed, so the entry greys out afterwards

    @classmethod
    def _problem(cls) -> Optional[str]:
        if cls._load_error:
            return cls._load_error
        if not LamaModel.is_available():
            return LamaModel.unavailable_reason()
        pt, meta_path = cls.model_dir / "model.pt", cls.model_dir / "meta.json"
        if not pt.is_file():
            return f"model.pt is missing in {cls.model_dir}."
        if not zipfile.is_zipfile(pt):  # TorchScript files are zip archives; cheap check, loads nothing
            return f"model.pt in {cls.model_dir} is corrupt (not a TorchScript archive)."
        if not meta_path.is_file():
            return f"meta.json is missing in {cls.model_dir}."
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return f"meta.json in {cls.model_dir} is invalid: {e}"
        if not isinstance(meta, dict):
            return f"meta.json in {cls.model_dir} is invalid: not a JSON object."
        missing = [k for k in META_KEYS if k not in meta]
        if missing:
            return f"meta.json in {cls.model_dir} is invalid: missing {', '.join(missing)}."
        return None

    @classmethod
    def is_available(cls) -> bool:
        return cls._problem() is None

    @classmethod
    def unavailable_reason(cls) -> str:
        return cls._problem() or ""

    def load(self, device: torch.device, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        problem = self._problem()
        if problem:
            raise RuntimeError(problem)
        try:
            self._model = _TorchScriptLama(self.model_dir / "model.pt", device)
        except Exception as e:
            type(self)._load_error = f"model.pt in {self.model_dir} could not be loaded: {e}"
            raise RuntimeError(type(self)._load_error) from e
        self._device = device
        logger.info(f"{self.label} loaded on {device} from {self.model_dir}.")


def discover(root: Path = FINETUNED_DIR) -> List[Type[TunedLamaModel]]:
    """Register one TunedLamaModel subclass per valid folder under root. Never raises."""
    found: List[Type[TunedLamaModel]] = []
    try:
        folders = sorted(p for p in Path(root).iterdir() if p.is_dir()) if Path(root).is_dir() else []
    except OSError as e:
        logger.warning(f"Cannot scan {root} for fine-tuned models: {e}")
        return found
    for folder in folders:
        if not _NAME_RE.fullmatch(folder.name):
            logger.warning(f"Skipping fine-tuned model folder {folder.name!r}: name must match [A-Za-z0-9_-]+")
            continue
        cls = type(f"TunedLama_{folder.name}", (TunedLamaModel,), {
            "name": f"lama-tuned-{folder.name}",
            "label": f"LaMa (tuned: {folder.name})",
            "model_dir": folder,
            "_load_error": None,
        })
        found.append(register(cls))
    return found
