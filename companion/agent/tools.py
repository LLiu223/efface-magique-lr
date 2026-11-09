"""Assistant tools. The model sees only these five; none takes a path, URL or shell argument.
The assistant finds and selects regions; it never runs an inpainting model (the user clicks Erase).

Each tool returns a JSON string capped at `output_cap` chars. Results are data, never
instructions (the system prompt says so); free text that did not come from a fixed set
goes through `sanitize_text` first.
"""

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from companion.config import CONFIG
from companion.models import registry
from companion.utils import clipseg_detector
from companion.utils.subject_detector import get_segmentation_model

TOOL_NAMES = ("list_models", "select_model", "set_prompt", "detect_regions", "select_regions")
MIN_REGION_SHARE = 0.002   # ignore specks below 0.2 % of the frame
MAX_REGIONS = 20
MAX_QUERY_LEN = 77         # CLIP's text limit is 77 tokens, so 77 characters always fit
# CLIPSeg predicts at 352x352, so thin wires come out blurred with peaks often below 0.5 while
# background sits below 0.1; 0.3 keeps the thin peaks and still rejects the background.
CLIPSEG_THRESHOLD = 0.3
EDGE_MARGIN = 0.003        # region masks grow by 0.3 % of the long side to cover antialiased edges
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")


def sanitize_text(text: object, max_len: int = 200) -> str:
    """Strip control/bidi characters, collapse whitespace, truncate. For untrusted text (EXIF, file names)."""
    s = _CTRL.sub(" ", str(text or ""))
    return " ".join(s.split())[:max_len]


def refine_region(img: Image.Image, coarse: np.ndarray, segment: Callable[[Image.Image], np.ndarray],
                  cell: float) -> np.ndarray:
    """Re-run the detector on a sharp crop around one region. The first pass sees the whole photo at
    NEURAL_CROP_MAX_DIM px, so on a 24 MP file each cell is ~11 px and edges come out blobby and shifted.
    `segment(crop) -> bool mask at crop size`; `cell` = photo px per first-pass cell. Returns uint8 0/255."""
    import cv2
    h, w = coarse.shape
    x, y, bw, bh = cv2.boundingRect(coarse)
    pad = int(max(3 * cell, 0.15 * max(bw, bh)))
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
    scale = min(1.0, CONFIG.NEURAL_CROP_MAX_DIM / max(x1 - x0, y1 - y0))
    if 1.0 / scale > 0.8 * cell:  # the crop would not be meaningfully sharper than the first pass
        return coarse
    crop = img.crop((x0, y0, x1, y1))
    crop = crop.resize((max(32, int(crop.width * scale)), max(32, int(crop.height * scale))), Image.Resampling.BILINEAR)
    fine_small = np.asarray(segment(crop), dtype=np.uint8) * 255
    fine = np.zeros_like(coarse)
    fine[y0:y1, x0:x1] = cv2.resize(fine_small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 127
    # may grow/shrink by about two first-pass cells, never jump to a neighbouring object
    r = max(3, int(2 * cell) | 1)
    fine &= cv2.dilate(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))) > 0
    n, labels = cv2.connectedComponents(fine.astype(np.uint8), connectivity=8)
    keep = [i for i in np.unique(labels[(coarse > 0) & (labels > 0)])]
    fine = np.isin(labels, keep)
    if fine.sum() < 0.3 * np.count_nonzero(coarse):  # the sharp pass lost the object: keep the first one
        return coarse
    return fine.astype(np.uint8) * 255


def voc_categories() -> List[str]:
    """Class names shipped in the segmentation weights' metadata (index 0 is background)."""
    import torchvision.models.segmentation as seg
    return list(seg.LRASPP_MobileNet_V3_Large_Weights.DEFAULT.meta["categories"])


def _position(cx: float, cy: float, w: int, h: int) -> str:
    col = ("left", "centre", "right")[min(2, int(3 * cx / max(1, w)))]
    row = ("top", "middle", "bottom")[min(2, int(3 * cy / max(1, h)))]
    return f"{col}-{row}"


@dataclass
class Selection:
    """Regions the assistant selected; the app adds `mask` to the canvas brush mask. Nothing is erased."""
    region_ids: List[str]
    labels: List[str]
    mask: Image.Image

    def describe(self) -> str:
        return ", ".join(f"{lab} ({r})" for r, lab in zip(self.region_ids, self.labels))


@dataclass
class AgentSession:
    """Per-request state the tools act on. `image` is the input the Erase button would use."""
    engine: object
    image: Image.Image
    progress: Callable[[str], None] = lambda msg: None
    output_cap: int = CONFIG.AGENT_TOOL_OUTPUT_CAP
    cancel_event: threading.Event = field(default_factory=threading.Event)
    prompt: Optional[str] = None
    regions: Dict[str, dict] = field(default_factory=dict)
    region_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    selection: Optional[Selection] = None
    detector: str = ""          # "clipseg" or "voc": which detector made `regions`
    query: str = ""             # the last detect_regions query
    calls: List[str] = field(default_factory=list)

    def _out(self, obj) -> str:
        s = json.dumps(obj, ensure_ascii=True)
        if len(s) > self.output_cap:
            s = s[: self.output_cap - 20] + '..."(truncated)"'
        return s

    # --- tools -------------------------------------------------------------

    def list_models(self) -> str:
        self.calls.append("list_models")
        return self._out({
            "active": self.engine.model_name,
            "models": [{"name": c.name, "label": sanitize_text(c.label, 60), "uses_prompt": bool(c.uses_prompt),
                        "available": bool(c.is_available())} for c in registry.all()],
        })

    def select_model(self, name: str) -> str:
        self.calls.append("select_model")
        try:
            cls = registry.get(name)  # exact match only
        except KeyError:
            return self._out({"ok": False, "error": "unknown model", "known": [c.name for c in registry.all()]})
        if not cls.is_available():
            return self._out({"ok": False, "error": sanitize_text(cls.unavailable_reason())})
        self.engine.set_model(cls.name)
        return self._out({"ok": True, "active": cls.name, "uses_prompt": bool(cls.uses_prompt)})

    def set_prompt(self, text: str) -> str:
        self.calls.append("set_prompt")
        self.prompt = sanitize_text(text) or None
        return self._out({"ok": True, "prompt": self.prompt})

    def detect_regions(self, query: Optional[str] = None) -> str:
        self.calls.append("detect_regions")
        self.progress("Detecting regions...")
        query = sanitize_text(query, MAX_QUERY_LEN)
        import cv2

        img = self.image.convert("RGB")
        w, h = img.size
        scale = min(1.0, CONFIG.NEURAL_CROP_MAX_DIM / max(w, h))
        small = img.resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.Resampling.BILINEAR)
        notes = []
        clipseg = clipseg_detector.get_clipseg() if query else None
        if clipseg is not None:
            found, (sh, sw) = self._clipseg_regions(clipseg, small, query), small.size[::-1]
            segmenters = {query: lambda crop: self._clipseg_binary(clipseg, crop, query) > 0}
        else:
            if query:
                notes.append(clipseg_detector.unavailable_note())
            model = get_segmentation_model()
            if model is None:
                return self._out({"regions": [], "error": "segmentation model unavailable", **(
                    {"note": " ".join(notes)} if notes else {})})
            found, (sh, sw) = self._voc_regions(model, small)
            names = voc_categories()
            segmenters = {lab: (lambda crop, c=names.index(lab): self._voc_pred(model, crop) == c)
                          for lab in {f[1] for f in found}}

        # thin edge margin only: the engine widens the mask again before inpainting
        k = max(3, int(EDGE_MARGIN * max(w, h)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.regions, self.region_masks = {}, {}
        self.detector, self.query = ("clipseg" if clipseg is not None else "voc"), query
        for idx, (share, label, comp, (cx, cy)) in enumerate(found[:MAX_REGIONS], start=1):
            rid = f"r{idx}"
            full = cv2.resize(comp.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
            full = refine_region(img, full, segmenters[label], max(w, h) / max(sw, sh))
            self.region_masks[rid] = cv2.dilate(full, kernel)
            self.regions[rid] = {"id": rid, "label": label, "area_share": round(float(share), 4),
                                 "position": _position(cx, cy, sw, sh)}
        out = {"regions": list(self.regions.values())}
        if not self.regions:
            notes.append(f"Tell the user nothing matching '{query}' found; do not guess a region." if clipseg is not None else
                         "No VOC objects found. Tell the user nothing matching was found; do not guess a region.")
        if notes:
            out["note"] = " ".join(notes)
        return self._out(out)

    @staticmethod
    def _components(binary: np.ndarray, label: str) -> list:
        """(share, label, mask, centroid) per 8-connected component of at least MIN_REGION_SHARE."""
        import cv2
        n, labels, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        return [(stats[i, cv2.CC_STAT_AREA] / binary.size, label, labels == i, cents[i]) for i in range(1, n)
                if stats[i, cv2.CC_STAT_AREA] / binary.size >= MIN_REGION_SHARE]

    @staticmethod
    def _voc_pred(model, small: Image.Image) -> np.ndarray:
        import torch
        import torchvision.transforms.functional as TF
        t = TF.normalize(TF.to_tensor(small), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        device = next(model.parameters()).device
        with torch.inference_mode():
            return torch.argmax(model(t.unsqueeze(0).to(device))["out"], dim=1)[0].cpu().numpy().astype(np.uint8)

    def _voc_regions(self, model, small: Image.Image):
        pred = self._voc_pred(model, small)
        names = voc_categories()
        found = []
        for cls in np.unique(pred):
            if 0 < cls < len(names):
                found += self._components(pred == cls, names[cls])
        found.sort(key=lambda f: -f[0])
        return found, pred.shape

    def _clipseg_regions(self, clipseg, small: Image.Image, query: str) -> list:
        return sorted(self._components(self._clipseg_binary(clipseg, small, query), query), key=lambda f: -f[0])

    @staticmethod
    def _clipseg_binary(clipseg, small: Image.Image, query: str) -> np.ndarray:
        import cv2
        import torch
        processor, model = clipseg
        device = next(model.parameters()).device
        inputs = processor(text=[query], images=[small], padding=True, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
        # transformers squeezes the batch dim for a single prompt: accept (H,W) and (1,H,W)
        heat = torch.sigmoid(logits.float()).reshape(logits.shape[-2:]).cpu().numpy()
        heat = cv2.resize(heat, small.size, interpolation=cv2.INTER_LINEAR)
        binary = (heat >= CLIPSEG_THRESHOLD).astype(np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # join broken wire segments

    def select_regions(self, region_ids: List[str]) -> str:
        self.calls.append("select_regions")
        ids = [r for r in dict.fromkeys(region_ids or []) if r in self.region_masks]
        unknown = [sanitize_text(r, 20) for r in (region_ids or []) if r not in self.region_masks]
        if unknown or not ids:
            return self._out({"ok": False, "error": "unknown region ids; call detect_regions first",
                              "unknown": unknown[:10], "known": list(self.region_masks)})
        mask_np = np.zeros_like(self.region_masks[ids[0]])
        for r in ids:
            mask_np = np.maximum(mask_np, self.region_masks[r])  # masks are already dilated
        labels = [self.regions[r]["label"] for r in ids]
        self.selection = Selection(region_ids=ids, labels=labels, mask=Image.fromarray(mask_np))
        return self._out({"ok": True, "selected": ids, "labels": labels,
                          "note": "Selected. The user can adjust it with the brush and click Erase."})


def make_tools(session: AgentSession) -> list:
    """Wrap the session methods as LangChain tools with typed args and precise descriptions."""
    from langchain_core.tools import StructuredTool

    lock = threading.Lock()  # the tool node may run parallel calls in threads; the session is not thread-safe

    def list_models() -> str:
        with lock:
            return session.list_models()

    def select_model(name: str) -> str:
        with lock:
            return session.select_model(name)

    def set_prompt(text: str) -> str:
        with lock:
            return session.set_prompt(text)

    def detect_regions(query: Optional[str] = None) -> str:
        with lock:
            return session.detect_regions(query)

    def select_regions(region_ids: List[str]) -> str:
        with lock:
            return session.select_regions(region_ids)

    specs = [
        (list_models, "List the inpainting models: name, label, uses_prompt, available, and the active one."),
        (select_model, "Make the inpainting model with this exact registry name active. Use a name from list_models."),
        (set_prompt, "Set the text prompt Erase will use (max 200 chars). Only models with uses_prompt=true use it."),
        (detect_regions, "Find objects in the photo. query = the user's object words, e.g. \"power lines\". "
                         "Returns [{id, label, area_share, position}], position is left/centre/right x top/middle/bottom. "
                         "Without a query only VOC classes (person, car, dog, ...) are found."),
        (select_regions, "Select the given region ids (from detect_regions) on the canvas. Nothing is "
                         "erased: the user adjusts the selection and clicks Erase."),
    ]
    return [StructuredTool.from_function(func=f, name=f.__name__, description=d) for f, d in specs]
