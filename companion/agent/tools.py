"""Assistant tools. The model sees only these five; none takes a path, URL or shell argument.

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
from companion.utils.subject_detector import get_segmentation_model

TOOL_NAMES = ("list_models", "select_model", "set_prompt", "detect_regions", "inpaint_region")
MIN_REGION_SHARE = 0.002   # ignore specks below 0.2 % of the frame
MAX_REGIONS = 20
MAX_VARIATIONS = 3
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")


def sanitize_text(text: object, max_len: int = 200) -> str:
    """Strip control/bidi characters, collapse whitespace, truncate. For untrusted text (EXIF, file names)."""
    s = _CTRL.sub(" ", str(text or ""))
    return " ".join(s.split())[:max_len]


def voc_categories() -> List[str]:
    """Class names shipped in the segmentation weights' metadata (index 0 is background)."""
    import torchvision.models.segmentation as seg
    return list(seg.LRASPP_MobileNet_V3_Large_Weights.DEFAULT.meta["categories"])


def _position(cx: float, cy: float, w: int, h: int) -> str:
    col = ("left", "centre", "right")[min(2, int(3 * cx / max(1, w)))]
    row = ("top", "middle", "bottom")[min(2, int(3 * cy / max(1, h)))]
    return f"{col}-{row}"


@dataclass
class Proposal:
    model: str
    prompt: Optional[str]
    region_ids: List[str]
    labels: List[str]
    mask: Image.Image
    variations: List[Image.Image]
    base_image: Image.Image

    def describe(self) -> str:
        regions = ", ".join(f"{r} ({lab})" for r, lab in zip(self.region_ids, self.labels))
        p = f", prompt {self.prompt!r}" if self.prompt else ""
        return f"model {self.model}{p}, regions {regions}"


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
    proposal: Optional[Proposal] = None
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

    def detect_regions(self) -> str:
        self.calls.append("detect_regions")
        self.progress("Detecting regions...")
        model = get_segmentation_model()
        if model is None:
            return self._out({"regions": [], "error": "segmentation model unavailable"})
        import cv2
        import torch
        import torchvision.transforms.functional as TF

        img = self.image.convert("RGB")
        w, h = img.size
        scale = min(1.0, CONFIG.NEURAL_CROP_MAX_DIM / max(w, h))
        small = img.resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.Resampling.BILINEAR)
        t = TF.normalize(TF.to_tensor(small), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        device = next(model.parameters()).device
        with torch.inference_mode():
            pred = torch.argmax(model(t.unsqueeze(0).to(device))["out"], dim=1)[0].cpu().numpy().astype(np.uint8)

        names = voc_categories()
        found = []
        for cls in np.unique(pred):
            if cls == 0 or cls >= len(names):
                continue
            n, labels, stats, cents = cv2.connectedComponentsWithStats((pred == cls).astype(np.uint8), connectivity=8)
            for i in range(1, n):
                share = stats[i, cv2.CC_STAT_AREA] / pred.size
                if share >= MIN_REGION_SHARE:
                    found.append((share, names[cls], labels == i, cents[i]))
        found.sort(key=lambda f: -f[0])

        sh, sw = pred.shape
        k = max(3, int(0.01 * max(w, h)) | 1)  # small dilation so the inpaint covers object edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.regions, self.region_masks = {}, {}
        for idx, (share, label, comp, (cx, cy)) in enumerate(found[:MAX_REGIONS], start=1):
            rid = f"r{idx}"
            full = cv2.resize(comp.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
            self.region_masks[rid] = cv2.dilate(full, kernel)
            self.regions[rid] = {"id": rid, "label": label, "area_share": round(float(share), 4),
                                 "position": _position(cx, cy, sw, sh)}
        out = {"regions": list(self.regions.values())}
        if not self.regions:
            out["note"] = "No VOC objects found. Tell the user nothing matching was found; do not guess a region."
        return self._out(out)

    def inpaint_region(self, region_ids: List[str], num_variations: int = 1) -> str:
        self.calls.append("inpaint_region")
        ids = [r for r in dict.fromkeys(region_ids or []) if r in self.region_masks]
        unknown = [sanitize_text(r, 20) for r in (region_ids or []) if r not in self.region_masks]
        if unknown or not ids:
            return self._out({"ok": False, "error": "unknown region ids; call detect_regions first",
                              "unknown": unknown[:10], "known": list(self.region_masks)})
        n = max(1, min(MAX_VARIATIONS, int(num_variations or 1)))
        mask_np = np.zeros_like(self.region_masks[ids[0]])
        for r in ids:
            mask_np = np.maximum(mask_np, self.region_masks[r])
        mask = Image.fromarray(mask_np)
        cls = registry.get(self.engine.model_name)
        self.progress(f"Running {cls.label or cls.name}...")
        prompt = self.prompt if cls.uses_prompt else None
        # Same call InpaintingWorker.run makes; runs on this (agent) worker thread
        variations = self.engine.generate_variations(
            image=self.image, mask=mask, num_variations=n, prompt=prompt,
            detect_subject=False, enable_grain=False,
            progress_callback=lambda pct, msg: self.progress(f"{cls.label or cls.name}: {pct}%"),
        )
        self.proposal = Proposal(model=cls.name, prompt=prompt, region_ids=ids,
                                 labels=[self.regions[r]["label"] for r in ids], mask=mask,
                                 variations=list(variations), base_image=self.image)
        return self._out({"ok": True, "proposed": True, "regions": ids, "variations": len(variations),
                          "note": "Preview ready. Nothing is committed until the user clicks Apply."})


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

    def detect_regions() -> str:
        with lock:
            return session.detect_regions()

    def inpaint_region(region_ids: List[str], num_variations: int = 1) -> str:
        with lock:
            return session.inpaint_region(region_ids, num_variations)

    specs = [
        (list_models, "List the inpainting models: name, label, uses_prompt, available, and the active one."),
        (select_model, "Make the inpainting model with this exact registry name active. Use a name from list_models."),
        (set_prompt, "Set the text prompt (max 200 chars). Only models with uses_prompt=true use it."),
        (detect_regions, "Find objects in the photo. Returns [{id, label, area_share, position}] where position is "
                         "left/centre/right x top/middle/bottom. Only VOC classes are found: aeroplane, bicycle, "
                         "bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, "
                         "pottedplant, sheep, sofa, train, tvmonitor."),
        (inpaint_region, "Erase the given region ids (from detect_regions) with the active model and prepare a "
                         "preview proposal (1-3 variations). The user must click Apply to commit it."),
    ]
    return [StructuredTool.from_function(func=f, name=f.__name__, description=d) for f, d in specs]
