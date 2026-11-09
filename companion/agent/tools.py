"""Assistant tools. The model sees only these five; none takes a path, URL or shell argument.
The assistant finds and selects regions; it never runs an inpainting model (the user clicks Erase).

Each tool returns a JSON string capped at `output_cap` chars. Results are data, never
instructions (the system prompt says so); free text that did not come from a fixed set
goes through `sanitize_text` first.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from companion.config import CONFIG
from companion.models import registry
from companion.utils import precise_models as precise
from companion.utils.subject_detector import get_segmentation_model

logger = logging.getLogger("EffaceMagiqueAgent")

TOOL_NAMES = ("list_models", "select_model", "set_prompt", "detect_regions", "select_regions")
MIN_REGION_SHARE = 0.002   # ignore specks below 0.2 % of the frame (coarse VOC components)
MIN_OUTLINE_SHARE = 0.0001  # SAM outlines are exact: keep objects down to 0.01 % (about 47x47 px on 22 MP)
MAX_REGIONS = 20
MAX_QUERY_LEN = 77         # a short noun phrase; far below Grounding DINO's 256-token text limit
EDGE_MARGIN = 0.003        # region masks grow by 0.3 % of the long side to cover antialiased edges
# Grounding DINO's model card and paper defaults (box_threshold=0.35, text_threshold=0.25)
GDINO_BOX_THRESHOLD = 0.35
GDINO_TEXT_THRESHOLD = 0.25
GDINO_MERGE_IOU = 0.7      # boxes overlapping more than this are one instance found twice
# Grounding DINO's own input size (its processor caps the long side at 1333). The 512 px used for the
# coarse VOC pass shrinks a 22 MP photo 11x, and small objects become a few pixels it cannot recognise.
GDINO_INPUT_DIM = 1333
GDINO_TILE = 2 * GDINO_INPUT_DIM   # photo px per tile, so objects get 2x the whole-view resolution...
GDINO_TILE_OVERLAP = 0.25          # ...and anything smaller than a quarter tile is whole in some tile
SAM_PAD = 0.15             # SAM sees the box plus 15 % of its long side as context
SAM_MIN_COVERAGE = 0.3     # SAM must cover 30 % of a VOC region, else it missed it (thin wires)
SAM_MAX_POINTS = 3
MATTE_KEEP = 0.1           # the canvas is binary (> 10): a low cut keeps stray hairs
MATTE_MAX_DIM = 1024
MATTE_WORDS = frozenset(
    "person persons people man men woman women child children kid kids boy boys girl girls baby babies human humans "
    "face faces head heads hair fur beard animal animals pet pets bird birds cat cats cow cows dog dogs horse horses "
    "sheep puppy puppies kitten kittens".split())
# people/animals that can hold, wear or carry things: MATTE_WORDS without the body parts
HOLDER_WORDS = MATTE_WORDS - frozenset("face faces head heads hair fur beard".split())
# a fixed list, never user text: Grounding DINO finds only what is named, so no ground or sky through gaps
HELD_QUERY = "bag. backpack. handbag. drink. cup. bottle. carton. phone. umbrella. camera. hat. glasses. leash."
MAX_HELD = 5               # SAM calls per person
HELD_TOUCH = 0.02          # contact: the person mask dilated by 2 % of its long side (min 3 px)
HELD_BOX_PAD = 0.1         # a held object lies >= 70 % inside the person's box grown by 10 % of its long side
HELD_INSIDE = 0.7
HELD_MAX_AREA = 0.5        # never more than half the person's area (ground, sky, a wall)
HELD_MAX_OTHER = 0.5       # an object half covered by another person region is that person's
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


def interior_points(mask: np.ndarray, n: int = SAM_MAX_POINTS) -> list:
    """Up to n well-separated (x, y) points deep inside the mask (distance-transform maxima)."""
    import cv2
    # padded so a mask touching the frame edge still has background around it
    dist = cv2.distanceTransform(np.pad(mask > 0, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1].copy()
    pts = []
    for _ in range(n):
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        r = float(dist[y, x])
        if r < 1:
            break
        pts.append((int(x), int(y)))
        cv2.circle(dist, (int(x), int(y)), int(2 * r) + 1, 0, -1)
    return pts


def wants_matte(label: str) -> bool:
    """Soft hair/fur edges are worth ViTMatte only for people and animals."""
    return any(word in MATTE_WORDS for word in re.findall(r"[a-z]+", label.lower()))


def is_holder(label: str) -> bool:
    """People and animals get what they hold, wear or carry merged into their region."""
    return any(word in HOLDER_WORDS for word in re.findall(r"[a-z]+", label.lower()))


def voc_categories() -> List[str]:
    """Class names shipped in the segmentation weights' metadata (index 0 is background)."""
    import torchvision.models.segmentation as seg
    return list(seg.LRASPP_MobileNet_V3_Large_Weights.DEFAULT.meta["categories"])


def _free_cuda():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to(inputs, param) -> dict:
    """Processor outputs on the model's device; float tensors in its dtype (fp16 on CUDA for SAM 2)."""
    return {k: v.to(param.device, param.dtype) if v.is_floating_point() else v.to(param.device)
            for k, v in inputs.items()}


def _position(cx: float, cy: float, w: int, h: int) -> str:
    col = ("left", "centre", "right")[min(2, int(3 * cx / max(1, w)))]
    row = ("top", "middle", "bottom")[min(2, int(3 * cy / max(1, h)))]
    return f"{col}-{row}"


def _tiles(w: int, h: int) -> list:
    """Overlapping GDINO_TILE squares covering the photo, edge tiles flush with the border. [] when the
    whole view already shows the photo at no less than half Grounding DINO's own resolution."""
    if max(w, h) <= GDINO_TILE:
        return []
    stride = int(GDINO_TILE * (1 - GDINO_TILE_OVERLAP))

    def starts(n: int) -> list:
        if n <= GDINO_TILE:
            return [0]
        k = -(-(n - GDINO_TILE) // stride)  # ceil: last tile ends exactly at the border
        return [round(i * (n - GDINO_TILE) / k) for i in range(k + 1)]

    return [(x, y, min(w, x + GDINO_TILE), min(h, y + GDINO_TILE)) for y in starts(h) for x in starts(w)]


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
    detector: str = ""          # "grounding_dino" or "voc": which detector made `regions`
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
        precise_error = False
        outlined = []  # (share, label, full-size uint8 0/255 mask, position)
        warned = set()

        def stage_failed(stage: str, e: Exception):
            """A stage that raises (e.g. CUDA out of memory next to Stable Diffusion) falls back; one warning per call."""
            if stage not in warned:
                warned.add(stage)
                logger.warning(f"{stage} failed, falling back: {type(e).__name__}: {str(e)[:200]}")
            _free_cuda()

        # Grounding DINO only gives boxes, so it is used when SAM 2 can outline them
        gdino = precise.GROUNDING_DINO.get() if query and precise.SAM2.get() is not None else None
        if gdino is not None:
            detector = "grounding_dino"
            self.progress("Finding objects...")
            try:
                for label, box in self._gdino_find(gdino, img, query)[:MAX_REGIONS]:
                    self.progress("Outlining...")
                    mask = self._sam_outline(img, box)
                    share = np.count_nonzero(mask) / mask.size
                    if share >= MIN_OUTLINE_SHARE:
                        outlined.append((share, label, mask,
                                         _position((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, w, h)))
            except Exception as e:
                stage_failed("Grounding DINO / SAM 2", e)
                gdino, outlined, precise_error = None, [], True  # run the VOC path instead
            outlined.sort(key=lambda f: -f[0])
        if gdino is None:
            if query:
                notes.append(precise.unavailable_note(precise_error))
            model = get_segmentation_model()
            if model is None:
                return self._out({"regions": [], "error": "segmentation model unavailable", **(
                    {"note": " ".join(notes)} if notes else {})})
            found, (sh, sw) = self._voc_regions(model, small)
            names = voc_categories()
            segmenters = {lab: (lambda crop, c=names.index(lab): self._voc_pred(model, crop) == c)
                          for lab in {f[1] for f in found}}
            detector = "voc"
            for share, label, comp, (cx, cy) in found[:MAX_REGIONS]:
                coarse = cv2.resize(comp.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
                self.progress("Outlining...")
                try:
                    full = self._sam_refine(img, coarse)
                except Exception as e:
                    stage_failed("SAM 2", e)
                    full = None
                if full is None:
                    full = refine_region(img, coarse, segmenters[label], max(w, h) / max(sw, sh))
                outlined.append((share, label, full, _position(cx, cy, sw, sh)))

        # thin edge margin only: the engine widens the mask again before inpainting
        k = max(3, int(EDGE_MARGIN * max(w, h)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.regions, self.region_masks = {}, {}
        self.detector, self.query = detector, query
        holders = [f[2] for f in outlined if is_holder(f[1])]
        for idx, (share, label, full, position) in enumerate(outlined, start=1):
            rid = f"r{idx}"
            held = []
            if is_holder(label):
                try:
                    union, held = self._add_held(img, full, [m for m in holders if m is not full])
                except Exception as e:
                    stage_failed("Held objects", e)  # keep the plain person mask
                if held:
                    full = union
                    share = np.count_nonzero(full) / full.size
                    ys, xs = np.nonzero(full)
                    position = _position(xs.mean(), ys.mean(), w, h)
            if wants_matte(label) and precise.VITMATTE.get() is not None:
                self.progress("Refining hair and fur edges...")
                try:
                    full = self._matte(img, full)
                except Exception as e:
                    stage_failed("ViTMatte", e)  # keep the outline without soft edges
            self.region_masks[rid] = cv2.dilate(full, kernel)
            self.regions[rid] = {"id": rid, "label": label, "area_share": round(float(share), 4), "position": position}
            if held:
                self.regions[rid]["with"] = held
        out = {"regions": list(self.regions.values())}
        if not self.regions:
            notes.append(f"Tell the user nothing matching '{query}' found; do not guess a region." if detector != "voc" else
                         "No VOC objects found. Tell the user nothing matching was found; do not guess a region.")
        if notes:
            out["note"] = " ".join(notes)
        return self._out(out)

    def _add_held(self, img: Image.Image, mask: np.ndarray, others: List[np.ndarray]):
        """(mask | held objects, [their labels]). One Grounding DINO call with HELD_QUERY on the person's crop,
        one SAM call per candidate (MAX_HELD); kept only when touching the person, inside its box, small, and
        not another person's. ([] when Grounding DINO or SAM is missing, or nothing was added.)"""
        import cv2
        gdino = precise.GROUNDING_DINO.get()
        if gdino is None or precise.SAM2.get() is None or not mask.any():
            return mask, []
        h, w = mask.shape
        x, y, bw, bh = cv2.boundingRect(mask)
        long_side = max(bw, bh)
        pad = SAM_PAD * long_side
        x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
        x1, y1 = min(w, int(x + bw + pad) + 1), min(h, int(y + bh + pad) + 1)
        scale = min(1.0, GDINO_INPUT_DIM / max(x1 - x0, y1 - y0))
        crop = img.crop((x0, y0, x1, y1))
        crop = crop.resize((max(32, int(crop.width * scale)), max(32, int(crop.height * scale))),
                           Image.Resampling.BILINEAR)
        self.progress("Finding what they hold...")
        found = [(lab, box) for lab, box, _ in self._gdino_boxes(gdino, crop, HELD_QUERY) if not is_holder(lab)]
        if not found:
            return mask, []
        person = mask > 0
        r = max(3, int(HELD_TOUCH * long_side))
        near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        e = int(HELD_BOX_PAD * long_side)
        bx0, by0, bx1, by1 = max(0, x - e), max(0, y - e), min(w, x + bw + e), min(h, y + bh + e)
        other = np.zeros_like(person)
        for o in others:
            other |= o > 0
        n_person = np.count_nonzero(person)
        union, labels = person.copy(), []
        fx, fy = (x1 - x0) / crop.width, (y1 - y0) / crop.height
        for lab, box in found[:MAX_HELD]:
            box = np.array([box[0] * fx + x0, box[1] * fy + y0, box[2] * fx + x0, box[3] * fy + y0])
            m = self._sam_outline(img, box) > 0
            n = np.count_nonzero(m)
            if (n == 0 or not (m & near).any()                                    # (a) touches the person
                    or np.count_nonzero(m[by0:by1, bx0:bx1]) < HELD_INSIDE * n    # (b) inside the box
                    or n > HELD_MAX_AREA * n_person                               # (c) not ground or sky
                    or np.count_nonzero(m & other) >= HELD_MAX_OTHER * n):        # (e) not someone else's
                continue
            if (m & ~union).any():
                union |= m
                labels.append("object" if lab == HELD_QUERY else lab)  # an empty span falls back to the query
        if not labels:
            return mask, []
        return union.astype(np.uint8) * 255, list(dict.fromkeys(labels))

    def _gdino_find(self, gdino, img: Image.Image, query: str) -> list:
        """[(label, xyxy box in photo px)], best score first: the whole photo at Grounding DINO's own size,
        plus overlapping tiles on large photos so small objects get enough pixels."""
        import torch
        from torchvision.ops import nms
        w, h = img.size
        views = [(0, 0, w, h)] + _tiles(w, h)
        labels, boxes, scores = [], [], []
        for vx0, vy0, vx1, vy1 in views:
            view = img.crop((vx0, vy0, vx1, vy1))
            s = min(1.0, GDINO_INPUT_DIM / max(view.size))
            view = view.resize((max(32, int(view.width * s)), max(32, int(view.height * s))), Image.Resampling.BILINEAR)
            sx, sy = (vx1 - vx0) / view.width, (vy1 - vy0) / view.height
            for label, box, score in self._gdino_boxes(gdino, view, query):
                b = np.array([vx0 + box[0] * sx, vy0 + box[1] * sy, vx0 + box[2] * sx, vy0 + box[3] * sy])
                # a box cut by a tile's inner edge is part of a larger object: the whole view or a neighbour has it
                if (vx0, vy0, vx1, vy1) != (0, 0, w, h) and (
                        (vx0 > 0 and b[0] <= vx0 + 2 * sx) or (vy0 > 0 and b[1] <= vy0 + 2 * sy)
                        or (vx1 < w and b[2] >= vx1 - 2 * sx) or (vy1 < h and b[3] >= vy1 - 2 * sy)):
                    continue
                labels.append(label), boxes.append(b), scores.append(score)
        if not boxes:
            return []
        keep = nms(torch.tensor(np.array(boxes), dtype=torch.float32), torch.tensor(scores, dtype=torch.float32),
                   GDINO_MERGE_IOU).tolist()
        return [(labels[i], boxes[i]) for i in keep]

    @staticmethod
    def _gdino_boxes(gdino, small: Image.Image, query: str) -> list:
        """[(label, xyxy box in `small` px, score)], one per instance, best score first."""
        import torch
        from torchvision.ops import nms
        processor, model = gdino
        p = next(model.parameters())
        inputs = processor(images=small, text=query.lower().rstrip(".") + ".", return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**_to(inputs, p))
        res = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=GDINO_BOX_THRESHOLD, text_threshold=GDINO_TEXT_THRESHOLD,
            target_sizes=[small.size[::-1]])[0]
        boxes, scores = res["boxes"].float().cpu(), res["scores"].float().cpu()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, small.width)  # boxes may poke out of the frame
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, small.height)
        labels = res.get("text_labels", res.get("labels"))
        # the text span can carry the "." separator and tokenizer markers such as [SEP]
        return [(sanitize_text(re.sub(r"\[[A-Z]+\]|\.", " ", str(labels[i])), 60) or query, boxes[i].numpy(),
                 float(scores[i])) for i in nms(boxes, scores, GDINO_MERGE_IOU).tolist()]

    @staticmethod
    def _sam_outline(img: Image.Image, box, points=()) -> np.ndarray:
        """SAM 2 mask for one box (+ interior points), on a padded full-resolution crop. uint8 0/255."""
        import torch
        processor, model = precise.SAM2.get()
        w, h = img.size
        x0, y0, x1, y1 = (float(v) for v in box)
        pad = SAM_PAD * max(x1 - x0, y1 - y0)
        cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        cx1, cy1 = min(w, int(x1 + pad) + 1), min(h, int(y1 + pad) + 1)
        kw = {"input_boxes": [[[x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]]]}
        if points:
            kw["input_points"] = [[[[px - cx0, py - cy0] for px, py in points]]]
            kw["input_labels"] = [[[1] * len(points)]]
        inputs = processor(images=img.crop((cx0, cy0, cx1, cy1)), return_tensors="pt", **kw)
        with torch.inference_mode():
            out = model(**_to(inputs, next(model.parameters())), multimask_output=False)
        m = processor.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"])[0]
        full = np.zeros((h, w), np.uint8)
        full[cy0:cy1, cx0:cx1] = np.asarray(m).reshape(m.shape[-2:]).astype(np.uint8) * 255
        return full

    def _sam_refine(self, img: Image.Image, coarse: np.ndarray) -> Optional[np.ndarray]:
        """SAM 2 outline of a VOC region from its bounding box and interior points. None when SAM is
        missing or missed the region (covers < SAM_MIN_COVERAGE of it): the caller then uses refine_region."""
        import cv2
        if precise.SAM2.get() is None:
            return None
        x, y, bw, bh = cv2.boundingRect(coarse)
        mask = self._sam_outline(img, (x, y, x + bw, y + bh), interior_points(coarse))
        if np.count_nonzero(mask & coarse) < SAM_MIN_COVERAGE * np.count_nonzero(coarse):
            return None
        return mask

    @staticmethod
    def _matte(img: Image.Image, mask: np.ndarray) -> np.ndarray:
        """mask | (ViTMatte alpha >= MATTE_KEEP) on a capped crop; the trimap's unknown band is ~2 % of the box."""
        import cv2
        import torch
        processor, model = precise.VITMATTE.get()
        h, w = mask.shape
        x, y, bw, bh = cv2.boundingRect(mask)
        band = max(3, round(0.02 * max(bw, bh)))
        x0, y0 = max(0, x - 2 * band), max(0, y - 2 * band)
        x1, y1 = min(w, x + bw + 2 * band), min(h, y + bh + 2 * band)
        cw, ch = x1 - x0, y1 - y0
        scale = min(1.0, MATTE_MAX_DIM / max(cw, ch))
        sw, sh = max(1, round(cw * scale)), max(1, round(ch * scale))
        crop = img.crop((x0, y0, x1, y1)).resize((sw, sh), Image.Resampling.BILINEAR)
        m = cv2.resize(mask[y0:y1, x0:x1], (sw, sh), interpolation=cv2.INTER_NEAREST)
        b = max(3, round(band * scale))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b + 1, 2 * b + 1))
        tri = np.where(cv2.erode(m, k) > 0, 255, np.where(cv2.dilate(m, k) > 0, 128, 0)).astype(np.uint8)
        inputs = processor(images=crop, trimaps=Image.fromarray(tri), return_tensors="pt")
        p = next(model.parameters())
        with torch.inference_mode():
            alpha = model(pixel_values=inputs["pixel_values"].to(p.device, p.dtype)).alphas
        alpha = alpha.float().cpu().numpy().reshape(alpha.shape[-2:])[:sh, :sw].copy()  # the processor pads to /32
        alpha[tri == 0] = 0  # only the unknown band may grow
        keep = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_LINEAR) >= MATTE_KEEP
        out = mask.copy()
        out[y0:y1, x0:x1] |= keep.astype(np.uint8) * 255
        if p.device.type == "cuda":
            torch.cuda.empty_cache()  # hand the matte's activations back to the inpainting models
        return out

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
                         "Returns [{id, label, area_share, position}], position is left/centre/right x top/middle/bottom; a person region may list held items in \"with\". "
                         "Without a query only VOC classes (person, car, dog, ...) are found."),
        (select_regions, "Select the given region ids (from detect_regions) on the canvas. Nothing is "
                         "erased: the user adjusts the selection and clicks Erase."),
    ]
    return [StructuredTool.from_function(func=f, name=f.__name__, description=d) for f, d in specs]
