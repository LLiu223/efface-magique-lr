"""
recovery.py
Efface Magique LR - Crash-recovery autosave for unsynced modification layers.

Qt-free so it can be unit-tested. Stores, per photo, a manifest plus one mask PNG and one
RGB patch PNG per layer, both cropped to the padded mask bbox. Never the full image.
"""

import hashlib
import json
import os
import shutil
import uuid
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

FORMAT_VERSION = 1
MANIFEST = "manifest.json"
# Same padding as MainWindow._recompute_composite_stack
PAD = 25


def recovery_dir(root: str, key: str) -> str:
    """Folder for one photo; the key is hashed so photo paths never appear in folder names."""
    return os.path.join(root, hashlib.sha1(key.encode("utf-8")).hexdigest())


def padded_bbox(mask: Image.Image, pad: int = PAD) -> Optional[Tuple[int, int, int, int]]:
    """(x1, y1, x2, y2) of mask > 10 grown by pad and clipped to the image, or None if empty."""
    arr = np.asarray(mask.convert("L") if mask.mode != "L" else mask) > 10
    ys = np.flatnonzero(arr.any(axis=1))
    xs = np.flatnonzero(arr.any(axis=0))
    if not len(xs):
        return None
    w, h = mask.size
    return (max(0, int(xs[0]) - pad), max(0, int(ys[0]) - pad),
            min(w, int(xs[-1]) + 1 + pad), min(h, int(ys[-1]) + 1 + pad))


def save(dir_path: str, key: str, base_size: Tuple[int, int], layers) -> None:
    """Write mask/patch PNGs then atomically replace the manifest. Raises OSError on failure."""
    os.makedirs(dir_path, exist_ok=True)
    token = uuid.uuid4().hex[:8]  # fresh file names per save: the old manifest stays consistent until replaced
    entries = []
    for i, layer in enumerate(layers):
        box = padded_bbox(layer.mask)
        if box is None:
            continue  # an empty mask has no effect on the composite
        mask_name, patch_name = f"{token}_{i}_mask.png", f"{token}_{i}_patch.png"
        layer.mask.convert("L").crop(box).save(os.path.join(dir_path, mask_name), compress_level=1)
        layer.get_active_image().convert("RGB").crop(box).save(os.path.join(dir_path, patch_name), compress_level=1)
        mode = layer.engine_mode
        entries.append({
            "id": str(layer.layer_id),
            "name": str(layer.name),
            "visible": bool(layer.visible),
            "engine_mode": getattr(mode, "value", mode),
            "prompt": layer.prompt,
            "seed": layer.seed,
            "detect_subject": bool(layer.detect_subject),
            "enable_grain": bool(layer.enable_grain),
            "box": list(box),
            "mask": mask_name,
            "patch": patch_name,
        })
    manifest = {"version": FORMAT_VERSION, "key": key, "base_size": list(base_size), "layers": entries}
    tmp = os.path.join(dir_path, MANIFEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(dir_path, MANIFEST))
    keep = {MANIFEST} | {e["mask"] for e in entries} | {e["patch"] for e in entries}
    for name in os.listdir(dir_path):
        if name not in keep:
            try:
                os.remove(os.path.join(dir_path, name))
            except OSError:
                pass


def load(dir_path: str, key: str, base_size: Tuple[int, int]) -> Optional[List[dict]]:
    """
    Return validated layer dicts (manifest fields plus "mask" and "patch" PIL images), or None
    if there is no manifest. Raises ValueError for a corrupt or mismatched manifest.
    """
    path = os.path.join(dir_path, MANIFEST)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"unreadable manifest: {e}") from None
    if not isinstance(m, dict) or m.get("version") != FORMAT_VERSION or m.get("key") != key:
        raise ValueError("wrong manifest version or key")
    w, h = base_size
    if m.get("base_size") != [w, h]:
        raise ValueError(f"base size {m.get('base_size')} does not match {[w, h]}")
    if not isinstance(m.get("layers"), list):
        raise ValueError("layers is not a list")
    out = []
    for e in m["layers"]:
        if not isinstance(e, dict):
            raise ValueError("layer entry is not an object")
        box = e.get("box")
        if not (isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)):
            raise ValueError("bad crop box")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("crop box outside image")
        for field, typ in (("id", str), ("name", str), ("visible", bool), ("engine_mode", str),
                           ("detect_subject", bool), ("enable_grain", bool), ("mask", str), ("patch", str)):
            if not isinstance(e.get(field), typ):
                raise ValueError(f"bad field {field}")
        if e.get("prompt") is not None and not isinstance(e["prompt"], str):
            raise ValueError("bad field prompt")
        if e.get("seed") is not None and type(e["seed"]) is not int:
            raise ValueError("bad field seed")
        images = {}
        for field, mode in (("mask", "L"), ("patch", "RGB")):
            name = e[field]
            if os.path.basename(name) != name or name in ("", ".", ".."):
                raise ValueError(f"bad {field} file name")
            try:
                with Image.open(os.path.join(dir_path, name)) as im:
                    # Check the header size before decoding so an oversized PNG is never loaded
                    if im.size != (x2 - x1, y2 - y1):
                        raise ValueError(f"{field} size does not match crop box")
                    im.load()
                    images[field] = im.convert(mode)
            except ValueError:
                raise
            except Exception as err:
                raise ValueError(f"unreadable {field} PNG: {err}") from None
        out.append({**e, "box": (x1, y1, x2, y2), **images})
    return out


def clear(dir_path: str) -> None:
    """Delete a photo's recovery folder (missing is fine)."""
    shutil.rmtree(dir_path, ignore_errors=True)
