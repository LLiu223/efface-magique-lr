"""
Reproducible erase-pipeline benchmark for Efface Magique LR.

Times each stage of InpaintingEngine.generate_variations on synthetic 12/24/60 MP
images with a fixed seed and a fixed mask, then the 8-bit and 16-bit TIFF saves,
and reports peak tracemalloc memory per size.

    python tools/benchmark.py --stub              # no LaMa weights needed
    python tools/benchmark.py                     # real LaMa (weights must be cached)
    python tools/benchmark.py --stub --sizes 12,24

The stages below repeat the orchestration glue of generate_variations (same calls,
same order, same arguments) so each one can be timed; the heavy lifting is done by
the real functions. run_stages() output is checked byte-for-byte against
generate_variations in tests/test_benchmark_smoke.py.

Gaps from the app path (all deliberate):
- num_variations=1 (the app asks for 3 in Generative AI mode; each extra variation
  repeats inpaint + blend + paste).
- detect_subject=False and enable_grain=False (subject detection needs its own model).
- The Qt worker thread, progress callbacks and the final free_device_memory() are not timed.
"""

import argparse
import math
import os
import platform
import sys
import tempfile
import time
import tracemalloc

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from companion.config import CONFIG  # noqa: E402
from companion.device import get_optimal_device, free_device_memory  # noqa: E402
from companion.model_engine import EngineMode, InpaintingEngine  # noqa: E402
from companion.pipeline import (  # noqa: E402
    calculate_context_crop,
    dilate_mask_for_contact_shadows,
    estimate_sensor_noise_profile,
    feathered_sigmoid_blend,
    seamless_distance_feather_blend,
    synthesize_and_match_sensor_grain,
)
from companion.utils.tiff16 import merge_into_16bit, write_tiff16  # noqa: E402

PRESETS = {12: (4240, 2832), 24: (6000, 4000), 60: (9504, 6336)}
SEED = 1234
MASK_W, MASK_H = 300, 200  # a typical brush stroke, same absolute size at every resolution


class StubLama:
    """STUB engine: cv2.inpaint stands in for the LaMa network. Timings are NOT LaMa timings."""

    def __call__(self, image, mask):
        out = cv2.inpaint(np.asarray(image.convert("RGB")), np.asarray(mask.convert("L")), 5, cv2.INPAINT_TELEA)
        return Image.fromarray(out)


def size_for(mp):
    if mp in PRESETS:
        return PRESETS[mp]
    w = int(round(math.sqrt(mp * 1e6 * 1.5)))  # 3:2 like the presets
    return w, int(round(w / 1.5))


def make_inputs(w, h):
    rng = np.random.default_rng(SEED)
    image = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
    mask = Image.new("L", (w, h), 0)
    cx, cy = int(w * 0.6), int(h * 0.4)
    ImageDraw.Draw(mask).ellipse((cx - MASK_W // 2, cy - MASK_H // 2, cx + MASK_W // 2, cy + MASK_H // 2), fill=255)
    return image, mask


def make_engine(stub, mode):
    engine = InpaintingEngine(device=get_optimal_device(), mode=mode)
    if stub:
        engine._lama_model = StubLama()
        engine._is_loaded = True
        return engine
    # Check the cached weights first so a missing model never triggers a download.
    weights = os.environ.get("LAMA_MODEL")
    if not weights:
        try:
            from simple_lama_inpainting.models.model import LAMA_MODEL_URL
            from simple_lama_inpainting.utils.util import get_cache_path_by_url
        except ImportError as e:
            sys.exit(f"ERROR: simple-lama-inpainting is not installed ({e}). Run with --stub to benchmark without it.")
        weights = get_cache_path_by_url(LAMA_MODEL_URL)
    if not os.path.exists(weights):
        sys.exit(f"ERROR: LaMa weights not found at {weights}. Run with --stub to benchmark without them.")
    engine.load_model()
    if engine._lama_model is None:
        sys.exit("ERROR: LaMa model failed to load (see warning above). Run with --stub to benchmark without it.")
    return engine


def run_stages(engine, image, mask, seed=SEED, margin_ratio=0.85, feather_radius=20):
    """Mirror of generate_variations(num_variations=1, detect_subject=False, enable_grain=False)."""
    t = {}
    is_fast_mode = engine.mode == EngineMode.FAST
    is_cpu = engine.device.type == "cpu"

    s = time.perf_counter()
    # generate_variations decodes the mask for its empty check, then again for dilation sizing
    mask_np = np.array(mask.convert("L"))
    assert np.count_nonzero(mask_np > 10) > 0
    effective_mask = mask
    effective_mask_l = effective_mask.convert("L") if effective_mask.mode != "L" else effective_mask
    mask_np_all = np.asarray(effective_mask_l)
    bx, by, bw, bh = cv2.boundingRect((mask_np_all > 10).astype(np.uint8))
    box_dim_all = max(bw, bh) if (bw > 0 and bh > 0) else 100
    if box_dim_all < 60:
        dilate_r = max(8, min(14, int(box_dim_all * 0.18) + 6))
    else:
        dilate_r = max(8, min(16, int(box_dim_all * 0.05) + 8))
    dilated_mask_np = dilate_mask_for_contact_shadows(effective_mask, radius=dilate_r)
    dilated_mask = Image.fromarray(dilated_mask_np.astype(np.uint8))
    t["dilate"] = time.perf_counter() - s

    s = time.perf_counter()
    effective_min_dim = 256 if (is_cpu or is_fast_mode) else 512
    crop_x1, crop_y1, crop_x2, crop_y2 = calculate_context_crop(
        image.size,
        dilated_mask,
        min_margin_ratio=0.35 if (is_cpu or is_fast_mode) else 0.50,
        default_margin_ratio=0.50 if (is_cpu or is_fast_mode) else margin_ratio,
        min_dim=effective_min_dim,
        custom_padding=CONFIG.DEFAULT_CONTEXT_PADDING if (is_fast_mode or is_cpu) else None,
    )
    crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1
    cropped_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2)).convert("RGB")
    cropped_mask = dilated_mask.crop((crop_x1, crop_y1, crop_x2, crop_y2)).convert("L")
    t["crop"] = time.perf_counter() - s

    s = time.perf_counter()
    noise_profile = estimate_sensor_noise_profile(cropped_img, cropped_mask)
    crop_mask_l = cropped_mask.convert("L") if cropped_mask.mode != "L" else cropped_mask
    cbx, cby, cbw, cbh = cv2.boundingRect((np.asarray(crop_mask_l) > 10).astype(np.uint8))
    box_dim = max(cbw, cbh) if (cbw > 0 and cbh > 0) else 100
    if is_fast_mode or box_dim < 60:
        effective_feather = max(4, min(8, int(box_dim * 0.2)))
    else:
        effective_feather = max(10, min(16, int(feather_radius)))
    t["noise_profile"] = time.perf_counter() - s

    s = time.perf_counter()
    base_rgb = image if image.mode == "RGB" else image.convert("RGB")
    t["convert_base"] = time.perf_counter() - s

    s = time.perf_counter()
    if is_fast_mode:
        raw_patch = engine._run_lama_neural_inference(cropped_img, cropped_mask)
    else:
        raw_patch = engine._run_variation_patch(cropped_img, cropped_mask, seed=seed, prompt=None)
    if raw_patch.size != (crop_w, crop_h):
        raw_patch = raw_patch.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
    if engine.device.type == "cuda":
        torch.cuda.synchronize()
    t["inpaint"] = time.perf_counter() - s

    s = time.perf_counter()
    if is_fast_mode:
        blended_crop = seamless_distance_feather_blend(cropped_img, raw_patch, cropped_mask, feather_radius=effective_feather)
    else:
        grain_matched_patch = synthesize_and_match_sensor_grain(
            raw_patch, noise_profile, cropped_mask, seed=seed, enable_grain=False)
        blended_crop = feathered_sigmoid_blend(
            cropped_img, grain_matched_patch, cropped_mask, feather_radius=effective_feather, seed=seed)
    t["blend"] = time.perf_counter() - s

    s = time.perf_counter()
    result_img = base_rgb.copy()
    result_img.paste(blended_crop, (crop_x1, crop_y1))
    t["paste"] = time.perf_counter() - s
    return result_img, t


def save_stages(image, result_img, base16, out_dir):
    """Timed saves as MainWindow does them: Pillow 8-bit TIFF, and merge_into_16bit + write_tiff16."""
    t = {}
    p8, p16 = os.path.join(out_dir, "out8.tif"), os.path.join(out_dir, "out16.tif")
    try:
        s = time.perf_counter()
        result_img.save(p8, compression="tiff_deflate")
        t["save_tiff8"] = time.perf_counter() - s

        s = time.perf_counter()
        merged = merge_into_16bit(base16, image, result_img)
        t["merge16"] = time.perf_counter() - s

        s = time.perf_counter()
        write_tiff16(p16, merged, None, None)
        t["save_tiff16"] = time.perf_counter() - s
        del merged
    finally:
        for p in (p8, p16):
            if os.path.exists(p):
                os.remove(p)
    return t


def machine_description(device):
    return [
        f"CPU: {platform.processor() or 'unknown'}",
        f"OS: {platform.platform()}",
        f"Python: {platform.python_version()}",
        f"numpy {np.__version__}, opencv {cv2.__version__}, torch {torch.__version__}, Pillow {Image.__version__}",
        f"Device: {device.type}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stub", action="store_true", help="use the cv2.inpaint STUB engine instead of LaMa (no weights)")
    ap.add_argument("--sizes", default="12,24,60", help="comma-separated megapixel sizes (default 12,24,60)")
    ap.add_argument("--mode", choices=[m.value for m in EngineMode], default=EngineMode.FIREFLY.value,
                    help="engine mode (default firefly, the app default)")
    args = ap.parse_args(argv)
    sizes = [float(x) if "." in x else int(x) for x in args.sizes.split(",")]

    engine = make_engine(args.stub, EngineMode(args.mode))
    label = "STUB" if args.stub else "LaMa"
    print(f"# Efface Magique LR erase benchmark - {'STUB ENGINE (cv2.inpaint, not LaMa)' if args.stub else 'real LaMa engine'}")
    for line in machine_description(engine.device):
        print(f"- {line}")
    print(f"- Mode: {args.mode}, 1 variation, mask {MASK_W}x{MASK_H} ellipse, seed {SEED}")
    print()
    print("| engine | size | stage | ms |")
    print("|---|---|---|---:|")

    peaks = []
    for mp in sizes:  # sequential on purpose: a 60 MP run alone can need over 1 GB
        w, h = size_for(mp)
        image, mask = make_inputs(w, h)
        base16 = np.asarray(image).astype(np.uint16) * 257
        tag = f"{mp} MP ({w}x{h})"
        with tempfile.TemporaryDirectory(prefix="effmag_bench_") as out_dir:
            tracemalloc.start()
            try:
                result_img, t = run_stages(engine, image, mask)
                t.update(save_stages(image, result_img, base16, out_dir))
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        free_device_memory(engine.device)
        t["total"] = sum(t.values())
        for stage, sec in t.items():
            print(f"| {label} | {tag} | {stage} | {sec * 1000:.1f} |")
        peaks.append((tag, peak))
        del image, mask, base16, result_img

    print()
    print("| engine | size | peak tracemalloc MiB |")
    print("|---|---|---:|")
    for tag, peak in peaks:
        print(f"| {label} | {tag} | {peak / 2**20:.1f} |")
    print()
    print("tracemalloc sees Python/numpy allocations only, not Pillow, OpenCV or torch native buffers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
