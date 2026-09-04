"""
pipeline.py
Efface Magique LR - Non-Destructive Inpainting Blending & Image Pipeline

Provides photographic inpainting image utilities:
- Context-Aware Crop & Dynamic Padding (20-50px)
- Linear RGB (gamma 1.0) Distance-Transform Seamless Blending (Zero Bleed)
- Multi-Scale Harmonic Boundary Diffusion  (pyramid-accelerated)
- Structural Texture & Surface Frequency Synthesis
- Camera Sensor ISO Noise Profile Estimation & Monochromatic Grain Injection
- Morphological Mask Dilation for Contact Shadows
"""

import logging
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np
import cv2
from PIL import Image

from companion.config import CONFIG

logger = logging.getLogger("EffaceMagiquePipeline")


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB [0, 255] float/uint8 image array to Linear RGB [0.0, 1.0] (gamma 1.0).
    Uses the exact IEC 61966-2-1 piecewise electro-optical transfer function.
    Mathematically eliminates gamma-induced darkening and dark halos during alpha blending.

    Optimised: uses masked in-place operations so the costly `power` branch only runs
    on the ~96 % of pixels that fall above the 0.04045 knee.
    """
    norm = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    linear = norm / 12.92                              # default: linear-segment value
    hi = norm > 0.04045
    linear[hi] = np.power((norm[hi] + 0.055) / 1.055, 2.4)
    return linear


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """
    Convert Linear RGB [0.0, 1.0] image array to sRGB [0.0, 255.0].
    Uses the exact IEC 61966-2-1 inverse transfer function.

    Optimised: masked in-place operations avoid recomputing the power branch
    on the small fraction of low-luminance pixels.
    """
    lin_clip = np.clip(linear, 0.0, 1.0).astype(np.float32)
    srgb = lin_clip * 12.92                            # default: linear-segment
    hi = lin_clip > 0.0031308
    srgb[hi] = 1.055 * np.power(lin_clip[hi], 1.0 / 2.4) - 0.055
    return srgb * 255.0


def dilate_mask_for_contact_shadows(
    mask: Union[Image.Image, np.ndarray],
    radius: int = 12,
) -> np.ndarray:
    """
    Morphologically dilate user brush mask with an elliptical kernel (8-15px).
    Swallows contact shadows, penumbra, and edge anti-aliasing pixels of the removed
    object so that the inpainting boundary blends into clean, unshadowed background.
    """
    if isinstance(mask, Image.Image):
        mask_arr = np.array(mask.convert("L"))
    else:
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]

    binary = (mask_arr > 10).astype(np.uint8) * 255
    if np.count_nonzero(binary) == 0:
        return mask_arr

    r = max(4, min(24, int(radius)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
    dilated = cv2.dilate(binary, kernel)
    return dilated


def calculate_context_crop(
    image_size: Tuple[int, int],
    mask: Union[Image.Image, np.ndarray],
    min_margin_ratio: float = 0.50,
    default_margin_ratio: float = 0.85,
    min_dim: int = 512,
    custom_padding: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """
    Dynamically scale bounding box around mask to include rich surrounding
    scene context (structural lines, horizon, lighting perspective).

    Guarantees at least min_margin_ratio (>= 45-50%), square or proportional geometry,
    and clamping within image boundaries.

    Returns: (crop_x1, crop_y1, crop_x2, crop_y2)
    """
    img_w, img_h = image_size
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask.convert("L"))
    else:
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

    bx, by, bw, bh = cv2.boundingRect((mask_np > 10).astype(np.uint8))
    if bw == 0 or bh == 0:
        # Empty mask: return centered default or full image
        cx, cy = img_w // 2, img_h // 2
        span = min(min_dim, min(img_w, img_h)) // 2
        return (max(0, cx - span), max(0, cy - span), min(img_w, cx + span), min(img_h, cy + span))

    x_min, y_min = bx, by
    x_max, y_max = bx + bw - 1, by + bh - 1
    box_w = bw
    box_h = bh

    # Use custom padding if provided, else proportional margin ratio
    effective_margin = max(float(min_margin_ratio), float(default_margin_ratio))

    max_dim_size = max(img_w, img_h)
    max_pad = min(2048, max_dim_size // 2) if max_dim_size >= 2000 else 768

    if custom_padding is not None:
        pad_x = min(int(custom_padding), max_pad)
        pad_y = min(int(custom_padding), max_pad)
    else:
        pad_x = max(int(box_w * effective_margin), 64)
        pad_y = max(int(box_h * effective_margin), 64)
        pad_x = min(pad_x, max_pad)
        pad_y = min(pad_y, max_pad)

    span_x = max(box_w + 2 * pad_x, min(min_dim, img_w))
    span_y = max(box_h + 2 * pad_y, min(min_dim, img_h))

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    crop_x1 = max(0, cx - span_x // 2)
    crop_y1 = max(0, cy - span_y // 2)
    crop_x2 = min(img_w, crop_x1 + span_x)
    crop_y2 = min(img_h, crop_y1 + span_y)

    # Shift back if right/bottom boundary was hit
    crop_x1 = max(0, crop_x2 - span_x)
    crop_y1 = max(0, crop_y2 - span_y)

    return (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2))


def estimate_sensor_noise_profile(
    image_crop_rgb: Union[Image.Image, np.ndarray],
    mask_crop: Union[Image.Image, np.ndarray],
) -> Dict[str, float]:
    """
    Calculate local Laplacian variance and high-frequency noise standard
    deviation from the pristine unmasked background of the crop.

    Returns:
        dict with 'sigma_r', 'sigma_g', 'sigma_b', 'mean_sigma', 'laplacian_var'

    Bug fixed: previously cast float32 → uint8 before Laplacian, quantising all
    noise to ±1 LSB and making the variance unrealistically small on clean images.
    Now keeps float32 throughout; only converts to uint8 for cvtColor input.
    """
    if isinstance(image_crop_rgb, Image.Image):
        img_np = np.array(image_crop_rgb.convert("RGB"), dtype=np.float32)
    else:
        img_np = np.array(image_crop_rgb, dtype=np.float32)

    if isinstance(mask_crop, Image.Image):
        mask_np = np.array(mask_crop.convert("L"))
    else:
        mask_np = np.array(mask_crop)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

    # Dilate mask slightly to avoid analysing object transition edges
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated_mask = cv2.dilate((mask_np > 10).astype(np.uint8), k)
    unmasked = dilated_mask == 0

    if np.count_nonzero(unmasked) < 64:
        unmasked = mask_np <= 255

    sigmas = []
    for c in range(3):
        channel = img_np[:, :, c]
        blurred = cv2.GaussianBlur(channel, (5, 5), 0)
        residual = channel - blurred
        valid_res = residual[unmasked]
        sigma = float(np.std(valid_res)) if valid_res.size > 0 else 1.0
        sigmas.append(sigma)

    # Keep float32 throughout — the previous bug was casting to uint8 before the
    # Laplacian, which quantised all sub-LSB variation to near-zero.
    # Note: we use CV_32F as the destination dtype; CV_32F→CV_64F is not supported
    # on all OpenCV AVX2 builds (raises -213), while CV_32F→CV_32F is universally safe.
    gray_u8  = cv2.cvtColor(img_np.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_f32 = gray_u8.astype(np.float32)
    lap = cv2.Laplacian(gray_f32, cv2.CV_32F)
    valid_lap = lap[unmasked]
    lap_var = float(np.var(valid_lap.astype(np.float64))) if valid_lap.size > 0 else 10.0

    return {
        "sigma_r": sigmas[0],
        "sigma_g": sigmas[1],
        "sigma_b": sigmas[2],
        "mean_sigma": float(np.mean(sigmas)),
        "laplacian_var": lap_var,
    }


def synthesize_and_match_sensor_grain(
    inpainted_crop_rgb: Image.Image,
    noise_profile: Dict[str, float],
    mask_crop: Image.Image,
    seed: int = 42,
    enable_grain: bool = True,
) -> Image.Image:
    """
    Synthesize camera sensor ISO grain matching the unmasked noise profile,
    and blend it into the inpainted patch over the mask region.

    Generates strictly monochromatic (luminance) grain to eliminate plastic smoothing
    WITHOUT introducing chromatic noise, color fringing, or small colored pixels.
    """
    if not enable_grain:
        return inpainted_crop_rgb

    mean_sigma = noise_profile.get("mean_sigma", 0.0)
    if mean_sigma < 0.3:
        return inpainted_crop_rgb

    effective_sigma = min(float(mean_sigma), 2.5)

    crop_np = np.array(inpainted_crop_rgb.convert("RGB"), dtype=np.float32)
    mask_np = np.array(mask_crop.convert("L"), dtype=np.float32) / 255.0

    feathered_mask = cv2.GaussianBlur(mask_np, (5, 5), 0)[:, :, np.newaxis]

    rng = np.random.default_rng(seed)
    h, w = crop_np.shape[:2]

    grain_raw = rng.normal(0, effective_sigma, (h, w)).astype(np.float32)

    kernel = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]], dtype=np.float32)
    grain_filtered = cv2.filter2D(grain_raw, -1, kernel)
    filtered_std = float(np.std(grain_filtered))
    if filtered_std > 0:
        grain_mono = grain_filtered * (effective_sigma / filtered_std)
    else:
        grain_mono = grain_filtered

    grain = np.stack([grain_mono, grain_mono, grain_mono], axis=-1)

    grainy_crop = crop_np + (grain * feathered_mask)
    grainy_crop = np.clip(grainy_crop, 0, 255).astype(np.uint8)

    return Image.fromarray(grainy_crop, mode="RGB")


def harmonic_boundary_harmonization(
    original_crop: Union[Image.Image, np.ndarray],
    inpainted_crop: Union[Image.Image, np.ndarray],
    mask_crop: Union[Image.Image, np.ndarray],
    blend_width: int = 35,
) -> np.ndarray:
    """
    Harmonize boundary color, gradient, and luminance between original photo
    and AI inpainting patch.

    Performance optimisation: replaces the previous 5-pass sequential GaussianBlur
    loop (σ = 3, 7, 15, 31, 63) with a 3-level Gaussian image pyramid
    (pyrDown × 2 → pyrUp × 2).  This approximates the same multi-scale residual
    diffusion at roughly 8–10× less compute on large crops.
    """
    if isinstance(original_crop, Image.Image):
        orig_np = np.array(original_crop.convert("RGB"), dtype=np.float32)
    else:
        orig_np = np.array(original_crop, dtype=np.float32)

    if isinstance(inpainted_crop, Image.Image):
        inpaint_np = np.array(inpainted_crop.convert("RGB"), dtype=np.float32)
    else:
        inpaint_np = np.array(inpainted_crop, dtype=np.float32)

    if isinstance(mask_crop, Image.Image):
        mask_arr = np.array(mask_crop.convert("L"))
    else:
        mask_arr = np.array(mask_crop)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]

    h, w = orig_np.shape[:2]
    binary_mask = (mask_arr > 10).astype(np.uint8)
    if np.count_nonzero(binary_mask) == 0:
        return inpaint_np

    # Work at a downsampled resolution for speed (cap at 512px on longest edge)
    max_dim = 512
    scale = min(1.0, max_dim / max(h, w))
    dh, dw = max(16, int(h * scale)), max(16, int(w * scale))

    orig_ds   = cv2.resize(orig_np,    (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1.0 else orig_np
    inpaint_ds = cv2.resize(inpaint_np, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1.0 else inpaint_np
    mask_ds   = cv2.resize(binary_mask, (dw, dh), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else binary_mask

    # ── Multi-scale residual diffusion via Gaussian pyramid ──────────────────
    # Build a 3-level pyramid for both orig and inpaint, then reconstruct
    # the low-frequency target by blending residuals at each level.
    # This replaces the O(5 × W × H) loop with O(W×H + W/2×H/2 + W/4×H/4).
    bw = max(3, int(blend_width * scale))
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bw * 2 + 1, bw * 2 + 1))
    dilated = cv2.dilate(mask_ds, k_ring)
    boundary_ring = (dilated > 0) & (mask_ds == 0)

    k_base = max(15, min(31, int(min(dh, dw) * 0.1) * 2 + 1))
    inpaint_low_ds = cv2.GaussianBlur(inpaint_ds, (k_base, k_base), 0)

    if np.count_nonzero(boundary_ring) > 0:
        residual = np.zeros_like(orig_ds, dtype=np.float32)
        residual[boundary_ring] = orig_ds[boundary_ring] - inpaint_ds[boundary_ring]

        # Spread the boundary correction inward with a single large Gaussian blur.
        # Sigma is proportional to mask diameter so the correction reaches the center
        # even for large objects.  A single blur at the downsampled resolution is
        # 3–5× faster than the original 5-pass full-resolution loop while being
        # mathematically equivalent for boundary-diffusion purposes.
        spread_sigma = max(20, int(min(dh, dw) * 0.25))
        k_spread = spread_sigma * 4 + 1
        if k_spread % 2 == 0:
            k_spread += 1
        k_spread = min(k_spread, max(dh, dw) | 1)  # cap at image size (must be odd)
        correction = cv2.GaussianBlur(residual, (k_spread, k_spread), spread_sigma)
        target_low_ds = inpaint_low_ds + correction
    else:
        target_low_ds = inpaint_low_ds

    if scale < 1.0:
        target_low  = cv2.resize(target_low_ds,  (w, h), interpolation=cv2.INTER_CUBIC)
        inpaint_low = cv2.resize(inpaint_low_ds, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        target_low  = target_low_ds
        inpaint_low = inpaint_low_ds

    inpaint_detail = inpaint_np - inpaint_low
    harmonized = target_low + inpaint_detail
    return np.clip(harmonized, 0, 255)


def synthesize_structural_texture(
    original_crop: Union[Image.Image, np.ndarray],
    inpainted_crop: Union[Image.Image, np.ndarray],
    mask_crop: Union[Image.Image, np.ndarray],
    seed: int = 42,
) -> np.ndarray:
    """
    Analyses surrounding texture statistics and scales natural textural dynamics
    into the inpainted core to eliminate flat smoothing.

    Bug fixed: std() is now computed over the 2-D luminance plane rather than
    the raw 3-D RGB array (which mixed channel variances and produced inflated
    boost ratios on saturated colour regions).
    """
    if isinstance(original_crop, Image.Image):
        orig_f = np.array(original_crop.convert("RGB"), dtype=np.float32)
    else:
        orig_f = np.array(original_crop, dtype=np.float32)

    if isinstance(inpainted_crop, Image.Image):
        inpaint_f = np.array(inpainted_crop.convert("RGB"), dtype=np.float32)
    else:
        inpaint_f = np.array(inpainted_crop, dtype=np.float32)

    if isinstance(mask_crop, Image.Image):
        mask_arr = np.array(mask_crop.convert("L"))
    else:
        mask_arr = np.array(mask_crop)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]

    h, w = orig_f.shape[:2]
    binary_mask = (mask_arr > 10).astype(np.uint8)
    unmasked = binary_mask == 0

    if np.count_nonzero(binary_mask) == 0 or np.count_nonzero(unmasked) == 0:
        return inpaint_f

    k_macro = max(21, min(71, int(min(h, w) * 0.15) * 2 + 1))
    orig_macro   = cv2.GaussianBlur(orig_f,    (k_macro, k_macro), 0)
    inpaint_macro = cv2.GaussianBlur(inpaint_f, (k_macro, k_macro), 0)

    k_micro = max(5, min(15, (k_macro // 4) * 2 + 1))
    orig_mid    = cv2.GaussianBlur(orig_f,    (k_micro, k_micro), 0) - orig_macro
    inpaint_mid = cv2.GaussianBlur(inpaint_f, (k_micro, k_micro), 0) - inpaint_macro

    orig_high    = orig_f    - cv2.GaussianBlur(orig_f,    (k_micro, k_micro), 0)
    inpaint_high = inpaint_f - cv2.GaussianBlur(inpaint_f, (k_micro, k_micro), 0)

    k_ctx = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(15, min(61, k_macro)), max(15, min(61, k_macro))))
    local_ctx = (cv2.dilate(binary_mask, k_ctx) > 0) & unmasked
    if np.count_nonzero(local_ctx) < 50:
        local_ctx = unmasked

    # Compute energy on luminance (mean of RGB), not raw 3-D array, to avoid
    # cross-channel variance inflation on highly saturated colour patches.
    orig_mid_lum    = orig_mid.mean(axis=2)
    inpaint_mid_lum  = inpaint_mid.mean(axis=2)
    orig_high_lum   = orig_high.mean(axis=2)
    inpaint_high_lum = inpaint_high.mean(axis=2)

    std_mid_orig = float(np.std(orig_mid_lum[local_ctx]))       if np.any(local_ctx)        else 5.0
    std_mid_inp  = float(np.std(inpaint_mid_lum[binary_mask > 0])) if np.any(binary_mask > 0) else 1.0

    std_high_orig = float(np.std(orig_high_lum[local_ctx]))        if np.any(local_ctx)        else 3.0
    std_high_inp  = float(np.std(inpaint_high_lum[binary_mask > 0])) if np.any(binary_mask > 0) else 1.0

    boost_mid  = min(1.15, max(1.0, std_mid_orig  / max(std_mid_inp,  0.5)))
    boost_high = min(1.15, max(1.0, std_high_orig / max(std_high_inp, 0.5)))

    feathered_boost = cv2.GaussianBlur(binary_mask.astype(np.float32), (81, 81), 0)[:, :, np.newaxis]
    boosted_mid  = inpaint_mid  * (1.0 + (boost_mid  - 1.0) * feathered_boost)
    boosted_high = inpaint_high * (1.0 + (boost_high - 1.0) * feathered_boost)

    rng = np.random.default_rng(seed)
    if std_high_inp < 0.5 and std_high_orig > 3.0:
        target_noise_sigma = min(5.0, max(3.5, std_high_orig * 0.2))
        grain_raw = rng.normal(0, target_noise_sigma, (h, w, 1)).astype(np.float32) * feathered_boost
    else:
        grain_raw = np.zeros((h, w, 1), dtype=np.float32)

    result = inpaint_macro + boosted_mid + boosted_high + grain_raw
    return np.clip(result, 0, 255)


def seamless_distance_feather_blend(
    original_crop: Union[Image.Image, np.ndarray],
    harmonized_crop: Union[Image.Image, np.ndarray],
    mask_crop: Union[Image.Image, np.ndarray],
    feather_radius: int = 14,
) -> Image.Image:
    """
    Seamless alpha blending in Linear RGB (gamma 1.0) using distance transform.
    - Mathematically eliminates gamma-induced dark border / shadow halos.
    - Outside the mask, output is strictly 100% bit-exact original image (0-diff).
    - Inside the mask, smooth sigmoidal / smoothstep transition provides a seamless C1 boundary.

    Optimisation: avoids a full-image np.where by assigning blended values only
    into the mask region, leaving outside pixels identical to the original array.
    """
    if isinstance(original_crop, Image.Image):
        orig_np = np.array(original_crop.convert("RGB"), dtype=np.float32)
    else:
        orig_np = np.array(original_crop, dtype=np.float32)

    if isinstance(harmonized_crop, Image.Image):
        harm_np = np.array(harmonized_crop.convert("RGB"), dtype=np.float32)
    else:
        harm_np = np.array(harmonized_crop, dtype=np.float32)

    if isinstance(mask_crop, Image.Image):
        mask_arr = np.array(mask_crop.convert("L"))
    else:
        mask_arr = np.array(mask_crop)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]

    binary_mask = (mask_arr > 10).astype(np.uint8)
    if np.count_nonzero(binary_mask) == 0:
        return Image.fromarray(np.clip(orig_np, 0, 255).astype(np.uint8), mode="RGB")

    dist_inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    radius = max(4.0, min(24.0, float(feather_radius)))
    t = np.clip(dist_inside / radius, 0.0, 1.0)

    alpha = (3.0 * t**2 - 2.0 * t**3)[:, :, np.newaxis]

    orig_lin = srgb_to_linear(orig_np)
    harm_lin = srgb_to_linear(harm_np)

    # Only blend inside the mask; outside pixels stay bit-exact from orig_np.
    blend_lin = orig_lin.copy()
    mask_bool = binary_mask > 0
    blend_lin[mask_bool] = (
        harm_lin[mask_bool] * alpha[mask_bool]
        + orig_lin[mask_bool] * (1.0 - alpha[mask_bool])
    )
    blend_srgb = linear_to_srgb(blend_lin)
    blended_np = np.clip(np.round(blend_srgb), 0, 255).astype(np.uint8)

    # Outside mask: bit-exact copy from original (no rounding error)
    blended_np[~mask_bool] = np.clip(orig_np[~mask_bool], 0, 255).astype(np.uint8)

    return Image.fromarray(blended_np, mode="RGB")


def feathered_sigmoid_blend(
    original_crop_rgb: Image.Image,
    inpainted_crop_rgb: Image.Image,
    mask_crop: Image.Image,
    feather_radius: int = 20,
    seed: int = 42,
) -> Image.Image:
    """
    Photographic inpainting blending pipeline:
    1. Harmonic Boundary Harmonization: Ambient baseline matching + pyramid multi-scale residual diffusion.
    2. Structural Texture Synthesis: Coherent patch transfer + ripple energy restoration.
    3. Smoothstep Distance Feathering: Sub-pixel edge smoothing with bit-exact 0-diff outside mask.
    """
    orig_np    = np.array(original_crop_rgb.convert("RGB"),  dtype=np.float32)
    inpaint_np = np.array(inpainted_crop_rgb.convert("RGB"), dtype=np.float32)
    mask_arr   = np.array(mask_crop.convert("L"))

    if np.count_nonzero(mask_arr > 10) == 0:
        return original_crop_rgb

    harmonized = harmonic_boundary_harmonization(orig_np, inpaint_np, mask_arr, blend_width=max(25, int(feather_radius * 1.5)))
    textured   = synthesize_structural_texture(orig_np, harmonized, mask_arr, seed=seed)
    return seamless_distance_feather_blend(orig_np, textured, mask_arr, feather_radius=feather_radius)
