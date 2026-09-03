"""
companion.utils.subject_detector
Intelligent Subject Detection and Object-Aware Masking.

Provides:
- extract_subject_in_zone: Refines a loose user brush stroke ("coloring zone")
  so that ONLY the subject inside the zone is masked, leaving surrounding
  background pixels within the colored zone 100% untouched.
- Hybrid Neural Segmentation (PyTorch LRASPP MobileNetV3) + OpenCV GrabCut.
"""

import logging
from typing import Optional, Union, Tuple
import numpy as np
import cv2
from PIL import Image
import torch

logger = logging.getLogger("EffaceMagiqueSubjectDetector")

# Global cached neural segmentation model
_SEG_MODEL = None
_SEG_DEVICE = None


def get_segmentation_model(device: Optional[torch.device] = None) -> Optional[torch.nn.Module]:
    """Load and cache lightweight LRASPP MobileNetV3 segmentation model."""
    global _SEG_MODEL, _SEG_DEVICE
    if _SEG_MODEL is not None and (_SEG_DEVICE == device or device is None):
        return _SEG_MODEL

    try:
        import torchvision.models.segmentation as seg
        weights = seg.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        model = seg.lraspp_mobilenet_v3_large(weights=weights)
        target_device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model = model.to(target_device)
        model.eval()
        _SEG_MODEL = model
        _SEG_DEVICE = target_device
        logger.info(f"Subject segmentation model loaded on {target_device}.")
        return _SEG_MODEL
    except Exception as e:
        logger.warning(f"Could not load neural segmentation model: {e}. Falling back to GrabCut.")
        return None


def run_neural_subject_detection(
    image_rgb: np.ndarray,
    mask_zone: np.ndarray,
    device: Optional[torch.device] = None,
) -> Optional[np.ndarray]:
    """
    Run neural semantic segmentation to detect subject (person, animal, vehicle, etc.)
    overlapping the user's brushed zone.
    
    Returns binary mask (uint8 0 or 255) of subject within zone, or None if no subject detected.
    """
    model = get_segmentation_model(device)
    if model is None:
        return None

    try:
        import torchvision.transforms.functional as TF

        h, w = image_rgb.shape[:2]
        # Resize to max dimension 512 for fast real-time inference
        max_dim = 512
        scale = min(1.0, max_dim / max(h, w))
        target_w, target_h = int(w * scale), int(h * scale)

        img_pil = Image.fromarray(image_rgb)
        img_resized = img_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)

        # Normalize with standard ImageNet stats
        tensor_img = TF.to_tensor(img_resized)
        tensor_norm = TF.normalize(
            tensor_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ).unsqueeze(0).to(_SEG_DEVICE)

        with torch.inference_mode():
            out = model(tensor_norm)["out"]
            # Argmax across classes (class 0 is background)
            pred_classes = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

        # Foreground is any recognized non-background class (persons, animals, objects, etc.)
        fg_pred = (pred_classes > 0).astype(np.uint8) * 255

        # Resize back to original dimensions
        fg_full = cv2.resize(fg_pred, (w, h), interpolation=cv2.INTER_NEAREST)

        # Intersect with dilated brush zone to only keep subjects inside the colored zone
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_zone = cv2.dilate((mask_zone > 10).astype(np.uint8) * 255, k_dilate)
        subject_in_zone = cv2.bitwise_and(fg_full, dilated_zone)

        # Only accept if detected subject occupies a meaningful portion of the brushed zone
        zone_pixels = np.count_nonzero(mask_zone > 10)
        subject_pixels = np.count_nonzero(subject_in_zone > 0)
        if zone_pixels > 0 and subject_pixels >= 0.10 * zone_pixels:
            return subject_in_zone
        return None
    except Exception as e:
        logger.warning(f"Neural subject detection error: {e}")
        return None


def run_grabcut_refinement(
    image_rgb: np.ndarray,
    mask_zone: np.ndarray,
    seed_mask: Optional[np.ndarray] = None,
    iterations: int = 3,
) -> np.ndarray:
    """
    Run OpenCV GrabCut foreground/background segmentation using surrounding
    unmasked pixels as background samples to snap tightly to the subject contour.
    """
    h, w = image_rgb.shape[:2]
    zone_binary = (mask_zone > 10).astype(np.uint8)

    # Initialize GrabCut mask
    # GC_BGD = 0 (definite background outside brush zone)
    # GC_PR_FGD = 3 (probable foreground inside brush zone)
    # GC_PR_BGD = 2 (probable background)
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)

    # Dilate zone slightly to establish background border
    k_border = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    zone_border = cv2.dilate(zone_binary * 255, k_border) > 0

    # Inside the user zone is candidate foreground
    gc_mask[zone_binary > 0] = cv2.GC_PR_FGD

    # If neural seed mask is provided, mark its core as definite foreground
    if seed_mask is not None:
        seed_binary = (seed_mask > 10).astype(np.uint8)
        k_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        seed_core = cv2.erode(seed_binary * 255, k_core) > 0
        gc_mask[seed_core] = cv2.GC_FGD

    # Only mark the outer margin of the brush stroke as probable background (allows edge-snapping),
    # while preserving the entire internal core of the stroke (gear, backpack, camera) as foreground.
    k_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    zone_core = cv2.erode(zone_binary * 255, k_margin) > 0
    zone_margin = (zone_binary > 0) & (~zone_core)
    gc_mask[zone_margin & (gc_mask != cv2.GC_FGD)] = cv2.GC_PR_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Find bounding rect of the brush zone with margin
    coords = np.column_stack(np.where(zone_border))
    if coords.shape[0] == 0:
        return mask_zone

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add margin for background context
    pad = 30
    x1 = max(0, x_min - pad)
    y1 = max(0, y_min - pad)
    x2 = min(w, x_max + pad + 1)
    y2 = min(h, y_max + pad + 1)

    crop_img = image_rgb[y1:y2, x1:x2].copy()
    crop_gc = gc_mask[y1:y2, x1:x2].copy()

    # Must have at least some background and some foreground
    has_fg = np.any((crop_gc == cv2.GC_FGD) | (crop_gc == cv2.GC_PR_FGD))
    has_bg = np.any(crop_gc == cv2.GC_BGD)
    if not (has_fg and has_bg):
        return mask_zone

    try:
        cv2.grabCut(crop_img, crop_gc, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
        refined_crop = np.where((crop_gc == cv2.GC_FGD) | (crop_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # Place back into full mask
        refined_full = np.zeros((h, w), dtype=np.uint8)
        refined_full[y1:y2, x1:x2] = refined_crop

        # Constrain strictly to within slightly dilated brush zone (never leak outside)
        k_clip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        allowed_zone = cv2.dilate((mask_zone > 10).astype(np.uint8) * 255, k_clip)
        refined_full = cv2.bitwise_and(refined_full, allowed_zone)

        # Remove tiny noise specs and close small holes inside the subject
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined_full = cv2.morphologyEx(refined_full, cv2.MORPH_CLOSE, k_close)

        # Verification: if GrabCut collapsed to empty or stayed exactly identical to full mask,
        # verify reasonable subject ratio
        orig_count = np.count_nonzero(zone_binary)
        ref_count = np.count_nonzero(refined_full > 0)
        if ref_count >= 0.05 * orig_count:
            return refined_full
        return mask_zone
    except Exception as e:
        logger.warning(f"GrabCut refinement error: {e}")
        return mask_zone


def extract_subject_in_zone(
    image: Union[Image.Image, np.ndarray],
    mask_zone: Union[Image.Image, np.ndarray],
    device: Optional[torch.device] = None,
) -> Image.Image:
    """
    Extract and isolate the subject inside the user's brushed zone.
    
    Ensures that background pixels in the coloring zone (ocean, rocks, sky, etc.)
    are excluded from the mask so they remain 100% untouched during inpainting.
    
    Args:
        image: Full input RGB image (PIL or numpy array).
        mask_zone: Grayscale mask containing user's painted brush strokes.
        device: PyTorch compute device (CUDA / CPU).
        
    Returns:
        PIL Image (mode "L") containing refined subject mask.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.array(image)

    if isinstance(mask_zone, Image.Image):
        mask_np = np.array(mask_zone.convert("L"))
    else:
        mask_np = np.array(mask_zone)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

    if np.count_nonzero(mask_np > 10) == 0:
        return Image.fromarray(mask_np, mode="L")

    # 1. First attempt: Neural subject segmentation (person, animal, vehicle, etc.)
    neural_subject = run_neural_subject_detection(img_np, mask_np, device=device)

    # 2. Second stage: GrabCut edge-snapping and trimap refinement
    refined_subject = run_grabcut_refinement(img_np, mask_np, seed_mask=neural_subject)

    # 3. Safety Margin Dilation: expand the detected subject boundary outward by 7-11px.
    # This completely swallows all optical anti-aliasing, hair strands, lens chromatic fringing,
    # and sunlit rim-lighting around the person's silhouette, eliminating ghost outlines/borders.
    # Still constrained inside the user's painted coloring zone (never bleeds into unselected areas).
    h, w = img_np.shape[:2]
    k_size = 11 if max(h, w) >= 2500 else 7
    k_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    expanded_subject = cv2.dilate(refined_subject, k_expand)

    k_clip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    allowed_zone = cv2.dilate((mask_np > 10).astype(np.uint8) * 255, k_clip)
    refined_subject = cv2.bitwise_and(expanded_subject, allowed_zone)

    # Fallback to user mask if refined mask is empty
    if np.count_nonzero(refined_subject > 10) == 0:
        logger.info("Subject detection fell back to original painted zone.")
        return Image.fromarray(mask_np, mode="L")

    logger.info(
        f"Subject detected in zone: refined mask from "
        f"{np.count_nonzero(mask_np > 10)} to {np.count_nonzero(refined_subject > 10)} px "
        f"({np.count_nonzero(refined_subject > 10) / max(1, np.count_nonzero(mask_np > 10)) * 100:.1f}% subject)."
    )
    return Image.fromarray(refined_subject, mode="L")
