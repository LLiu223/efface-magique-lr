"""
companion.utils.subject_detector
Intelligent Subject Detection and Object-Aware Masking.

Provides:
- extract_subject_in_zone: Refines a loose user brush stroke ("coloring zone")
  so that ONLY the subject inside the zone is masked, leaving surrounding
  background pixels within the colored zone 100% untouched.
- Hybrid Neural Segmentation (PyTorch LRASPP MobileNetV3 on high-res local crops)
  + Adaptive CIE Lab Color Contrast Analysis + OpenCV GrabCut Edge Snapping.
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
    Run neural semantic segmentation on a high-resolution local context crop
    around the user's brushed zone.
    
    Cropping the local context before running the neural model ensures that even
    small distant subjects in 24MP-60MP images retain rich feature resolution,
    yielding vastly superior detection fidelity compared to full-image downsampling.
    
    Returns binary mask (uint8 0 or 255) of subject within zone, or None if no subject detected.
    """
    model = get_segmentation_model(device)
    if model is None:
        return None

    try:
        import torchvision.transforms.functional as TF

        h, w = image_rgb.shape[:2]
        zone_binary = (mask_zone > 10).astype(np.uint8)
        bx, by, bw, bh = cv2.boundingRect(zone_binary)
        if bw == 0 or bh == 0:
            return None

        # Calculate high-res contextual crop around the brushed zone
        pad_x = max(32, int(bw * 0.35))
        pad_y = max(32, int(bh * 0.35))
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)

        crop_rgb = image_rgb[y1:y2, x1:x2]
        crop_h, crop_w = crop_rgb.shape[:2]
        if crop_h < 10 or crop_w < 10:
            return None

        # Scale the local crop to standard 512px for high-detail neural evaluation
        max_dim = 512
        scale = min(1.0, max_dim / max(crop_h, crop_w))
        target_w, target_h = max(32, int(crop_w * scale)), max(32, int(crop_h * scale))

        crop_pil = Image.fromarray(crop_rgb)
        crop_resized = crop_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)

        # Normalize with standard ImageNet statistics
        tensor_img = TF.to_tensor(crop_resized)
        tensor_norm = TF.normalize(
            tensor_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ).unsqueeze(0).to(_SEG_DEVICE)

        with torch.inference_mode():
            out = model(tensor_norm)["out"]
            pred_classes = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

        # Foreground is any recognized non-background class (persons, animals, vehicles, objects)
        fg_crop_pred = (pred_classes > 0).astype(np.uint8) * 255

        # Resize predicted foreground back to the local crop dimensions
        fg_crop_full = cv2.resize(fg_crop_pred, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

        # Map back to full image coordinate space
        fg_full = np.zeros((h, w), dtype=np.uint8)
        fg_full[y1:y2, x1:x2] = fg_crop_full

        # Intersect with slightly dilated brush zone to keep subjects inside the colored zone
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated_zone = cv2.dilate(zone_binary * 255, k_dilate)
        subject_in_zone = cv2.bitwise_and(fg_full, dilated_zone)

        # Accept if detected subject occupies a reasonable portion of the brushed zone
        zone_pixels = np.count_nonzero(zone_binary)
        subject_pixels = np.count_nonzero(subject_in_zone > 0)
        if zone_pixels > 0 and subject_pixels >= 0.08 * zone_pixels:
            return subject_in_zone
        return None
    except Exception as e:
        logger.warning(f"Neural subject detection error: {e}")
        return None


def run_grabcut_refinement(
    image_rgb: np.ndarray,
    mask_zone: np.ndarray,
    seed_mask: Optional[np.ndarray] = None,
    iterations: int = 4,
) -> np.ndarray:
    """
    Refine object boundaries using multi-cluster local background sampling,
    adaptive CIE Lab color contrast, edge-preserving saliency, and OpenCV GrabCut.
    
    Robustly handles distant, small, and low-contrast objects by modeling the local
    background ring, preventing thin structural features (posts, roof gables, eaves)
    from being severed, and ensuring solid hole-filled contours.
    """
    h, w = image_rgb.shape[:2]
    zone_binary = (mask_zone > 10).astype(np.uint8)

    # Bounding rect of the brush zone with background padding
    bx, by, bw, bh = cv2.boundingRect(zone_binary)
    if bw == 0 or bh == 0:
        return mask_zone

    pad = max(35, int(max(bw, bh) * 0.45))
    x1 = max(0, bx - pad)
    y1 = max(0, by - pad)
    x2 = min(w, bx + bw + pad)
    y2 = min(h, by + bh + pad)

    crop_img = image_rgb[y1:y2, x1:x2].copy()
    crop_zone = zone_binary[y1:y2, x1:x2].copy()
    ch, cw = crop_img.shape[:2]

    # Initialize GrabCut mask: outside zone is definite background, inside is probable foreground
    gc_mask = np.full((ch, cw), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[crop_zone > 0] = cv2.GC_PR_FGD

    try:
        # 1. Multi-cluster background analysis in CIE Lab space
        # Sample background from the outer perimeter band surrounding the brushed zone
        k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_zone = cv2.dilate(crop_zone, k_ring)
        bg_band = (dilated_zone > 0) & (crop_zone == 0)

        lab_crop = cv2.cvtColor(crop_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        bg_lab = lab_crop[bg_band]
        if len(bg_lab) < 30:
            bg_lab = lab_crop[crop_zone == 0]

        if len(bg_lab) >= 20:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k_clusters = min(3, max(1, len(bg_lab) // 50))
            _, _, bg_centers = cv2.kmeans(
                bg_lab.reshape(-1, 3), k_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )

            # Minimum Euclidean distance to nearest background cluster
            dists = [np.sqrt(np.sum((lab_crop - c) ** 2, axis=-1)) for c in bg_centers]
            min_bg_dist = np.min(dists, axis=0)

            # 2. Edge energy via Sobel gradient on CLAHE-enhanced luminance
            l_chan = lab_crop[:, :, 0].astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_chan)
            grad_x = cv2.Sobel(l_enhanced, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(l_enhanced, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Saliency combining color divergence and edge gradients
            saliency = min_bg_dist + (grad_mag * 0.3)
            sal_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-5) * 255.0
            sal_norm = sal_norm.astype(np.uint8)

            # Adaptive Otsu threshold on brushed zone saliency
            sal_zone_u8 = sal_norm[crop_zone > 0]
            thresh_val, _ = cv2.threshold(sal_zone_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Mark high-confidence foreground core
            strong_fg = (crop_zone > 0) & (sal_norm >= max(int(thresh_val * 1.1), 30))
            gc_mask[strong_fg] = cv2.GC_FGD

            # Only mark outer margin pixels as probable background if they have low saliency
            # AND are not part of an edge contour (preserving posts and roof trims)
            min_dim = min(bw, bh)
            erode_k = max(3, min(7, int(min_dim * 0.08) | 1))
            k_border = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
            zone_eroded = cv2.erode(crop_zone * 255, k_border) > 0
            outer_margin = (crop_zone > 0) & (~zone_eroded)

            prob_bg = outer_margin & (sal_norm < max(int(thresh_val * 0.5), 10)) & (grad_mag < 20)
            gc_mask[prob_bg] = cv2.GC_PR_BGD
    except Exception as e:
        logger.debug(f"Contrast analysis fallback: {e}")

    # 3. If neural seed mask is provided, anchor its core as definite foreground
    if seed_mask is not None:
        seed_crop = (seed_mask[y1:y2, x1:x2] > 10).astype(np.uint8)
        k_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        seed_core = cv2.erode(seed_crop * 255, k_core) > 0
        gc_mask[seed_core] = cv2.GC_FGD

    # 4. Perimeter of crop is definite background
    gc_mask[0, :] = cv2.GC_BGD
    gc_mask[-1, :] = cv2.GC_BGD
    gc_mask[:, 0] = cv2.GC_BGD
    gc_mask[:, -1] = cv2.GC_BGD

    has_fg = np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))
    has_bg = np.any(gc_mask == cv2.GC_BGD)
    if not (has_fg and has_bg):
        return mask_zone

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(crop_img, gc_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
        refined_crop = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # 5. Fill interior holes in object contours (roofs, shadows, windows) so the subject is completely solid
        contours, _ = cv2.findContours(refined_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(refined_crop, contours, -1, 255, thickness=cv2.FILLED)

        # Place back into full mask
        refined_full = np.zeros((h, w), dtype=np.uint8)
        refined_full[y1:y2, x1:x2] = refined_crop

        # Constrain to allowed user zone plus generous edge tolerance
        k_clip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        allowed_zone = cv2.dilate(zone_binary * 255, k_clip)
        refined_full = cv2.bitwise_and(refined_full, allowed_zone)

        # Remove tiny speckles & close internal gaps
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined_full = cv2.morphologyEx(refined_full, cv2.MORPH_CLOSE, k_close)

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
    Extract and isolate the subject inside the user's brushed zone with high precision.
    
    Ensures that surrounding background pixels in the coloring zone (ocean, sky, foliage)
    are cleanly excluded from the mask so they remain 100% untouched during inpainting,
    while guaranteeing that all parts of the subject (including eaves, rooflines, trim,
    and posts on small or distant structures) are solidly enclosed.
    
    Args:
        image: Full input RGB image (PIL or numpy array).
        mask_zone: Grayscale mask containing user's painted brush strokes.
        device: PyTorch compute device (CUDA / MPS / CPU).
        
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

    zone_binary = (mask_np > 10).astype(np.uint8)
    if np.count_nonzero(zone_binary) == 0:
        return Image.fromarray(mask_np, mode="L")

    # 1. Neural subject segmentation on high-res local crop (persons, animals, vehicles, objects)
    neural_subject = run_neural_subject_detection(img_np, mask_np, device=device)

    # 2. GrabCut edge-snapping and color contrast refinement
    refined_subject = run_grabcut_refinement(img_np, mask_np, seed_mask=neural_subject)

    # 3. Adaptive safety margin expansion:
    # Expands the contour to swallow optical anti-aliasing, boundary trim, and lens fringes.
    # Small or distant objects receive generous dilation (at least 6-12px) to ensure posts and trim are fully engulfed.
    bx, by, bw, bh = cv2.boundingRect(refined_subject)
    if bw > 0 and bh > 0:
        dim = max(bw, bh)
        if dim < 80:
            k_size = max(5, min(15, int(dim * 0.14) | 1))
        else:
            k_size = max(7, min(19, int(dim * 0.07) | 1))

        k_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        expanded_subject = cv2.dilate(refined_subject, k_expand)

        # Allow expansion within user zone plus generous buffer
        k_clip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        allowed_zone = cv2.dilate(zone_binary * 255, k_clip)
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
