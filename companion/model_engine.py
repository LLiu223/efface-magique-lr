"""
model_engine.py
Efface Magique LR - Dual-Mode Inpainting Engine (Fast & Generative AI variations)
GPU Accelerated (NVIDIA CUDA / Tensor Cores / Apple MPS / Multi-threaded CPU)

Features:
- Pluggable inpainting model chosen from companion.models.registry (default "lama");
  only one model is resident at a time (set_model unloads before the next one loads)
- Dual Engine Mode: EngineMode.FAST (single model pass) and EngineMode.FIREFLY
  ("Generative AI" label: model inpainting with slight seeded variations, no diffusion)
- Asynchronous InpaintingWorker (QThread) with cancellation & progress reporting
- Context-Aware Local Bounding-Box Cropping (20-50px) for maximum CPU & GPU speed
- Memory Management & Garbage Collection via torch.inference_mode() and free_device_memory()
"""

import logging
import random
import threading
from enum import Enum
from typing import Optional, Callable, List
import numpy as np
import torch
from PIL import Image
import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from companion.config import CONFIG
from companion.device import (
    get_optimal_device,
    setup_device_optimizations,
    free_device_memory,
)
from companion.pipeline import (
    calculate_context_crop,
    estimate_sensor_noise_profile,
    synthesize_and_match_sensor_grain,
    feathered_sigmoid_blend,
    seamless_distance_feather_blend,
    dilate_mask_for_contact_shadows,
)
from companion.utils.subject_detector import extract_subject_in_zone
from companion.models import registry, InpaintModel, DEFAULT_MODEL

logger = logging.getLogger("EffaceMagiqueModelEngine")


class EngineMode(str, Enum):
    """Inpainting mode selector."""
    FAST = "fast"          # Single model pass for spots, wires, and rapid blemish fixes
    FIREFLY = "firefly"    # Model inpainting with slight seeded variations (UI label "Generative AI")


class InpaintingEngine:
    """
    Inpainting pipeline around the active registry model (default "lama") with two
    modes: FAST (one model pass) and FIREFLY (model inpainting with slight seeded
    variations; for models without uses_prompt an optional prompt only adds a deterministic
    colour tint of a few levels, prompt models such as sd_inpaint get the prompt itself).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        mode: EngineMode = EngineMode.FIREFLY,
    ):
        self.device = device or get_optimal_device()
        self.mode = mode
        self._model_cls = registry.get(DEFAULT_MODEL)
        self._model: Optional[InpaintModel] = None  # loaded instance; None if not loaded or load failed
        self._is_loaded = False
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_name(self) -> str:
        return self._model_cls.name

    def set_model(self, name: str):
        """Switch the active model. Unloads and frees the old one first; the new one loads lazily."""
        new_cls = registry.get(name)  # raises KeyError before anything is unloaded
        with self._load_lock:  # plain Lock: never call set_model while holding it
            if new_cls is self._model_cls:
                return
            if self._model is not None:
                self._model.unload()
                self._model = None
            free_device_memory(self.device)
            self._model_cls = new_cls
            self._is_loaded = False
        logger.info(f"Inpainting model switched to {name}.")

    def set_mode(self, mode: EngineMode):
        """Switch between Fast GAN and Firefly Generative modes."""
        self.mode = mode
        logger.info(f"Inpainting engine switched to {self.mode.value} mode.")

    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None):
        """Load required inpainting models into memory with GPU optimizations and kernel warmup."""
        if self._is_loaded:
            return
        with self._load_lock:  # plain Lock: never call load_model/generate_variations inside
            if self._is_loaded:
                return
            self._load_model_locked(progress_callback)

    def _load_model_locked(self, progress_callback: Optional[Callable[[str], None]] = None):
        if progress_callback:
            progress_callback(f"Initializing AI inpainting engine on {self.device}...")

        setup_device_optimizations(self.device)

        model = self._model_cls()
        try:
            model.load(self.device, progress_callback)
            self._model = model
        except Exception as e:
            logger.warning(f"{model.label} ({model.name}) initialization: {e}")
            self._model = None

        self._is_loaded = True

    @staticmethod
    def _generate_noise_variation(
        h: int,
        w: int,
        seed: int,
        seed_offset: int,
        device: torch.device,
        prompt: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate a stochastic variation field (H×W×3 float32) conditioned on seed
        and optional prompt string (the prompt only adds a deterministic colour
        tint of up to +/-4 levels per channel).

        Variation modes (keyed on seed_offset % 3):
          0 → flat zero (structural-only, no colour shift)
          1 → medium-frequency noise grid   (5 × scale)
          2 → fine-frequency noise grid     (6 × scale)
        Returns numpy array (H, W, 3) ready to be blended into the patch.
        """
        rng_cpu = np.random.default_rng(seed)

        def _make_mono(grid_dim_fn, strength: float) -> np.ndarray:
            grid_dim = grid_dim_fn()
            vh = max(1, h // grid_dim)
            vw = max(1, w // grid_dim)
            n = rng_cpu.normal(0, 1.0, (vh, vw)).astype(np.float32)
            n -= n.mean()
            return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) * strength

        if seed_offset == 0:
            mono = np.zeros((h, w), dtype=np.float32)
        elif seed_offset == 1:
            mono = _make_mono(lambda: max(16, min(h, w) // 16), 5.0)
        else:
            mono = _make_mono(lambda: max(24, min(h, w) // 8),  6.0)

        # Broadcast mono to 3-channel
        variation = np.stack([mono, mono, mono], axis=-1)

        # Optional prompt tint (deterministic hash → (hash%5-2)*2, i.e. up to ±4 levels per channel)
        if prompt:
            prompt_hash = sum(ord(c) for c in prompt)
            tint = np.array(
                [(prompt_hash % 5) - 2, ((prompt_hash // 5) % 5) - 2, ((prompt_hash // 25) % 5) - 2],
                dtype=np.float32,
            ) * 2.0
            variation = variation + tint

        return variation

    def _run_variation_patch(
        self,
        image_crop_rgb: Image.Image,
        mask_crop_l: Image.Image,
        seed: int,
        prompt: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate one patch variation: active-model inpainting (cv2.inpaint fallback if
        the model failed to load) plus slight seeded low-frequency noise inside the mask.
        No diffusion; the optional prompt only adds a few levels of colour tint.
        """
        crop_w, crop_h = image_crop_rgb.size
        mask_arr = np.array(mask_crop_l) > 10

        # 1. Base structure generation using the active model
        if self._model is not None:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_mask_arr = cv2.dilate(mask_arr.astype(np.uint8) * 255, k)
            mask_for_model = Image.fromarray(dilated_mask_arr)
            base_patch = self._model.inpaint(image_crop_rgb, mask_for_model, seed, prompt)
            patch_np = np.array(base_patch.convert("RGB"), dtype=np.float32)
        else:
            orig_np = np.array(image_crop_rgb.convert("RGB"))
            cv_inpaint = cv2.inpaint(orig_np, mask_arr.astype(np.uint8) * 255, 5, cv2.INPAINT_NS)
            patch_np = cv_inpaint.astype(np.float32)

        # 2. Slight seeded noise variation inside the mask
        seed_offset = seed % 3
        variation_layer = self._generate_noise_variation(
            h=crop_h, w=crop_w, seed=seed, seed_offset=seed_offset,
            device=self.device,
            # a real prompt model already used the prompt; the hash tint is only a stand-in for models without one
            prompt=None if self._model_cls.uses_prompt else prompt,
        )
        mask_float = cv2.GaussianBlur(mask_arr.astype(np.float32), (15, 15), 0)[:, :, np.newaxis]
        modulated_patch = np.clip(patch_np + (variation_layer * mask_float), 0, 255).astype(np.uint8)
        return Image.fromarray(modulated_patch)

    def generate_variations(
        self,
        image: Image.Image,
        mask: Image.Image,
        num_variations: int = 3,
        prompt: Optional[str] = None,
        base_seed: Optional[int] = None,
        margin_ratio: float = 0.85,
        feather_radius: int = 20,
        detect_subject: bool = False,
        enable_grain: bool = False,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[Image.Image]:
        """
        Run the complete photographic inpainting pipeline and return candidate variations.
        Uses local bounding-box cropping with configurable padding for maximum execution speed on both CPU and GPU.
        """
        if not self._is_loaded:
            # forward load messages (e.g. the Stable Diffusion CPU warning) to the worker's progress
            self.load_model((lambda msg: progress_callback(5, msg)) if progress_callback else None)

        mask_np = np.array(mask.convert("L"))
        if np.count_nonzero(mask_np > 10) == 0:
            logger.warning("Empty mask received; returning original image.")
            return [image] * num_variations

        is_fast_mode = (self.mode == EngineMode.FAST)
        if is_fast_mode and self._model is None:
            raise RuntimeError(
                f"Fast mode unavailable: {self._model_cls.label} model ({self.model_name!r}) failed to load "
                "(check its install and model weights)."
            )
        is_cpu = (self.device.type == "cpu")

        effective_mask = mask
        if detect_subject:
            if progress_callback:
                progress_callback(5, "Isolating subject inside coloring zone...")
            effective_mask = extract_subject_in_zone(image, mask, device=self.device)

        # 1. Automatic Mask Dilation
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

        img_w, img_h = image.size

        if progress_callback:
            progress_callback(10, "Calculating intelligent context-aware scene crop...")

        # 2. Context-Aware Local Crop & Padding
        # Use tighter bounding box (25-40px context padding) for fast spots and CPU execution
        effective_min_dim = 256 if (is_cpu or is_fast_mode) else 512
        crop_x1, crop_y1, crop_x2, crop_y2 = calculate_context_crop(
            image.size,
            dilated_mask,
            min_margin_ratio=0.35 if (is_cpu or is_fast_mode) else 0.50,
            default_margin_ratio=0.50 if (is_cpu or is_fast_mode) else margin_ratio,
            min_dim=effective_min_dim,
            custom_padding=CONFIG.DEFAULT_CONTEXT_PADDING if (is_fast_mode or is_cpu) else None,
        )
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        cropped_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2)).convert("RGB")
        cropped_mask = dilated_mask.crop((crop_x1, crop_y1, crop_x2, crop_y2)).convert("L")

        if progress_callback:
            progress_callback(25, f"Analyzing camera sensor grain on {crop_w}x{crop_h}px crop...")

        # 3. Camera Sensor Noise Profile Estimation
        noise_profile = estimate_sensor_noise_profile(cropped_img, cropped_mask)

        # Determine seeds
        if base_seed is None:
            base_seed = random.randint(1000, 999999)
        seeds = [base_seed + (i * 1000) for i in range(num_variations)]

        variations: List[Image.Image] = []

        crop_mask_l = cropped_mask.convert("L") if cropped_mask.mode != "L" else cropped_mask
        cbx, cby, cbw, cbh = cv2.boundingRect((np.asarray(crop_mask_l) > 10).astype(np.uint8))
        box_dim = max(cbw, cbh) if (cbw > 0 and cbh > 0) else 100

        if is_fast_mode or box_dim < 60:
            effective_feather = max(4, min(8, int(box_dim * 0.2)))
        else:
            effective_feather = max(10, min(16, int(feather_radius)))

        # Convert once outside the loop; an RGB image is used as-is (convert would copy it).
        # Safe because each variation below pastes into its own base_rgb.copy().
        base_rgb = image if image.mode == "RGB" else image.convert("RGB")

        try:
            for i, seed in enumerate(seeds):
                pct_start = 30 + int(i * (50 / num_variations))
                if progress_callback:
                    if is_fast_mode:
                        progress_callback(pct_start, f"Running fast inpainting on {self.device}...")
                    else:
                        var_desc = f" (Prompt: '{prompt}')" if prompt else ""
                        progress_callback(pct_start, f"Generating {self._model_cls.label} variation {i + 1}/{num_variations}{var_desc} (seed {seed})...")

                # 4. Generate raw patch
                if is_fast_mode:
                    raw_patch = self._model.inpaint(cropped_img, cropped_mask, seed, prompt)
                else:
                    raw_patch = self._run_variation_patch(cropped_img, cropped_mask, seed=seed, prompt=prompt)

                if raw_patch.size != (crop_w, crop_h):
                    raw_patch = raw_patch.resize((crop_w, crop_h), Image.Resampling.LANCZOS)

                # 5. Blending and Sensor Grain Matching
                if is_fast_mode:
                    blended_crop = seamless_distance_feather_blend(
                        cropped_img,
                        raw_patch,
                        cropped_mask,
                        feather_radius=effective_feather,
                    )
                else:
                    grain_matched_patch = synthesize_and_match_sensor_grain(
                        raw_patch,
                        noise_profile,
                        cropped_mask,
                        seed=seed,
                        enable_grain=enable_grain,
                    )
                    blended_crop = feathered_sigmoid_blend(
                        cropped_img,
                        grain_matched_patch,
                        cropped_mask,
                        feather_radius=effective_feather,
                        seed=seed,
                    )

                # 6. Paste back into full-resolution image copy (reuse base_rgb conversion)
                result_img = base_rgb.copy()
                result_img.paste(blended_crop, (crop_x1, crop_y1))
                variations.append(result_img)

                if is_fast_mode and num_variations > 1:
                    while len(variations) < num_variations:
                        variations.append(result_img.copy())
                    break

            if progress_callback:
                progress_callback(95, f"Generated {len(variations)} candidate variation(s).")

            return variations
        finally:
            if CONFIG.AUTO_GC_MEMORY:
                free_device_memory(self.device)

    def inpaint_full_resolution(
        self,
        image: Image.Image,
        mask: Image.Image,
        margin_ratio: float = 0.85,
        feather_radius: int = 20,
        detect_subject: bool = True,
        enable_grain: bool = True,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        prompt: Optional[str] = None,
    ) -> Image.Image:
        """
        Backward-compatible single-result inpainting call.
        Returns the primary (first) variation from the photographic pipeline.
        """
        results = self.generate_variations(
            image=image,
            mask=mask,
            num_variations=1,
            prompt=prompt,
            margin_ratio=margin_ratio,
            feather_radius=feather_radius,
            detect_subject=detect_subject,
            enable_grain=enable_grain,
            progress_callback=progress_callback,
        )
        return results[0]


class InpaintingWorker(QThread):
    """Asynchronous background worker thread executing inpainting pipeline."""
    progress = pyqtSignal(int, str)
    variationsReady = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        engine: InpaintingEngine,
        image: Image.Image,
        mask: Image.Image,
        num_variations: int = 3,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
        detect_subject: bool = True,
        enable_grain: bool = False,
        margin_ratio: float = 0.85,
        feather_radius: int = 20,
    ):
        super().__init__()
        self.engine = engine
        self.image = image
        self.mask = mask
        self.num_variations = num_variations
        self.prompt = prompt
        self.seed = seed
        self.detect_subject = detect_subject
        self.enable_grain = enable_grain
        self.margin_ratio = margin_ratio
        self.feather_radius = feather_radius

    def run(self):
        try:
            self.progress.emit(5, "Analyzing scene context and mask bounds...")
            variations = self.engine.generate_variations(
                image=self.image,
                mask=self.mask,
                num_variations=self.num_variations,
                prompt=self.prompt,
                base_seed=self.seed,
                margin_ratio=self.margin_ratio,
                feather_radius=self.feather_radius,
                detect_subject=self.detect_subject,
                enable_grain=self.enable_grain,
                progress_callback=lambda pct, msg: self.progress.emit(pct, msg),
            )
            self.variationsReady.emit(variations)
        except Exception as e:
            logger.exception("Inpainting worker execution failed")
            self.error.emit(str(e))
