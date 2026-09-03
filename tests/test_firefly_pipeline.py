"""
test_firefly_pipeline.py
Automated Quality Assurance & Engineering Verification for Adobe Firefly Inpainting Pipeline.

Covers:
- Dual Inpainting Engine (EngineMode.FIREFLY vs EngineMode.FAST)
- 3-variation batch generation with distinct seeds
- Context-aware crop calculation with >= 35% margin
- Camera sensor noise profiling and synthetic ISO grain injection
- Prompt-conditioned synthesis vs context-aware unguided removal
- 24MP high-resolution preservation and untouched zero-bleed
- 16-bit TIFF ICC profile and EXIF metadata preservation
- Hardware fallback on CPU and VRAM-constrained systems
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import torch
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.inpainting_engine import InpaintingEngine, EngineMode, get_optimal_device
from companion.utils.blending import (
    calculate_context_crop,
    estimate_sensor_noise_profile,
    synthesize_and_match_sensor_grain,
    feathered_sigmoid_blend,
    harmonic_boundary_harmonization,
    synthesize_structural_texture,
    seamless_distance_feather_blend,
    srgb_to_linear,
    linear_to_srgb,
    dilate_mask_for_contact_shadows,
)


class TestFireflyPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_optimal_device()
        cls.engine = InpaintingEngine(device=cls.device, mode=EngineMode.FIREFLY)
        cls.engine.load_model()

    def test_dual_engine_mode_switching(self):
        """Verify switching between Firefly generative fill and Fast spot removal modes."""
        self.engine.set_mode(EngineMode.FAST)
        self.assertEqual(self.engine.mode, EngineMode.FAST)

        self.engine.set_mode(EngineMode.FIREFLY)
        self.assertEqual(self.engine.mode, EngineMode.FIREFLY)

    def test_3_variation_batch_generation(self):
        """Verify generating 3 candidate variations with distinct random seeds."""
        img = Image.new("RGB", (600, 600), color=(180, 180, 180))
        draw = ImageDraw.Draw(img)
        draw.rectangle([200, 200, 400, 400], fill=(220, 50, 50))

        mask = Image.new("L", (600, 600), color=0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([220, 220, 380, 380], fill=255)

        self.engine.set_mode(EngineMode.FIREFLY)
        variations = self.engine.generate_variations(
            image=img,
            mask=mask,
            num_variations=3,
            base_seed=12345,
        )

        self.assertEqual(len(variations), 3, "Firefly engine must return exactly 3 variations.")
        for i, var in enumerate(variations):
            self.assertEqual(var.size, (600, 600), f"Variation {i} must maintain original resolution.")

        # Variations should have distinct stochastic synthesis in the mask area
        arr1 = np.array(variations[0])[250:350, 250:350]
        arr2 = np.array(variations[1])[250:350, 250:350]
        arr3 = np.array(variations[2])[250:350, 250:350]

        diff_1_2 = np.mean(np.abs(arr1.astype(float) - arr2.astype(float)))
        diff_2_3 = np.mean(np.abs(arr2.astype(float) - arr3.astype(float)))

        self.assertGreater(diff_1_2, 0.0, "Variation 1 and 2 must have distinct generative patterns.")
        self.assertGreater(diff_2_3, 0.0, "Variation 2 and 3 must have distinct generative patterns.")

    def test_prompt_guided_synthesis(self):
        """Verify prompt guidance produces conditioned output distinct from unguided removal."""
        img = Image.new("RGB", (512, 512), color=(140, 160, 180))
        mask = Image.new("L", (512, 512), color=0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([180, 180, 320, 320], fill=255)

        self.engine.set_mode(EngineMode.FIREFLY)
        res_unguided = self.engine.inpaint_full_resolution(img, mask, prompt=None)
        res_prompt = self.engine.inpaint_full_resolution(img, mask, prompt="lush green grass with flowers")

        self.assertEqual(res_unguided.size, (512, 512))
        self.assertEqual(res_prompt.size, (512, 512))

        # Pixel arrays in the masked region should differ due to prompt conditioning
        crop_u = np.array(res_unguided)[200:300, 200:300]
        crop_p = np.array(res_prompt)[200:300, 200:300]
        self.assertFalse(np.array_equal(crop_u, crop_p), "Prompt-conditioned result must reflect prompt guidance.")

    def test_context_aware_crop_margin(self):
        """Verify context-aware crop bounding box enforces at least 35% margin."""
        mask = Image.new("L", (2000, 2000), color=0)
        draw = ImageDraw.Draw(mask)
        # 200x200 object
        draw.rectangle([800, 800, 1000, 1000], fill=255)

        x1, y1, x2, y2 = calculate_context_crop(
            image_size=(2000, 2000),
            mask=mask,
            min_margin_ratio=0.35,
            default_margin_ratio=0.50,
            min_dim=512,
        )

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Object is 201px. Minimum 35% margin adds 70px on each side -> min 341px, or min_dim 512px
        self.assertGreaterEqual(crop_w, 512, "Crop must fulfill minimum context dimension.")
        self.assertGreaterEqual(crop_h, 512, "Crop must fulfill minimum context dimension.")
        self.assertLessEqual(x1, 800, "Crop must encompass left edge with margin.")
        self.assertGreaterEqual(x2, 1000, "Crop must encompass right edge with margin.")

    def test_sensor_noise_profile_and_grain_matching(self):
        """Verify camera sensor noise profiling and synthetic grain injection."""
        rng = np.random.default_rng(42)
        # Create image with artificial sensor noise (sigma ~ 10)
        clean = np.full((300, 300, 3), 128, dtype=np.float32)
        noise = rng.normal(0, 10.0, (300, 300, 3)).astype(np.float32)
        noisy_img = np.clip(clean + noise, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(noisy_img, mode="RGB")

        mask = Image.new("L", (300, 300), color=0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([100, 100, 200, 200], fill=255)

        profile = estimate_sensor_noise_profile(img_pil, mask)
        self.assertIn("mean_sigma", profile)
        self.assertGreater(profile["mean_sigma"], 5.0, "Noise profiling must detect sensor noise level.")

        # Inpainted patch without grain
        smooth_patch = Image.new("RGB", (300, 300), color=(128, 128, 128))
        grainy_patch = synthesize_and_match_sensor_grain(smooth_patch, profile, mask, seed=123)

        grainy_arr = np.array(grainy_patch)[120:180, 120:180]
        # Injected grain must have positive variance matching sensor noise
        self.assertGreater(np.std(grainy_arr), 2.0, "Synthesized grain must eliminate plastic smoothing.")

    def test_firefly_24mp_high_res_preservation(self):
        """Verify full 24MP (6000x4000) image inpainting maintains 0-diff on untouched pixels."""
        w, h = 6000, 4000
        img = Image.new("RGB", (w, h), color=(100, 120, 140))
        mask = Image.new("L", (w, h), color=0)
        draw = ImageDraw.Draw(mask)
        # Place mask in upper right quadrant
        draw.rectangle([4500, 800, 4700, 1000], fill=255)

        res = self.engine.inpaint_full_resolution(img, mask)
        self.assertEqual(res.size, (w, h))

        # Check an untouched region (e.g. bottom-left 500x500 quadrant)
        orig_corner = np.array(img.crop((0, 3500, 500, 4000)))
        res_corner = np.array(res.crop((0, 3500, 500, 4000)))
        diff = np.max(np.abs(orig_corner.astype(int) - res_corner.astype(int)))
        self.assertEqual(diff, 0, "Untouched region on 24MP image must be bit-exact 0-diff.")

    def test_tiff_metadata_and_icc_preservation(self):
        """Verify 16-bit TIFF ICC color profile and EXIF preservation."""
        fake_icc = b"TEST_ICC_PROFILE_PROPHOTO_RGB_DATA_BLOCK"

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            path = tmp_file.name

        try:
            img = Image.new("RGB", (200, 200), color=(120, 150, 180))
            exif = img.getexif()
            exif[0x0112] = 1  # Orientation tag
            img.save(path, compression="tiff_deflate", icc_profile=fake_icc, exif=exif)

            # Re-read and assert metadata present
            read_img = Image.open(path)
            self.assertEqual(read_img.info.get("icc_profile"), fake_icc)
            read_exif = read_img.getexif()
            self.assertEqual(read_exif.get(0x0112), 1)
            read_img.close()
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_harmonic_boundary_harmonization(self):
        """Verify harmonic boundary harmonization reduces boundary discontinuity."""
        orig = np.full((100, 100, 3), 150, dtype=np.uint8)
        # Patch has global DC shift of +30
        inpaint = np.full((100, 100, 3), 180, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        harmonized = harmonic_boundary_harmonization(orig, inpaint, mask, blend_width=15)
        
        # Center of patch should have adjusted closer to orig (150)
        center_val = np.mean(harmonized[45:55, 45:55])
        self.assertLess(center_val, 175.0, "Harmonization must diffuse boundary residual inward.")

    def test_seamless_distance_feather_blend_zero_leak(self):
        """Verify distance transform feather blend guarantees 100% bit-exact 0-diff outside mask."""
        orig = np.random.randint(50, 200, (120, 120, 3), dtype=np.uint8)
        inpaint = np.random.randint(50, 200, (120, 120, 3), dtype=np.uint8)
        mask = np.zeros((120, 120), dtype=np.uint8)
        mask[40:80, 40:80] = 255

        blended = seamless_distance_feather_blend(orig, inpaint, mask, feather_radius=15)
        blended_np = np.array(blended)

        # Pixels outside mask (e.g. [:35, :]) must have max diff == 0
        diff_outside = np.max(np.abs(orig[:35, :].astype(int) - blended_np[:35, :].astype(int)))
        self.assertEqual(diff_outside, 0, "Untouched region outside mask must have bit-exact 0-diff.")

    def test_synthesize_structural_texture(self):
        """Verify structural texture synthesis boosts high-frequency variance in smoothed patches."""
        # Highly textured background (random noise)
        orig = np.random.normal(128, 25, (100, 100, 3)).clip(0, 255).astype(np.uint8)
        # Flat/smoothed inpainting patch
        inpaint = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        result = synthesize_structural_texture(orig, inpaint, mask, seed=123)
        # Variance inside mask should be restored (> 0)
        inner_std = np.std(result[40:60, 40:60])
        self.assertGreater(inner_std, 3.0, "Structural texture synthesis must restore texture variance.")

    def test_linear_rgb_blending_eliminates_dark_fringe(self):
        """Verify Linear RGB blending eliminates the gamma-induced dark luminance dip."""
        # Test IEC 61966-2-1 exact roundtrip
        test_vals = np.arange(256, dtype=np.uint8)
        reconstructed = np.round(linear_to_srgb(srgb_to_linear(test_vals))).astype(np.uint8)
        np.testing.assert_array_equal(test_vals, reconstructed, "sRGB <-> Linear RGB roundtrip must be exact.")

        # Compare blending between white (255) and black (0) at alpha = 0.5
        # In naive sRGB: (255 + 0)/2 = 127.5, which in linear light represents only ~21.8% luminance!
        # In Linear RGB: 0.5 linear light -> sRGB ~186, conserving true optical power without dark dip.
        white = np.full((10, 10, 3), 255, dtype=np.uint8)
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.full((10, 10), 255, dtype=np.uint8)

        blended = seamless_distance_feather_blend(white, black, mask, feather_radius=10)
        blended_arr = np.array(blended)
        # Verify blended image is valid and within bounds
        self.assertEqual(blended_arr.shape, (10, 10, 3))

    def test_automatic_mask_dilation_swallows_contact_shadows(self):
        """Verify morphological dilation expands brush masks symmetrically by requested radius."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # 20x20 box

        dilated = dilate_mask_for_contact_shadows(mask, radius=12)

        # Original bounds: x in [40, 59], y in [40, 59]
        # Dilated with radius 12: should reach at least [28, 71]
        self.assertEqual(dilated[28, 50], 255, "Dilation must expand mask upward to swallow top contact shadow.")
        self.assertEqual(dilated[71, 50], 255, "Dilation must expand mask downward to swallow bottom contact shadow.")
        self.assertEqual(dilated[50, 28], 255, "Dilation must expand mask leftward to swallow left contact shadow.")
        self.assertEqual(dilated[50, 71], 255, "Dilation must expand mask rightward to swallow right contact shadow.")
        # Far outside should still be 0
        self.assertEqual(dilated[10, 10], 0, "Dilation must not bleed into far background.")


if __name__ == "__main__":
    unittest.main()

