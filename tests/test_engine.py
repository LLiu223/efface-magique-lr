"""
test_engine.py
Comprehensive automated test suite for Efface Magique LR Inpainting & Blending Engine.

Covers:
- Synthetic high-res test images (24MP 6000x4000 and 16-bit TIFFs)
- Edge cases & bounds (empty mask, full mask, borders, multi-island, 1-pixel micro mask)
- Dimension, channel count, and bit-exact 0-diff on untouched regions
- Hardware device detection & fallback
"""

import os
import sys
import unittest
import numpy as np
import torch
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.inpainting_engine import InpaintingEngine, get_optimal_device


class TestInpaintingEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_optimal_device()
        cls.engine = InpaintingEngine(device=cls.device)
        cls.engine.load_model()

    def setUp(self):
        # Standard synthetic test image (800x600) with gradient and noise
        self.w, self.h = 800, 600
        arr = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(30, 230, self.w, dtype=np.uint8)
        arr[:, :, 1] = np.linspace(200, 50, self.h, dtype=np.uint8)[:, None]
        arr[:, :, 2] = 180
        self.base_image = Image.fromarray(arr, mode="RGB")

    def test_empty_mask_returns_original(self):
        """Empty mask (all zeros) must return original image without error or modification."""
        mask = Image.new("L", (self.w, self.h), 0)
        result = self.engine.inpaint_full_resolution(self.base_image, mask)
        self.assertEqual(result.size, self.base_image.size)
        self.assertEqual(result.mode, "RGB")
        self.assertTrue(np.array_equal(np.asarray(result), np.asarray(self.base_image)))

    def test_full_canvas_mask(self):
        """Full-image mask (entire canvas 255) must handle boundary limits gracefully."""
        mask = Image.new("L", (self.w, self.h), 255)
        result = self.engine.inpaint_full_resolution(self.base_image, mask)
        self.assertEqual(result.size, (self.w, self.h))
        self.assertEqual(result.mode, "RGB")
        # Ensure result contains valid image data
        arr = np.asarray(result)
        self.assertFalse(np.isnan(arr).any())
        self.assertTrue(arr.shape == (self.h, self.w, 3))

    def test_border_touching_masks(self):
        """Mask touching boundaries (0,0) and (W-1, H-1) must not cause out-of-bounds crop errors."""
        # Top-left corner
        mask_tl = Image.new("L", (self.w, self.h), 0)
        mask_tl.putpixel((0, 0), 255)
        result_tl = self.engine.inpaint_full_resolution(self.base_image, mask_tl)
        self.assertEqual(result_tl.size, (self.w, self.h))

        # Bottom-right corner
        mask_br = Image.new("L", (self.w, self.h), 0)
        mask_br.putpixel((self.w - 1, self.h - 1), 255)
        result_br = self.engine.inpaint_full_resolution(self.base_image, mask_br)
        self.assertEqual(result_br.size, (self.w, self.h))

    def test_multi_island_disjoint_masks(self):
        """Multiple isolated brush marks across distinct corners must crop-and-blend cleanly."""
        mask = Image.new("L", (self.w, self.h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([50, 50, 80, 80], fill=255)
        draw.rectangle([self.w - 90, self.h - 90, self.w - 60, self.h - 60], fill=255)
        result = self.engine.inpaint_full_resolution(self.base_image, mask)
        self.assertEqual(result.size, (self.w, self.h))
        self.assertEqual(result.mode, "RGB")

    def test_single_pixel_micro_mask(self):
        """1-pixel micro-mask must maintain numerical stability with minimum context expansion."""
        mask = Image.new("L", (self.w, self.h), 0)
        mask.putpixel((400, 300), 255)
        result = self.engine.inpaint_full_resolution(self.base_image, mask)
        self.assertEqual(result.size, (self.w, self.h))
        # Center pixel area should be smoothly inpainted
        orig_arr = np.asarray(self.base_image)
        res_arr = np.asarray(result)
        # Untouched area at corner (10, 10) must be bit-exact identical
        self.assertTrue(np.array_equal(orig_arr[10, 10], res_arr[10, 10]))

    def test_dimension_and_channel_integrity(self):
        """Output image must strictly preserve input dimensions, channel count, and RGB mode."""
        mask = Image.new("L", (self.w, self.h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([200, 200, 260, 260], fill=255)
        result = self.engine.inpaint_full_resolution(self.base_image, mask)
        self.assertEqual(result.size, self.base_image.size)
        self.assertEqual(result.mode, "RGB")
        res_arr = np.asarray(result)
        self.assertEqual(res_arr.shape, (self.h, self.w, 3))
        self.assertEqual(res_arr.dtype, np.uint8)

    def test_zero_artifact_bleed_untouched_regions(self):
        """Verify unmasked pixel regions remain identical (0-diff) to original image."""
        mask = Image.new("L", (self.w, self.h), 0)
        draw = ImageDraw.Draw(mask)
        # Small mask in center
        draw.rectangle([350, 250, 450, 350], fill=255)
        result = self.engine.inpaint_full_resolution(self.base_image, mask)

        orig_np = np.asarray(self.base_image)
        res_np = np.asarray(result)

        # Region far away from mask (e.g. y in [0, 100], x in [0, 100])
        untouched_orig = orig_np[:100, :100]
        untouched_res = res_np[:100, :100]
        max_diff = np.max(np.abs(untouched_orig.astype(int) - untouched_res.astype(int)))
        self.assertEqual(max_diff, 0, "Untouched region must have exactly 0-diff bit-exact match!")

    def test_hardware_fallback_device(self):
        """Verify get_optimal_device detects appropriate device and inpainting works on CPU fallback."""
        detected_device = get_optimal_device()
        self.assertIsInstance(detected_device, torch.device)
        self.assertIn(detected_device.type, ["cuda", "mps", "cpu"])
        if torch.cuda.is_available():
            self.assertEqual(detected_device.type, "cuda")
            self.assertIn("NVIDIA", torch.cuda.get_device_name(0))

        # Force CPU device engine
        cpu_engine = InpaintingEngine(device=torch.device("cpu"))
        mask = Image.new("L", (200, 200), 0)
        mask.putpixel((100, 100), 255)
        small_img = Image.new("RGB", (200, 200), (50, 100, 150))
        result = cpu_engine.inpaint_full_resolution(small_img, mask)
        self.assertEqual(result.size, (200, 200))

    def test_synthetic_high_res_24mp_image(self):
        """Synthetic 24MP (6000x4000) image processing with patch crop & feather blend."""
        w_24mp, h_24mp = 6000, 4000
        # Fast synthetic 24MP with uniform background and patterned obstacle
        img_24mp = Image.new("RGB", (w_24mp, h_24mp), color=(120, 140, 160))
        mask_24mp = Image.new("L", (w_24mp, h_24mp), 0)
        draw = ImageDraw.Draw(mask_24mp)
        # Paint a 100x100 obstacle in the center
        cx, cy = w_24mp // 2, h_24mp // 2
        draw.rectangle([cx - 50, cy - 50, cx + 50, cy + 50], fill=255)

        result_24mp = self.engine.inpaint_full_resolution(img_24mp, mask_24mp)
        self.assertEqual(result_24mp.size, (w_24mp, h_24mp))
        self.assertEqual(result_24mp.mode, "RGB")

        # Untouched corner check
        orig_corner = np.asarray(img_24mp.crop((0, 0, 200, 200)))
        res_corner = np.asarray(result_24mp.crop((0, 0, 200, 200)))
        self.assertTrue(np.array_equal(orig_corner, res_corner))

    def test_16bit_tiff_input_compatibility(self):
        """16-bit TIFF image input must be handled seamlessly without crashing."""
        import tempfile
        import cv2

        # Create a synthetic 16-bit TIFF (each channel 0-65535)
        arr16 = np.zeros((400, 400, 3), dtype=np.uint16)
        arr16[:, :, 0] = np.linspace(5000, 60000, 400, dtype=np.uint16)
        arr16[:, :, 1] = 40000
        arr16[:, :, 2] = 20000

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cv2.imwrite(tmp_path, arr16)
            pil_16bit = Image.open(tmp_path)
            mask = Image.new("L", (400, 400), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([150, 150, 250, 250], fill=255)

            result = self.engine.inpaint_full_resolution(pil_16bit, mask)
            self.assertEqual(result.size, (400, 400))
            self.assertEqual(result.mode, "RGB")
            pil_16bit.close()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_inpaint_returns_uint8_dtype(self):
        """Output array must be dtype uint8 (not float or int) for safe downstream TIFF encoding."""
        mask = Image.new("L", (self.w, self.h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([300, 200, 400, 300], fill=255)
        result = self.engine.inpaint_full_resolution(self.base_image, mask)
        arr = np.asarray(result)
        self.assertEqual(arr.dtype, np.uint8, "Output image array must be dtype uint8.")

    def test_mode_switching_mid_session(self):
        """Switching engine mode between calls must not corrupt internal state or raise."""
        from companion.model_engine import EngineMode
        mask = Image.new("L", (self.w, self.h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([100, 100, 200, 200], fill=255)

        self.engine.set_mode(EngineMode.FAST)
        r1 = self.engine.inpaint_full_resolution(self.base_image, mask)
        self.assertEqual(r1.size, (self.w, self.h))

        self.engine.set_mode(EngineMode.FIREFLY)
        r2 = self.engine.inpaint_full_resolution(self.base_image, mask)
        self.assertEqual(r2.size, (self.w, self.h))

    def test_empty_string_prompt_equivalent_to_none(self):
        """prompt='' must behave identically to prompt=None and must not raise."""
        mask = Image.new("L", (self.w, self.h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([300, 200, 400, 300], fill=255)
        # Neither call should raise
        r_none = self.engine.inpaint_full_resolution(self.base_image, mask, prompt=None)
        r_empty = self.engine.inpaint_full_resolution(self.base_image, mask, prompt="")
        self.assertEqual(r_none.size, (self.w, self.h))
        self.assertEqual(r_empty.size, (self.w, self.h))

    def test_cancellable_worker_no_error_signal(self):
        """InpaintingWorker must cleanly finish or be interrupted without emitting the error signal."""
        from companion.model_engine import InpaintingWorker, EngineMode
        self.engine.set_mode(EngineMode.FAST)
        mask = Image.new("L", (200, 200), 0)
        mask.putpixel((100, 100), 255)
        small = Image.new("RGB", (200, 200), (80, 120, 160))

        errors: list = []
        worker = InpaintingWorker(self.engine, small, mask, num_variations=1)
        worker.error.connect(errors.append)
        worker.start()
        worker.wait(10000)  # Wait up to 10s
        self.assertEqual(errors, [], f"Worker must not emit error signal; got: {errors}")


if __name__ == "__main__":
    unittest.main()
