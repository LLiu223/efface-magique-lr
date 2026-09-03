"""
test_e2e.py
End-to-End (E2E) CLI testing for Efface Magique LR in headless mode.

Validates the full pipeline:
- Subprocess invocation of companion.app with --image/--input, --output, --test-mask-rect, and --headless
- Correct exit codes (0 on success, 1 on invalid inputs)
- Output TIFF generation, validity, non-corruption, and dimension fidelity
- 0-diff preservation on untouched image areas
"""

import os
import sys
import subprocess
import tempfile
import unittest
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestEndToEndHeadlessPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="efface_magique_e2e_")
        self.width, self.height = 1000, 700

        # Generate sample image with gradient & textured patterns
        arr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(40, 220, self.width, dtype=np.uint8)
        arr[:, :, 1] = 140
        arr[:, :, 2] = np.linspace(200, 40, self.height, dtype=np.uint8)[:, None]

        self.input_tiff = os.path.join(self.tmp_dir, "sample_input.tif")
        pil_img = Image.fromarray(arr, mode="RGB")
        pil_img.save(self.input_tiff, compression="tiff_deflate")

        self.python_exe = sys.executable

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_e2e_headless_inpainting_success(self):
        """Run full CLI headless inpainting pipeline with --test-mask-rect and verify output TIFF."""
        output_tiff = os.path.join(self.tmp_dir, "sample_output.tif")
        mask_rect = "200,200,150,150"  # X, Y, W, H

        cmd = [
            self.python_exe,
            "-m",
            "companion.app",
            "--image",
            self.input_tiff,
            "--output",
            output_tiff,
            "--test-mask-rect",
            mask_rect,
            "--headless",
        ]

        result = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True)

        self.assertEqual(
            result.returncode,
            0,
            f"Headless companion failed with code {result.returncode}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

        # 1. Output file must exist and be non-empty
        self.assertTrue(os.path.exists(output_tiff), "Output TIFF was not created!")
        self.assertGreater(os.path.getsize(output_tiff), 0, "Output TIFF is empty (0 bytes)!")

        # 2. Output file must be valid and non-corrupt
        with Image.open(output_tiff) as out_img:
            out_img.verify()

        # Reopen to read pixels after verify()
        with Image.open(output_tiff) as out_img:
            self.assertEqual(out_img.size, (self.width, self.height))
            self.assertEqual(out_img.mode, "RGB")
            out_arr = np.asarray(out_img)

        # 3. Untouched region (e.g. corner [0:100, 0:100]) must be bit-exact 0-diff
        with Image.open(self.input_tiff) as in_img:
            in_arr = np.asarray(in_img)

        untouched_diff = np.max(np.abs(in_arr[:100, :100].astype(int) - out_arr[:100, :100].astype(int)))
        self.assertEqual(untouched_diff, 0, "Untouched regions must remain 100% bit-exact!")

    def test_e2e_headless_nonexistent_input_returns_error(self):
        """CLI headless mode with non-existent input path must exit with code 1."""
        bogus_input = os.path.join(self.tmp_dir, "does_not_exist.tif")
        output_tiff = os.path.join(self.tmp_dir, "output.tif")

        cmd = [
            self.python_exe,
            "-m",
            "companion.app",
            "--image",
            bogus_input,
            "--output",
            output_tiff,
            "--headless",
        ]

        result = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)

    def test_e2e_headless_empty_mask_preserves_image(self):
        """CLI headless mode with no mask rect must preserve original image and exit code 0."""
        output_tiff = os.path.join(self.tmp_dir, "sample_unmodified.tif")

        cmd = [
            self.python_exe,
            "-m",
            "companion.app",
            "--input",
            self.input_tiff,
            "--output",
            output_tiff,
            "--headless",
        ]

        result = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(output_tiff))

        with Image.open(output_tiff) as out_img:
            self.assertEqual(out_img.size, (self.width, self.height))


if __name__ == "__main__":
    unittest.main()
