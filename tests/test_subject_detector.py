"""
test_subject_detector.py
Unit and integration tests for Subject Detection (Object-Aware Masking)
and Chromatic Noise Elimination.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.utils.subject_detector import (
    extract_subject_in_zone,
    run_grabcut_refinement,
    get_segmentation_model,
)
from companion.utils.blending import synthesize_and_match_sensor_grain


class TestSubjectDetector(unittest.TestCase):
    def test_subject_extraction_from_loose_coloring_zone(self):
        """
        Verify that when a user draws a loose coloring zone around a subject,
        extract_subject_in_zone isolates ONLY the subject and leaves the
        surrounding background within the coloring zone unmasked.
        """
        # Create a 300x300 scene: dark background (ocean)
        scene = np.full((300, 300, 3), (60, 70, 80), dtype=np.uint8)

        # Draw a bright subject in the center (e.g. 40x80 standing figure)
        scene[110:190, 130:170] = (220, 180, 140)
        scene_pil = Image.fromarray(scene, mode="RGB")

        # User draws a loose coloring zone with 30px padding all around the figure
        loose_zone = Image.new("L", (300, 300), 0)
        draw = ImageDraw.Draw(loose_zone)
        draw.rectangle([100, 80, 200, 220], fill=255)

        loose_pixels = np.count_nonzero(np.array(loose_zone) > 0)
        expected_subject_pixels = 40 * 80  # 3200 pixels

        refined_mask = extract_subject_in_zone(scene_pil, loose_zone)
        refined_arr = np.array(refined_mask)
        refined_pixels = np.count_nonzero(refined_arr > 0)

        # The refined mask must be significantly tighter than the loose coloring zone
        self.assertLess(
            refined_pixels,
            loose_pixels * 0.75,
            "Refined mask must cut away the background in the coloring zone.",
        )
        # The subject region must be predominantly preserved
        subject_core = refined_arr[115:185, 135:165]
        self.assertGreater(
            np.mean(subject_core > 0),
            0.70,
            "Subject core inside the coloring zone must be preserved.",
        )

        # Background corner inside the loose coloring zone (e.g. [85:105, 105:125]) must NOT be masked
        bg_corner = refined_arr[85:105, 105:125]
        self.assertEqual(
            np.count_nonzero(bg_corner > 0),
            0,
            "Background in the coloring zone must remain completely unmasked.",
        )

    def test_empty_mask_handling(self):
        """Verify empty mask safely returns empty mask."""
        scene_pil = Image.new("RGB", (200, 200), (100, 100, 100))
        empty_mask = Image.new("L", (200, 200), 0)
        res = extract_subject_in_zone(scene_pil, empty_mask)
        self.assertEqual(np.count_nonzero(np.array(res)), 0)

    def test_zero_chromatic_noise_in_sensor_grain(self):
        """
        Verify that synthesize_and_match_sensor_grain generates strictly
        monochromatic luminance grain with ZERO chromatic color noise
        (i.e. std(R - G) == 0 and std(G - B) == 0 at all pixels).
        """
        # Clean smooth patch
        clean_patch = Image.new("RGB", (200, 200), color=(100, 100, 100))
        mask = Image.new("L", (200, 200), color=255)
        profile = {"mean_sigma": 15.0, "sigma_r": 15.0, "sigma_g": 15.0, "sigma_b": 15.0}

        grainy_patch = synthesize_and_match_sensor_grain(
            clean_patch, profile, mask, seed=999, enable_grain=True
        )
        arr = np.array(grainy_patch, dtype=np.float32)

        # Difference between color channels must be exactly zero (no small colored pixels)
        diff_rg = arr[:, :, 0] - arr[:, :, 1]
        diff_gb = arr[:, :, 1] - arr[:, :, 2]

        max_color_diff = max(np.max(np.abs(diff_rg)), np.max(np.abs(diff_gb)))
        self.assertEqual(
            max_color_diff,
            0.0,
            "Sensor grain must be strictly monochromatic with ZERO chromatic color noise.",
        )

    def test_grain_toggle_disabled(self):
        """Verify that when enable_grain=False, the image remains completely clean."""
        clean_patch = Image.new("RGB", (100, 100), color=(128, 128, 128))
        mask = Image.new("L", (100, 100), color=255)
        profile = {"mean_sigma": 10.0}

        out = synthesize_and_match_sensor_grain(
            clean_patch, profile, mask, seed=42, enable_grain=False
        )
        self.assertTrue(np.array_equal(np.array(clean_patch), np.array(out)))


if __name__ == "__main__":
    unittest.main()
