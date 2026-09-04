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

    def test_distant_small_object_detection(self):
        """
        Verify that a small, distant, low-contrast structure (like a house, barn,
        or cabin on a distant hill) is accurately detected and cleanly masked
        including all trim and structural features without leaving severed edges.
        """
        # Low contrast landscape scene: muted grass/hill background
        scene = np.full((200, 200, 3), (78, 82, 75), dtype=np.uint8)

        # Draw a small 30x25 cabin at [85:110, 85:115]
        # Dark reddish wall
        scene[90:110, 88:112] = (95, 75, 72)
        # White / light roof eaves and corner posts
        scene[85:90, 86:114] = (115, 112, 110)  # roof gable
        scene[90:110, 86:88] = (115, 112, 110)   # left post
        scene[90:110, 112:114] = (115, 112, 110) # right post

        scene_pil = Image.fromarray(scene, mode="RGB")

        # User paints a loose coloring zone around the structure
        loose_zone = Image.new("L", (200, 200), 0)
        draw = ImageDraw.Draw(loose_zone)
        draw.rectangle([70, 70, 130, 125], fill=255)

        refined_mask = extract_subject_in_zone(scene_pil, loose_zone)
        ref_arr = np.array(refined_mask)

        # 1. Must isolate subject from loose background
        loose_count = np.count_nonzero(np.array(loose_zone) > 0)
        ref_count = np.count_nonzero(ref_arr > 0)
        self.assertLess(ref_count, loose_count, "Refined mask must cut away outer background.")

        # 2. Entire structure (wall + roof gable + corner posts) must be enclosed
        # Roof gable
        self.assertTrue(np.all(ref_arr[85:90, 88:112] > 0), "Roof gable must be masked.")
        # Left and right posts
        self.assertTrue(np.all(ref_arr[92:108, 86:88] > 0), "Left post must be masked.")
        self.assertTrue(np.all(ref_arr[92:108, 112:114] > 0), "Right post must be masked.")
        # Interior wall core
        self.assertTrue(np.all(ref_arr[95:105, 92:108] > 0), "Interior wall must be solid.")

        # 3. Outer background in loose zone (e.g. corner [72:76, 72:76]) must remain unmasked
        bg_corner = ref_arr[72:76, 72:76]
        self.assertEqual(np.count_nonzero(bg_corner > 0), 0, "Distant background must remain unmasked.")

    def test_grabcut_minimum_zone_threshold(self):
        """
        A very small brush zone (fewer pixels than CONFIG.SUBJECT_DETECT_MIN_ZONE_PX)
        must not crash and must return a valid non-empty or empty mask gracefully.
        """
        from companion.config import CONFIG
        scene = Image.new("RGB", (200, 200), (80, 90, 100))
        # Paint exactly 5 pixels — below the minimum zone threshold
        tiny_zone = Image.new("L", (200, 200), 0)
        tiny_zone.putpixel((100, 100), 255)
        tiny_zone.putpixel((101, 100), 255)
        tiny_zone.putpixel((102, 100), 255)

        result = extract_subject_in_zone(scene, tiny_zone)
        # Must not raise; may return any valid L-mode image
        self.assertEqual(result.mode, "L")
        self.assertEqual(result.size, (200, 200))

    def test_neural_model_caching(self):
        """Calling get_segmentation_model() twice must return the exact same object (not reload)."""
        import torch
        cpu = torch.device("cpu")
        model_a = get_segmentation_model(cpu)
        model_b = get_segmentation_model(cpu)
        if model_a is not None:
            self.assertIs(model_a, model_b, "Second call must return the cached model instance.")

    def test_subject_extraction_pil_and_numpy_equivalent(self):
        """PIL Image input and numpy array input must produce identical masks."""
        scene = np.full((150, 150, 3), (60, 70, 80), dtype=np.uint8)
        scene[60:90, 60:90] = (200, 170, 140)
        scene_pil = Image.fromarray(scene, mode="RGB")

        zone_pil = Image.new("L", (150, 150), 0)
        draw = ImageDraw.Draw(zone_pil)
        draw.rectangle([45, 45, 105, 105], fill=255)
        zone_np = np.array(zone_pil)

        result_pil  = extract_subject_in_zone(scene_pil, zone_pil)
        result_np   = extract_subject_in_zone(scene, zone_np)

        arr_pil = np.array(result_pil)
        arr_np  = np.array(result_np)
        # Both paths must produce the same pixel data
        self.assertTrue(np.array_equal(arr_pil, arr_np),
                        "PIL and numpy inputs must produce identical refined masks.")

    def test_full_image_mask_fallback(self):
        """
        A mask equal to the full image (all 255) must not crash GrabCut's
        perimeter-background assumption and must return a valid mask.
        """
        scene = Image.new("RGB", (100, 100), (128, 128, 128))
        full_mask = Image.new("L", (100, 100), 255)
        result = extract_subject_in_zone(scene, full_mask)
        self.assertEqual(result.mode, "L")
        self.assertEqual(result.size, (100, 100))
        # Must not be totally empty — either the full mask or a refined subset
        self.assertGreater(np.count_nonzero(np.array(result)), 0)

    def test_grabcut_kmin_two_cluster_coverage(self):
        """
        Verify that run_grabcut_refinement works correctly with a zone that has
        enough background pixels for at least 2 k-means clusters (the bug-fixed path).
        A zone with > 80 background pixels around the subject should not collapse.
        """
        scene = np.full((300, 300, 3), (50, 50, 200), dtype=np.uint8)  # blue background
        scene[130:170, 130:170] = (230, 180, 100)  # yellow square subject
        zone = np.zeros((300, 300), dtype=np.uint8)
        zone[100:200, 100:200] = 255  # 100×100 loose zone → plenty of background pixels

        result = run_grabcut_refinement(scene, zone)
        result_count = np.count_nonzero(result > 0)
        zone_count = np.count_nonzero(zone > 0)
        # Refinement must produce something, and must be smaller than the full zone
        self.assertGreater(result_count, 0, "GrabCut must detect subject with ≥2 clusters.")
        self.assertLess(result_count, zone_count, "GrabCut must tighten the mask with ≥2 clusters.")


if __name__ == "__main__":
    unittest.main()

