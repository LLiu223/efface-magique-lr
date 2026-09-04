"""
companion.utils.blending
High-Fidelity Photographic Inpainting Pipeline Utilities.

Backward-compatibility facade re-exporting from companion.pipeline:
- srgb_to_linear / linear_to_srgb
- dilate_mask_for_contact_shadows
- calculate_context_crop
- estimate_sensor_noise_profile
- synthesize_and_match_sensor_grain
- harmonic_boundary_harmonization
- synthesize_structural_texture
- seamless_distance_feather_blend
- feathered_sigmoid_blend
"""

from companion.pipeline import (
    srgb_to_linear,
    linear_to_srgb,
    dilate_mask_for_contact_shadows,
    calculate_context_crop,
    estimate_sensor_noise_profile,
    synthesize_and_match_sensor_grain,
    harmonic_boundary_harmonization,
    synthesize_structural_texture,
    seamless_distance_feather_blend,
    feathered_sigmoid_blend,
    logger,
)

__all__ = [
    "srgb_to_linear",
    "linear_to_srgb",
    "dilate_mask_for_contact_shadows",
    "calculate_context_crop",
    "estimate_sensor_noise_profile",
    "synthesize_and_match_sensor_grain",
    "harmonic_boundary_harmonization",
    "synthesize_structural_texture",
    "seamless_distance_feather_blend",
    "feathered_sigmoid_blend",
    "logger",
]
