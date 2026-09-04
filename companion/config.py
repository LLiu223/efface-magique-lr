"""
config.py
Efface Magique LR - Centralized Performance & Application Configuration

Defines global tuning parameters, inference resolution bounds for CPU vs GPU,
context padding thresholds, and canvas render throttling rates.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Global configuration settings for hardware, inference, and UI rendering."""
    
    # Context cropping & padding thresholds (in pixels)
    DEFAULT_CONTEXT_PADDING: int = 35        # Default padding around mask bounding box (20-50px)
    MIN_CONTEXT_PADDING: int = 20            # Minimum padding for tight spots
    MAX_CONTEXT_PADDING: int = 50            # Maximum padding for large objects
    DEFAULT_MARGIN_RATIO: float = 0.85       # Multiplier for context-aware expansion
    MIN_MARGIN_RATIO: float = 0.50           # Minimum context margin
    
    # Neural inference resolution limits (max dimension of cropped patch)
    # CPU: Keeps patches compact (<= 768px) so inference is near-instantaneous on standard laptops
    # GPU: Allows high resolution (<= 1664px) leveraging Tensor Cores and large VRAM
    CPU_MAX_INFERENCE_DIM: int = 768
    GPU_MAX_INFERENCE_DIM: int = 1664
    
    # Interactive Canvas Rendering
    CANVAS_MAX_FPS: int = 60
    CANVAS_THROTTLE_MS: int = 16             # 16ms timer interval = ~60 FPS throttled mask update
    
    # History & Memory Management
    MAX_UNDO_HISTORY: int = 30
    AUTO_GC_MEMORY: bool = True              # Automatically invoke garbage collection after inference
    
    # Default Brush Settings
    DEFAULT_BRUSH_RADIUS: int = 30
    MIN_BRUSH_RADIUS: int = 3
    MAX_BRUSH_RADIUS: int = 300
    DEFAULT_BRUSH_FEATHER: float = 0.25
    DEFAULT_BRUSH_OPACITY: float = 0.65


# Global singleton configuration instance
CONFIG = AppConfig()
