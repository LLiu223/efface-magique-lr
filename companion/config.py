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

    # Subject Detection Thresholds
    # Minimum painted-pixel count before running neural subject detection (avoids wasting
    # GPU time on single-pixel or micro-stroke accidental triggers)
    SUBJECT_DETECT_MIN_ZONE_PX: int = 50

    # GrabCut refinement iteration count — 4 iterations give a good accuracy/speed tradeoff
    GRABCUT_ITERATIONS: int = 4

    # Max dimension for the neural segmentation local crop (higher = more accurate, slower)
    NEURAL_CROP_MAX_DIM: int = 512

    # Layer thumbnail canvas size (width, height) in pixels
    THUMBNAIL_WIDTH: int = 56
    THUMBNAIL_HEIGHT: int = 42

    # Optional model weights (MI-GAN, Stable Diffusion) downloaded by install.bat. Repo-root
    # "models/" (gitignored), outside companion/ so the release zip never ships weights.
    MODELS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

    # Optional assistant (companion/agent) defaults; user choices live in QSettings
    AGENT_MAX_STEPS: int = 8                 # model calls per request
    AGENT_TOKEN_BUDGET: int = 20000          # input + output tokens per request
    AGENT_TIMEOUT_S: float = 120.0           # wall clock per request
    AGENT_TOOL_OUTPUT_CAP: int = 2000        # max chars of one tool result sent to the model


# Global singleton configuration instance
CONFIG = AppConfig()
