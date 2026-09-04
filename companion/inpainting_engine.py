"""
inpainting_engine.py
Efface Magique LR - Inpainting Engine Interface

Backward-compatibility facade re-exporting from companion.model_engine and companion.device.
"""

from companion.device import (
    get_optimal_device,
    setup_device_optimizations as setup_gpu_optimizations,
    setup_device_optimizations,
    get_device_telemetry,
    free_device_memory,
    DeviceType,
)
from companion.model_engine import (
    EngineMode,
    InpaintingEngine,
    InpaintingWorker,
    logger,
)

__all__ = [
    "get_optimal_device",
    "setup_gpu_optimizations",
    "setup_device_optimizations",
    "get_device_telemetry",
    "free_device_memory",
    "DeviceType",
    "EngineMode",
    "InpaintingEngine",
    "InpaintingWorker",
    "logger",
]
