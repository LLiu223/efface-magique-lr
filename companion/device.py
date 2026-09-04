"""
device.py
Efface Magique LR - Hardware Detection, Dynamic Device Fallback & Memory Management

Provides safe cross-hardware detection (NVIDIA CUDA -> Apple Silicon MPS -> CPU),
automatic multi-threading configuration for CPUs, cuDNN/TF32 tuning for GPUs,
and memory garbage collection.
"""

import gc
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import torch

logger = logging.getLogger("EffaceMagiqueDevice")


class DeviceType(str, Enum):
    """Supported deep learning hardware compute backends."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass(frozen=True)
class DeviceInfo:
    """Detailed hardware metadata and memory statistics for the active device."""
    device: torch.device
    device_type: DeviceType
    name: str
    is_gpu: bool
    vram_total_gb: Optional[float] = None
    vram_reserved_gb: Optional[float] = None
    cpu_threads: Optional[int] = None


def configure_cpu_threading(num_threads: Optional[int] = None) -> int:
    """
    Configure optimal CPU thread parallelism for PyTorch and OpenCV.
    Ensures maximum inference speed without saturating background OS processes.
    """
    total_cores = os.cpu_count() or 1
    if num_threads is not None and num_threads > 0:
        optimal_threads = int(num_threads)
    else:
        optimal_threads = min(8, max(1, total_cores))
    try:
        torch.set_num_threads(optimal_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(4, total_cores // 2)))
        cv2.setNumThreads(optimal_threads)
        logger.info(f"Configured CPU multi-threading: {optimal_threads} worker threads (system has {total_cores} cores)")
    except Exception as e:
        logger.warning(f"Failed to tune CPU thread settings: {e}")
    return optimal_threads


def setup_device_optimizations(device: torch.device):
    """Enable hardware-specific acceleration flags for the detected device."""
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"NVIDIA CUDA optimizations enabled (cuDNN benchmark=True, TF32=True) on {torch.cuda.get_device_name(0)}")
        except Exception as e:
            logger.warning(f"Failed to set CUDA optimization flags: {e}")
    elif device.type == "cpu":
        configure_cpu_threading()


def get_optimal_device() -> torch.device:
    """
    Detect and return the best available compute device with dynamic fallback:
    1. NVIDIA CUDA (with FP16 and cuDNN auto-tuning)
    2. Apple Silicon MPS (Metal Performance Shaders)
    3. Multi-threaded CPU fallback (clean, safe, non-crashing)
    """
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            setup_device_optimizations(device)
            logger.info(f"Active compute hardware: NVIDIA CUDA GPU ({device_name})")
            return device
        except Exception as e:
            logger.warning(f"CUDA initialization error ({e}); falling back to CPU.")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            logger.info("Active compute hardware: Apple Silicon Metal (MPS)")
            return device
        except Exception as e:
            logger.warning(f"Apple MPS initialization error ({e}); falling back to CPU.")

    device = torch.device("cpu")
    setup_device_optimizations(device)
    logger.info("Active compute hardware: Multi-threaded CPU fallback")
    return device


def get_device_info(device: Optional[torch.device] = None) -> DeviceInfo:
    """Gather comprehensive hardware telemetry and memory statistics."""
    dev = device or get_optimal_device()
    if dev.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        try:
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
            return DeviceInfo(
                device=dev,
                device_type=DeviceType.CUDA,
                name=gpu_name,
                is_gpu=True,
                vram_total_gb=total_gb,
                vram_reserved_gb=reserved_gb,
            )
        except Exception:
            return DeviceInfo(device=dev, device_type=DeviceType.CUDA, name=gpu_name, is_gpu=True)
    elif dev.type == "mps":
        return DeviceInfo(device=dev, device_type=DeviceType.MPS, name="Apple Silicon Metal", is_gpu=True)
    else:
        threads = torch.get_num_threads() if hasattr(torch, "get_num_threads") else (os.cpu_count() or 1)
        return DeviceInfo(
            device=dev,
            device_type=DeviceType.CPU,
            name=f"CPU (Multi-threaded x{threads})",
            is_gpu=False,
            cpu_threads=threads,
        )


def get_device_telemetry(device: torch.device) -> str:
    """Return formatted status bar telemetry string for the active device."""
    info = get_device_info(device)
    if info.device_type == DeviceType.CUDA:
        if info.vram_reserved_gb is not None and info.vram_total_gb is not None:
            return f"GPU: {info.name} (FP16 / TF32) | VRAM: {info.vram_reserved_gb:.1f}/{info.vram_total_gb:.1f} GB"
        return f"GPU: {info.name} (FP16 / Tensor Cores)"
    elif info.device_type == DeviceType.MPS:
        return "Apple Silicon Metal (MPS Accelerated)"
    return f"CPU ({info.cpu_threads or (os.cpu_count() or 1)} threads)"


def free_device_memory(device: Optional[torch.device] = None):
    """
    Explicitly free unused PyTorch intermediate tensors and run garbage collection.
    Prevents out-of-memory (OOM) leaks during long editing sessions on low-RAM laptops.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception as e:
            logger.debug(f"CUDA empty_cache error: {e}")
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception as e:
            logger.debug(f"MPS empty_cache error: {e}")
