"""Engine cleanup regressions: Fast mode fails loudly without LaMa, load_model is thread-safe."""

import sys
import threading
import time
import types

import pytest
import torch
from PIL import Image

from companion.model_engine import InpaintingEngine, EngineMode


def _image_and_mask():
    img = Image.new("RGB", (64, 64), (120, 130, 140))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (20, 20, 40, 40))
    return img, mask


def test_fast_mode_without_lama_raises():
    engine = InpaintingEngine(device=torch.device("cpu"), mode=EngineMode.FAST)
    engine._is_loaded = True
    engine._model = None
    img, mask = _image_and_mask()
    with pytest.raises(RuntimeError, match="Fast mode unavailable"):
        engine.generate_variations(img, mask, num_variations=1)


def test_concurrent_load_model_builds_lama_once(monkeypatch):
    constructions = []

    class FakeSimpleLama:
        def __init__(self, device=None):
            time.sleep(0.05)
            constructions.append(device)

    fake_module = types.ModuleType("simple_lama_inpainting")
    fake_module.SimpleLama = FakeSimpleLama
    monkeypatch.setitem(sys.modules, "simple_lama_inpainting", fake_module)

    engine = InpaintingEngine(device=torch.device("cpu"))
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        engine.load_model()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(constructions) == 1
    assert engine.is_loaded
    assert isinstance(engine._model._model, FakeSimpleLama)
