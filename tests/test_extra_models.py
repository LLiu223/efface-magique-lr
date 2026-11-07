"""Job 014: MI-GAN and Stable Diffusion inpainting backends. Fakes only: no network, no GPU, no weights."""

import dataclasses
import hashlib
import importlib.machinery
import importlib.util
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image
from PyQt6.QtCore import QSettings

import companion.model_engine as model_engine
from companion.config import CONFIG
from companion.model_engine import EngineMode, InpaintingEngine
from companion.models import migan, registry, sd_inpaint
from companion.models.migan import MiganModel
from companion.models.sd_inpaint import SdInpaintModel

CPU = torch.device("cpu")


def _fake_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)  # find_spec() returns it instead of raising
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def models_dir(monkeypatch, tmp_path):
    cfg = dataclasses.replace(CONFIG, MODELS_DIR=str(tmp_path))
    monkeypatch.setattr(migan, "CONFIG", cfg)
    monkeypatch.setattr(sd_inpaint, "CONFIG", cfg)
    return tmp_path


def _hide_packages(monkeypatch, *names):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n in names else real(n, *a))


# fakes -------------------------------------------------------------------------

class FakeOrtSession:
    calls = []

    def __init__(self, path, providers=None):
        self.path, self.providers = path, providers

    def run(self, outputs, feeds):
        FakeOrtSession.calls.append({k: v.shape for k, v in feeds.items()})
        return [255 - feeds["image"]]  # invert everything; the model must restore pixels outside the mask


@pytest.fixture
def fake_migan(monkeypatch, models_dir):
    FakeOrtSession.calls = []
    _fake_module(monkeypatch, "onnxruntime", InferenceSession=FakeOrtSession,
                 get_available_providers=lambda: ["CPUExecutionProvider"])
    path = models_dir / migan.WEIGHT_FILE
    path.write_bytes(b"fake onnx")
    monkeypatch.setattr(migan, "WEIGHT_SHA256", hashlib.sha256(b"fake onnx").hexdigest())
    return path


class FakePipe:
    instances = []

    def __init__(self, kwargs):
        self.kwargs, self.calls, self.device, self.sliced = kwargs, [], None, False
        FakePipe.instances.append(self)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls(dict(kwargs, path=path))

    def to(self, device):
        self.device = device
        return self

    def enable_attention_slicing(self):
        self.sliced = True

    def set_progress_bar_config(self, **kw):
        pass

    def __call__(self, prompt, image, mask_image, height, width, num_inference_steps, generator):
        self.calls.append(dict(prompt=prompt, size=image.size, mask_size=mask_image.size, height=height,
                               width=width, steps=num_inference_steps, seed=generator.initial_seed()))
        return types.SimpleNamespace(images=[Image.new("RGB", (width, height), (1, 2, 3))])


@pytest.fixture
def fake_sd(monkeypatch, models_dir):
    FakePipe.instances = []
    _fake_module(monkeypatch, "diffusers", StableDiffusionInpaintPipeline=FakePipe)
    _fake_module(monkeypatch, "transformers")
    _fake_module(monkeypatch, "accelerate")
    for rel in sd_inpaint.WEIGHT_FILES:
        f = models_dir / sd_inpaint.WEIGHT_SUBDIR / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    return models_dir


def _crop_and_mask(w, h):
    rng = np.random.default_rng(1)
    crop = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    m = np.zeros((h, w), np.uint8)
    m[h // 4: h // 2, w // 3: w // 2] = 255
    return crop, Image.fromarray(m)


def _assert_outside_unchanged(crop, mask, out):
    assert out.mode == "RGB" and out.size == crop.size
    keep = np.asarray(mask) <= 10
    assert np.array_equal(np.asarray(out)[keep], np.asarray(crop)[keep])
    assert not np.array_equal(np.asarray(out)[~keep], np.asarray(crop)[~keep])


# registration ---------------------------------------------------------------------

def test_models_registered_with_spec_attributes():
    assert registry.get("migan") is MiganModel and registry.get("sd_inpaint") is SdInpaintModel
    assert MiganModel.uses_prompt is False
    assert SdInpaintModel.uses_prompt is True and SdInpaintModel.steps == 20 and SdInpaintModel.max_dim_cpu == 512


# availability (AC4) -------------------------------------------------------------

def test_migan_unavailable_reasons(monkeypatch, models_dir):
    _hide_packages(monkeypatch, "onnxruntime")
    assert not MiganModel.is_available()
    assert "onnxruntime" in MiganModel.unavailable_reason()
    assert "requirements-extra.txt" in MiganModel.unavailable_reason()


def test_migan_missing_weight_file(monkeypatch, models_dir):
    _fake_module(monkeypatch, "onnxruntime")
    assert not MiganModel.is_available()
    reason = MiganModel.unavailable_reason()
    assert migan.WEIGHT_FILE in reason and "install.bat" in reason
    (models_dir / migan.WEIGHT_FILE).write_bytes(b"x")
    assert MiganModel.is_available()


def test_sd_unavailable_names_missing_package(monkeypatch, fake_sd):
    assert SdInpaintModel.is_available()
    _hide_packages(monkeypatch, "accelerate")
    assert not SdInpaintModel.is_available()
    reason = SdInpaintModel.unavailable_reason()
    assert "accelerate" in reason and "diffusers" not in reason and "requirements-extra.txt" in reason


def test_sd_unavailable_names_missing_weight_file(fake_sd):
    (fake_sd / "sd_inpaint" / "unet" / "diffusion_pytorch_model.fp16.safetensors").unlink()
    assert not SdInpaintModel.is_available()
    reason = SdInpaintModel.unavailable_reason()
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in reason and "vae/" not in reason
    assert "install.bat" in reason


def test_is_available_imports_nothing(monkeypatch, models_dir):
    before = set(sys.modules)
    MiganModel.is_available(), SdInpaintModel.is_available()
    MiganModel.unavailable_reason(), SdInpaintModel.unavailable_reason()
    assert not {"onnxruntime", "diffusers", "transformers", "accelerate"} & (set(sys.modules) - before)


# missing-dependency error from load() -------------------------------------------------

def test_load_raises_missing_dependency_text(monkeypatch, models_dir):
    _hide_packages(monkeypatch, "diffusers")
    with pytest.raises(ImportError, match=r"diffusers.*requirements-extra\.txt"):
        SdInpaintModel().load(CPU)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # makes `import onnxruntime` fail
    with pytest.raises(ImportError, match=r"onnxruntime.*requirements-extra\.txt"):
        MiganModel().load(CPU)


def test_migan_load_rejects_bad_checksum(fake_migan, monkeypatch):
    monkeypatch.setattr(migan, "WEIGHT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        MiganModel().load(CPU)


# load / unload order (one model in VRAM) --------------------------------------------------

def test_engine_load_unload_order(fake_migan, fake_sd, monkeypatch):
    monkeypatch.setattr(model_engine, "free_device_memory", lambda d: None)
    engine = InpaintingEngine(device=CPU)
    engine.set_model("migan")
    engine.load_model()
    m = engine._model
    assert isinstance(m, MiganModel) and m._session is not None

    engine.set_model("sd_inpaint")
    assert m._session is None and engine._model is None  # unloaded before the next one loads
    engine.load_model()
    sd = engine._model
    assert isinstance(sd, SdInpaintModel) and sd._pipe is FakePipe.instances[-1]

    engine.set_model("lama")
    assert sd._pipe is None and engine._model is None


def test_sd_load_options_cpu(fake_sd):
    msgs = []
    model = SdInpaintModel()
    model.load(CPU, msgs.append)
    kw = model._pipe.kwargs
    assert msgs[0].startswith("Stable Diffusion on CPU: this can take several minutes")
    assert kw["local_files_only"] is True and kw["use_safetensors"] is True and kw["variant"] == "fp16"
    assert kw["torch_dtype"] == torch.float32 and kw["safety_checker"] is None
    assert not model._pipe.sliced


def test_sd_load_options_cuda(fake_sd):
    msgs = []
    model = SdInpaintModel()
    model.load(torch.device("cuda"), msgs.append)
    assert model._pipe.kwargs["torch_dtype"] == torch.float16 and model._pipe.sliced
    assert not any("CPU" in m for m in msgs)


def test_engine_forwards_cpu_warning_to_worker_progress(fake_sd, monkeypatch):
    monkeypatch.setattr(model_engine, "free_device_memory", lambda d: None)
    engine = InpaintingEngine(device=CPU, mode=EngineMode.FAST)
    engine.set_model("sd_inpaint")
    crop, mask = _crop_and_mask(300, 200)
    msgs = []
    engine.generate_variations(crop, mask, num_variations=1, progress_callback=lambda p, m: msgs.append(m))
    assert any(m.startswith("Stable Diffusion on CPU") for m in msgs)


# size round trip (AC6) -------------------------------------------------------------------

@pytest.mark.parametrize("w,h", [(333, 219), (1301, 601), (7, 9)])
def test_sd_size_round_trip(fake_sd, w, h):
    model = SdInpaintModel()
    model.load(CPU)
    crop, mask = _crop_and_mask(w, h)
    out = model.inpaint(crop, mask, seed=1, prompt=None)
    call = model._pipe.calls[-1]
    assert call["width"] % 8 == 0 and call["height"] % 8 == 0
    assert max(call["width"], call["height"]) <= model.max_dim_cpu
    assert call["size"] == call["mask_size"] == (call["width"], call["height"])
    if w >= 16:
        _assert_outside_unchanged(crop, mask, out)
    assert out.size == (w, h)


@pytest.mark.parametrize("w,h", [(333, 219), (2001, 901)])
def test_migan_size_round_trip(fake_migan, w, h):
    model = MiganModel()
    model.load(CPU)
    crop, mask = _crop_and_mask(w, h)
    out = model.inpaint(crop, mask, seed=0, prompt="ignored")
    shapes = FakeOrtSession.calls[-1]
    assert shapes["image"][:2] == (1, 3) and shapes["mask"][:2] == (1, 1)
    assert shapes["image"][2:] == shapes["mask"][2:] and max(shapes["image"][2:]) <= model.max_dim_cpu
    _assert_outside_unchanged(crop, mask, out)


# prompt and seed (AC7) ---------------------------------------------------------------------

def test_sd_prompt_seed_and_steps_reach_pipeline(fake_sd):
    model = SdInpaintModel()
    model.load(CPU)
    crop, mask = _crop_and_mask(64, 64)
    model.inpaint(crop, mask, seed=1234, prompt="a red barn")
    model.inpaint(crop, mask, seed=7, prompt=None)
    first, second = model._pipe.calls[-2:]
    assert (first["prompt"], first["seed"], first["steps"]) == ("a red barn", 1234, 20)
    assert (second["prompt"], second["seed"]) == ("", 7)
    for steps, expected in ((0, 1), (500, 50), (-3, 1)):
        model.steps = steps
        model.inpaint(crop, mask, seed=1, prompt="x")
        assert model._pipe.calls[-1]["steps"] == expected


def test_no_tint_for_prompt_models(fake_sd, monkeypatch):
    seen = []
    real = InpaintingEngine._generate_noise_variation
    monkeypatch.setattr(InpaintingEngine, "_generate_noise_variation",
                        staticmethod(lambda *a, **k: seen.append(k["prompt"]) or real(*a, **k)))
    monkeypatch.setattr(model_engine, "free_device_memory", lambda d: None)
    crop, mask = _crop_and_mask(96, 96)

    engine = InpaintingEngine(device=CPU, mode=EngineMode.FIREFLY)
    engine.set_model("sd_inpaint")
    engine.load_model()
    patch = engine._run_variation_patch(crop, mask, seed=3, prompt="a red barn")  # seed % 3 == 0: no noise
    assert seen[-1] is None
    assert engine._model._pipe.calls[-1]["prompt"] == "a red barn"
    expected = engine._model.inpaint(crop, Image.fromarray(np.asarray(mask)), 3, "a red barn")
    hole = np.asarray(mask) > 10
    assert np.array_equal(np.asarray(patch)[hole], np.asarray(expected)[hole])  # no tint added

    engine._model_cls = registry.get("migan")  # uses_prompt=False keeps the legacy tint
    engine._run_variation_patch(crop, mask, seed=3, prompt="a red barn")
    assert seen[-1] == "a red barn"


# UI prompt field (AC8) -------------------------------------------------------------------

@pytest.fixture
def window(qtbot, monkeypatch, tmp_path, fake_sd):
    import companion.app as app
    path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(app, "_open_settings", lambda: QSettings(path, QSettings.Format.IniFormat))
    monkeypatch.setattr(model_engine, "free_device_memory", lambda d: None)
    w = app.MainWindow()
    qtbot.addWidget(w)
    return w


def test_prompt_field_follows_model_and_run_lock(window, monkeypatch):
    assert window.engine.model_name == "lama" and not window.txt_prompt.isEnabled()
    window.combo_model.setCurrentIndex(window.combo_model.findData("sd_inpaint"))
    assert window.txt_prompt.isEnabled()

    captured = {}
    real_init = model_engine.InpaintingWorker.__init__

    def spy_init(self, *a, **k):
        captured.update(k)
        real_init(self, *a, **k)

    monkeypatch.setattr(model_engine.InpaintingWorker, "__init__", spy_init)
    monkeypatch.setattr(model_engine.InpaintingWorker, "start", lambda self: None)
    window.txt_prompt.setText("  a red barn ")
    img = Image.new("RGB", (64, 64))
    window._start_inpainting_pipeline(img, Image.new("L", (64, 64), 0))
    assert captured["prompt"] == "a red barn"
    assert not window.txt_prompt.isEnabled() and not window.combo_model.isEnabled()
    window._on_inpainting_finished([])
    assert window.txt_prompt.isEnabled()

    window.combo_model.setCurrentIndex(window.combo_model.findData("lama"))
    assert not window.txt_prompt.isEnabled()
    window._start_inpainting_pipeline(img, Image.new("L", (64, 64), 0))
    assert captured["prompt"] is None  # leftover text is not sent to a model without prompts
    window._on_inpainting_finished([])


def test_new_models_in_dropdown_with_tooltip(window, monkeypatch):
    from PyQt6.QtCore import Qt
    combo = window.combo_model
    for name in ("migan", "sd_inpaint"):
        assert combo.findData(name) >= 0
    i = combo.findData("migan")  # fake_sd provides no MI-GAN weights
    assert not (combo.model().item(i).flags() & Qt.ItemFlag.ItemIsEnabled)
    assert combo.itemData(i, Qt.ItemDataRole.ToolTipRole) == MiganModel.unavailable_reason()


# integration (only with real local weights) ---------------------------------------------------

@pytest.mark.skipif(not MiganModel.is_available(), reason="MI-GAN weights/onnxruntime not installed")
def test_migan_integration_real_weights():
    model = MiganModel()
    model.load(CPU)
    crop, mask = _crop_and_mask(301, 203)
    out = model.inpaint(crop, mask, seed=0, prompt=None)
    _assert_outside_unchanged(crop, mask, out)
    model.unload()
    assert model._session is None


@pytest.mark.skipif(not SdInpaintModel.is_available(), reason="Stable Diffusion weights/packages not installed")
def test_sd_integration_real_weights():
    model = SdInpaintModel()
    model.steps = 1
    model.load(CPU)
    crop, mask = _crop_and_mask(67, 53)
    out = model.inpaint(crop, mask, seed=1, prompt="grass")
    assert out.size == crop.size
    model.unload()
    assert model._pipe is None
