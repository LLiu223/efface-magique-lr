"""Fine-tuned LaMa discovery (job 016): models/finetuned/<name>/ -> "LaMa (tuned: <name>)" registry entries."""

import json
import os
import subprocess
import sys
import zipfile

import pytest
import torch
from PIL import Image

from companion.models import registry, DEFAULT_MODEL, tuned
from companion.models.lama import LamaModel
from companion.models.tuned import TunedLamaModel
from companion.model_engine import InpaintingEngine

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = {"name": "x", "date": "2026-11-07T10:00:00", "base_ckpt": "best.ckpt", "manifest_hash": "0" * 64,
        "steps": 10, "lr": 1e-5, "metrics": {}}


class _Fill(torch.nn.Module):
    """big-lama.pt contract: (image [1,3,H,W] 0..1, mask [1,1,H,W]) -> image; fills the hole with mid grey."""

    def forward(self, image, mask):
        m = (mask > 0).to(image.dtype)
        return image * (1 - m) + 0.5 * m


def _make(root, name, model=True, meta=META):
    d = root / name
    d.mkdir(parents=True)
    if model:
        torch.jit.script(_Fill()).save(str(d / "model.pt"))
    if meta is not None:
        (d / "meta.json").write_text(meta if isinstance(meta, str) else json.dumps(meta), encoding="utf-8")
    return d


@pytest.fixture
def reg(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    return registry


def test_valid_folder_is_registered(reg, tmp_path):
    _make(tmp_path, "portraits_v2-1")
    [cls] = tuned.discover(tmp_path)
    assert cls.name == "lama-tuned-portraits_v2-1" and cls.label == "LaMa (tuned: portraits_v2-1)"
    assert issubclass(cls, TunedLamaModel) and issubclass(cls, LamaModel)
    assert reg.get("lama-tuned-portraits_v2-1") is cls
    assert cls.is_available() and cls.unavailable_reason() == ""
    assert reg.all()[0] is LamaModel and DEFAULT_MODEL == "lama"


def test_bad_folder_names_are_skipped(reg, tmp_path):
    for name in ("has space", "dot.name", "café", "semi;colon"):
        (_make(tmp_path, name, model=False) / "model.pt").write_bytes(b"x")  # contents irrelevant: never scanned
    (tmp_path / "loose_file").write_text("x")
    _make(tmp_path, "ok")
    assert [c.name for c in tuned.discover(tmp_path)] == ["lama-tuned-ok"]


def test_missing_root_or_file_root(reg, tmp_path):
    assert tuned.discover(tmp_path / "nope") == []
    (tmp_path / "f").write_text("x")
    assert tuned.discover(tmp_path / "f") == []


@pytest.mark.parametrize("setup, reason", [
    (lambda r: _make(r, "m", model=False), "model.pt is missing"),
    (lambda r: _make(r, "m", meta=None), "meta.json is missing"),
    (lambda r: _make(r, "m", meta="{not json"), "meta.json in"),
    (lambda r: _make(r, "m", meta="[1, 2]"), "not a JSON object"),
    (lambda r: _make(r, "m", meta={k: v for k, v in META.items() if k != "steps"}), "missing steps"),
    (lambda r: (_make(r, "m", model=False) / "model.pt").write_bytes(b"\x00garbage"), "corrupt"),
])
def test_broken_folders_are_unavailable_with_reason(reg, tmp_path, setup, reason):
    setup(tmp_path)
    [cls] = tuned.discover(tmp_path)
    assert cls.is_available() is False
    assert reason in cls.unavailable_reason()
    with pytest.raises(RuntimeError, match="model.pt|meta.json"):
        cls().load(torch.device("cpu"))
    assert reg.get("lama") is LamaModel


def test_unloadable_zip_fails_load_then_greys_out(reg, tmp_path):
    d = _make(tmp_path, "z", model=False)
    with zipfile.ZipFile(d / "model.pt", "w") as z:
        z.writestr("junk.txt", "not torchscript")
    [cls] = tuned.discover(tmp_path)
    assert cls.is_available()  # cheap check passes; only a real load can tell
    engine = InpaintingEngine(device=torch.device("cpu"))
    engine.set_model(cls.name)
    engine.load_model()  # must not raise or hang: existing load-failure state
    assert engine.is_loaded and engine._model is None
    assert cls.is_available() is False and "could not be loaded" in cls.unavailable_reason()


def test_load_uses_path_not_env_and_reuses_inpaint(reg, tmp_path, monkeypatch):
    _make(tmp_path, "grey")
    [cls] = tuned.discover(tmp_path)
    monkeypatch.setenv("LAMA_MODEL", str(tmp_path / "does-not-exist.pt"))
    loaded = []
    real_load = torch.jit.load
    monkeypatch.setattr(torch.jit, "load", lambda p, **kw: loaded.append(p) or real_load(p, **kw))
    m = cls()
    m.load(torch.device("cpu"))
    assert loaded == [str(tmp_path / "grey" / "model.pt")]
    assert os.environ["LAMA_MODEL"] == str(tmp_path / "does-not-exist.pt")
    assert cls.inpaint is LamaModel.inpaint
    crop = Image.new("RGB", (37, 21), (200, 10, 10))
    mask = Image.new("L", (37, 21), 0)
    mask.paste(255, (5, 5, 15, 15))
    out = m.inpaint(crop, mask)
    assert out.size == (37, 21)
    assert out.getpixel((0, 0)) == (200, 10, 10) and out.getpixel((10, 10)) == (127, 127, 127)
    m.unload()
    assert m._model is None


def test_default_dir_is_repo_models_finetuned():
    assert tuned.FINETUNED_DIR == __import__("pathlib").Path(_ROOT).resolve() / "models" / "finetuned"


def test_importing_companion_never_imports_tools_finetune():
    code = ("import sys, companion, companion.models, companion.app\n"
            "bad = [m for m in sys.modules if m == 'tools' or m.startswith('tools.finetune')]\n"
            "assert not bad, bad\nprint('ok')")
    p = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True,
                       env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    assert p.returncode == 0 and "ok" in p.stdout, p.stderr
