"""CLIPSeg loader: local files only, lazy and cached, failure cached, availability check. Never downloads."""

import os
import sys
import types

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.utils import clipseg_detector as cd  # noqa: E402


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(cd, "_CLIPSEG", None)
    monkeypatch.setattr(cd, "_LOAD_ERROR", None)
    monkeypatch.setattr(cd, "weight_dir", lambda: str(tmp_path))
    return tmp_path


def _fake_transformers(monkeypatch, fail=False):
    """A stand-in `transformers` module recording every from_pretrained call."""
    calls = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        @classmethod
        def from_pretrained(cls, path, **kw):
            calls.append(("model", path, kw))
            if fail:
                raise OSError("model.safetensors is corrupt")
            return cls()

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kw):
            calls.append(("processor", path, kw))
            return cls()

    mod = types.ModuleType("transformers")
    mod.CLIPSegForImageSegmentation, mod.CLIPSegProcessor = Model, Processor
    monkeypatch.setitem(sys.modules, "transformers", mod)
    monkeypatch.setattr(cd, "is_available", lambda: True)
    return calls


def test_loads_lazily_once_from_local_files_only(fresh, monkeypatch):
    calls = _fake_transformers(monkeypatch)
    assert cd.status() == "installed" and calls == []  # nothing loads before the first query
    proc, model = cd.get_clipseg()
    assert cd.get_clipseg()[1] is model  # cached
    assert len(calls) == 2
    for _, path, kw in calls:
        assert path == str(fresh) and kw.get("local_files_only") is True
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert next(model.parameters()).device.type == expected and not model.training
    assert cd.status() == "loaded"


def test_load_failure_is_cached_and_reported(fresh, monkeypatch, caplog):
    calls = _fake_transformers(monkeypatch, fail=True)
    assert cd.get_clipseg() is None and cd.get_clipseg() is None
    assert len([c for c in calls if c[0] == "model"]) == 1  # no retry per call
    assert sum("Could not load CLIPSeg" in r.message for r in caplog.records) == 1
    assert cd.status().startswith("failed (OSError: model.safetensors is corrupt")
    assert "failed to load" in cd.unavailable_note()


def test_is_available_needs_every_file(fresh, monkeypatch):
    monkeypatch.setattr(cd.importlib.util, "find_spec", lambda name: object())
    assert not cd.is_available() and cd.status() == "not installed"
    assert cd.get_clipseg() is None
    for f in cd.WEIGHT_FILES:
        (fresh / f).write_bytes(b"x")
    assert cd.is_available() and cd.status() == "installed"
    monkeypatch.setattr(cd.importlib.util, "find_spec", lambda name: None)
    assert not cd.is_available()
    assert "re-run install.bat" in cd.unavailable_note()
