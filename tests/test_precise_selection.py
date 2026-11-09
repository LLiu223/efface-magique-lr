"""Job 023: Grounding DINO -> SAM 2 -> ViTMatte, each optional with a fallback; qwen3:8b default; install.bat steps.
Fakes only: no weights, no network."""

import json
import os
import re
import sys
import types

import pytest
import torch
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

pytest.importorskip("langchain.agents")

import numpy as np  # noqa: E402

from agent_fakes import (PERSON, FakeClipSegModel, FakeClipSegProcessor, FakeEngine, FakeGdinoModel,  # noqa: E402
                         FakeGdinoProcessor, FakeSamModel, FakeSamProcessor, FakeSegNet, FakeVitMatteModel,
                         FakeVitMatteProcessor, dark)
from companion.agent import tools as agent_tools  # noqa: E402
from companion.agent.harness import _fallback_select, photo_context  # noqa: E402
from companion.agent.tools import AgentSession  # noqa: E402
from companion.utils import precise_models as pm  # noqa: E402


def _people(w=600, h=400, strand=False):
    """Two black 'people' (ellipses) on white, left and right; optionally a 1-px hair strand off the left one."""
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((60, 80, 180, 360), fill=(0, 0, 0))
    d.ellipse((420, 100, 540, 360), fill=(0, 0, 0))
    if strand:
        d.line((120, 80, 120, 70), fill=(0, 0, 0), width=1)  # 10 px straight up from the head
    return img


def _install(monkeypatch, gdino=None, sam=None, matte=None):
    """Make stages present: pass (processor, model) tuples; None leaves the stage not installed."""
    for stage, value in ((pm.GROUNDING_DINO, gdino), (pm.SAM2, sam), (pm.VITMATTE, matte)):
        if value == "missing":
            monkeypatch.setattr(stage, "get", lambda: None)
        elif value is not None:
            monkeypatch.setattr(stage, "get", lambda v=value: v)
            monkeypatch.setattr(stage, "is_available", lambda: True)


@pytest.fixture
def no_clipseg(monkeypatch):
    from companion.utils import clipseg_detector as cd
    monkeypatch.setattr(cd, "_CLIPSEG", None)
    monkeypatch.setattr(cd, "_LOAD_ERROR", None)
    monkeypatch.setattr(cd, "is_available", lambda: False)
    return cd


@pytest.fixture
def clipseg(monkeypatch):
    proc, model = FakeClipSegProcessor(), FakeClipSegModel(np.zeros((352, 352)))
    monkeypatch.setattr(agent_tools.clipseg_detector, "get_clipseg", lambda: (proc, model))
    return model, proc


TWO_PEOPLE = [("person", 0.1, 0.2, 0.3, 0.9, 0.8), ("person. [SEP]", 0.7, 0.25, 0.9, 0.9, 0.7),  # raw span: cleaned to "person"
              ("person", 0.105, 0.2, 0.3, 0.9, 0.5),   # the left one found twice: merged by IoU
              ("person", 0.4, 0.1, 0.5, 0.2, 0.2)]      # below GDINO_BOX_THRESHOLD


# --- find: Grounding DINO ---------------------------------------------------------------------------

def test_two_people_give_two_regions_with_positions(monkeypatch, no_clipseg):
    gp, sp = FakeGdinoProcessor(TWO_PEOPLE), FakeSamProcessor()
    _install(monkeypatch, gdino=(gp, FakeGdinoModel()), sam=(sp, FakeSamModel()))
    s = AgentSession(engine=FakeEngine(), image=_people())
    out = json.loads(s.detect_regions("Two People."))
    assert [(r["id"], r["label"], r["position"]) for r in out["regions"]] == [
        ("r1", "person", "left-middle"), ("r2", "person", "right-middle")]
    assert "note" not in out and s.detector == "grounding_dino"
    assert gp.texts == ["two people."]  # sanitised, lowercased, one trailing "."
    assert gp.thresholds == [(agent_tools.GDINO_BOX_THRESHOLD, agent_tools.GDINO_TEXT_THRESHOLD)] == [(0.35, 0.25)]
    assert len(sp.calls) == 2  # one SAM pass per instance, not per duplicate box


def test_gdino_regions_are_sam_outlines_not_boxes(monkeypatch, no_clipseg):
    img = _people()
    _install(monkeypatch, gdino=(FakeGdinoProcessor(TWO_PEOPLE[:1]), FakeGdinoModel()),
             sam=(FakeSamProcessor(), FakeSamModel()))
    s = AgentSession(engine=FakeEngine(), image=img)
    json.loads(s.detect_regions("person"))
    m = s.region_masks["r1"] > 0
    truth = dark(img) & (np.arange(600) < 300)[None, :]
    assert m[220, 120] and not m[85, 65]  # inside the ellipse; a box corner is not selected
    assert np.count_nonzero(m & truth) > 0.98 * np.count_nonzero(truth)
    assert np.count_nonzero(m & ~truth) < 0.1 * np.count_nonzero(truth)  # only the thin EDGE_MARGIN


def test_sam_runs_on_a_padded_crop_around_the_box(monkeypatch, no_clipseg):
    sp = FakeSamProcessor()
    _install(monkeypatch, gdino=(FakeGdinoProcessor(TWO_PEOPLE[:1]), FakeGdinoModel()), sam=(sp, FakeSamModel()))
    AgentSession(engine=FakeEngine(), image=_people()).detect_regions("person")
    (cw, ch), (x0, y0, x1, y1), _ = sp.calls[0]
    assert x0 > 0 and y0 > 0 and x1 < cw and y1 < ch  # context around the box
    assert cw < 600 and ch <= 400                       # a crop, not the whole photo


def test_photo_context_and_auto_select_know_grounding_dino(monkeypatch, no_clipseg):
    _install(monkeypatch, gdino=(FakeGdinoProcessor(TWO_PEOPLE), FakeGdinoModel()),
             sam=(FakeSamProcessor(), FakeSamModel()))
    s = AgentSession(engine=FakeEngine(), image=_people())
    assert "any named object, one region per instance" in photo_context(s)
    s.detect_regions("person")
    assert _fallback_select(s, "remove them") and s.selection.region_ids == ["r1", "r2"]


# --- outline: SAM 2 on CLIPSeg / VOC regions --------------------------------------------------------

def test_sam_outline_replaces_the_coarse_voc_mask(monkeypatch, no_clipseg):
    net = FakeSegNet([(PERSON, 0.05, 0.35)])  # a rectangle around the left person
    monkeypatch.setattr(agent_tools, "get_segmentation_model", lambda *a, **k: net)
    monkeypatch.setattr(agent_tools, "refine_region", lambda *a: pytest.fail("SAM present: refine_region not used"))
    sp = FakeSamProcessor()
    _install(monkeypatch, sam=(sp, FakeSamModel()))
    img = _people()
    s = AgentSession(engine=FakeEngine(), image=img)
    out = json.loads(s.detect_regions())
    assert [r["label"] for r in out["regions"]] == ["person"] and s.detector == "voc"
    m = s.region_masks["r1"] > 0
    assert m[220, 120] and not m[110, 40]  # rectangle corner (coarse) gone, the ellipse kept
    _, _, points = sp.calls[0]
    assert 1 <= len(points) <= agent_tools.SAM_MAX_POINTS


def test_interior_points_are_inside_and_apart():
    mask = np.zeros((200, 400), np.uint8)
    mask[50:150, 20:380] = 255
    pts = agent_tools.interior_points(mask)
    assert len(pts) == 3 and all(mask[y, x] for x, y in pts)
    assert min(abs(a[0] - b[0]) for i, a in enumerate(pts) for b in pts[i + 1:]) > 50


def test_thin_wire_keeps_refine_result_when_sam_misses_it(monkeypatch, clipseg, no_clipseg):
    """SAM on a long thin box can return a blob next to the wire: the 30 % coverage guard falls back."""
    model, _ = clipseg
    heat = np.zeros((352, 352))
    heat[100:103, :] = 0.9
    model.heat = heat
    blob = lambda crop: np.pad(np.ones((4, 4), bool), ((0, crop.height - 4), (0, crop.width - 4)))  # noqa: E731
    _install(monkeypatch, sam=(FakeSamProcessor(segment=blob), FakeSamModel()))
    refined = []
    real = agent_tools.refine_region
    monkeypatch.setattr(agent_tools, "refine_region", lambda *a: refined.append(1) or real(*a))
    s = AgentSession(engine=FakeEngine(), image=Image.new("RGB", (352, 352)))
    out = json.loads(s.detect_regions("power lines"))
    assert len(out["regions"]) == 1 and refined == [1]
    assert s.region_masks["r1"][101, 200] and s.region_masks["r1"][101, 20]  # the whole wire is still selected


# --- fallback order -----------------------------------------------------------------------------------

def test_gdino_missing_uses_clipseg(monkeypatch, clipseg):
    model, _ = clipseg
    model.heat = np.full((352, 352), 0.9)
    _install(monkeypatch, sam=(FakeSamProcessor(segment=lambda c: np.ones(c.size[::-1], bool)), FakeSamModel()))
    s = AgentSession(engine=FakeEngine(), image=_people())
    out = json.loads(s.detect_regions("sign"))
    assert s.detector == "clipseg" and out["regions"][0]["label"] == "sign"


def test_gdino_load_failure_falls_back_to_clipseg_then_voc(monkeypatch, clipseg, caplog):
    model, _ = clipseg
    model.heat = np.full((352, 352), 0.9)
    _install(monkeypatch, sam=(FakeSamProcessor(), FakeSamModel()))
    monkeypatch.setattr(pm.GROUNDING_DINO, "is_available", lambda: True)
    monkeypatch.setattr(pm.GROUNDING_DINO, "_loader", lambda path: (_ for _ in ()).throw(OSError("corrupt")))
    s = AgentSession(engine=FakeEngine(), image=_people())
    s.detect_regions("sign")
    s.detect_regions("sign")
    assert s.detector == "clipseg" and pm.GROUNDING_DINO.status() == "failed (OSError: corrupt)"
    assert sum("Could not load Grounding DINO" in r.message for r in caplog.records) == 1
    assert "any named object" not in photo_context(s)


def test_gdino_without_sam_is_not_used(monkeypatch, clipseg):
    model, _ = clipseg
    model.heat = np.full((352, 352), 0.9)
    _install(monkeypatch, gdino=(FakeGdinoProcessor(TWO_PEOPLE), FakeGdinoModel()))
    s = AgentSession(engine=FakeEngine(), image=_people())
    s.detect_regions("person")
    assert s.detector == "clipseg"  # boxes alone are not an outline


def test_nothing_installed_is_todays_voc_path_with_its_note(monkeypatch, no_clipseg):
    net = FakeSegNet([(PERSON, 0.0, 0.3)])
    monkeypatch.setattr(agent_tools, "get_segmentation_model", lambda *a, **k: net)
    called = []
    real = agent_tools.refine_region
    monkeypatch.setattr(agent_tools, "refine_region", lambda *a: called.append(1) or real(*a))
    s = AgentSession(engine=FakeEngine(), image=_people())
    out = json.loads(s.detect_regions("person"))
    assert s.detector == "voc" and called == [1] and "re-run install.bat" in out["note"]
    assert all(m.status() == "not installed" for m in pm.STAGES.values())


# --- soft edges: ViTMatte -----------------------------------------------------------------------------

def _left_person(monkeypatch, matte):
    _install(monkeypatch, gdino=(FakeGdinoProcessor(TWO_PEOPLE[:1]), FakeGdinoModel()),
             sam=(FakeSamProcessor(), FakeSamModel()), matte=matte)
    s = AgentSession(engine=FakeEngine(), image=_people(strand=True))
    s.detect_regions("person")
    return s.region_masks["r1"] > 0


def test_matte_keeps_a_one_pixel_strand_beyond_the_sam_edge(monkeypatch, no_clipseg):
    vp = FakeVitMatteProcessor()
    with_matte = _left_person(monkeypatch, (vp, FakeVitMatteModel()))
    tri = vp.trimaps[0]
    assert set(np.unique(tri)) == {0, 128, 255}
    assert with_matte[76, 120] and not with_matte[76, 110]  # the strand 4 px above the head, not beside it
    assert not _left_person(monkeypatch, "missing")[76, 120]  # without ViTMatte the strand is missed (skip, no error)


def test_matte_never_runs_for_power_lines(monkeypatch, clipseg):
    model, _ = clipseg
    heat = np.zeros((352, 352))
    heat[100:103, :] = 0.9
    model.heat = heat
    _install(monkeypatch, sam=(FakeSamProcessor(segment=lambda c: np.ones(c.size[::-1], bool)), FakeSamModel()))
    monkeypatch.setattr(pm.VITMATTE, "get", lambda: pytest.fail("ViTMatte must not load for power lines"))
    s = AgentSession(engine=FakeEngine(), image=Image.new("RGB", (352, 352)))
    assert json.loads(s.detect_regions("power lines"))["regions"]
    assert agent_tools.wants_matte("the woman on the left") and agent_tools.wants_matte("dogs")
    assert not agent_tools.wants_matte("power lines") and not agent_tools.wants_matte("manhole")


# --- loaders ------------------------------------------------------------------------------------------

def _fake_transformers(monkeypatch, fail=False):
    calls = []

    def model_cls(name):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            @classmethod
            def from_pretrained(cls, path, **kw):
                calls.append((name, path, kw))
                if fail:
                    raise OSError("bad weights")
                return cls()
        return Model

    def proc_cls(name):
        class Processor:
            @classmethod
            def from_pretrained(cls, path, **kw):
                calls.append((name, path, kw))
                return cls()
        return Processor

    mod = types.ModuleType("transformers")
    for n in ("GroundingDinoForObjectDetection", "Sam2Model", "VitMatteForImageMatting"):
        setattr(mod, n, model_cls(n))
    for n in ("GroundingDinoProcessor", "Sam2Processor", "VitMatteImageProcessor"):
        setattr(mod, n, proc_cls(n))
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return calls


@pytest.mark.parametrize("name", list(pm.STAGES))
def test_loader_lazy_cached_local_only_with_status(name, monkeypatch, tmp_path):
    stage = pm.STAGES[name]
    calls = _fake_transformers(monkeypatch)
    monkeypatch.setattr(stage, "weight_dir", lambda: str(tmp_path))
    monkeypatch.setattr(stage, "is_available", lambda: True)
    assert stage.status() == "installed" and calls == []
    proc, model = stage.get()
    assert stage.get()[1] is model and len(calls) == 2
    assert all(kw.get("local_files_only") is True and path == str(tmp_path) for _, path, kw in calls)
    p = next(model.parameters())
    cuda = torch.cuda.is_available()
    assert p.device.type == ("cuda" if cuda else "cpu") and not model.training
    assert p.dtype == (torch.float16 if cuda and name == "sam2" else torch.float32)
    assert stage.status() == "loaded"


def test_loader_failure_logged_once_and_cached(monkeypatch, tmp_path, caplog):
    calls = _fake_transformers(monkeypatch, fail=True)
    monkeypatch.setattr(pm.SAM2, "is_available", lambda: True)
    assert pm.SAM2.get() is None and pm.SAM2.get() is None
    assert len([c for c in calls if c[0] == "Sam2Model"]) == 1
    assert sum("Could not load SAM 2.1" in r.message for r in caplog.records) == 1
    assert pm.SAM2.status() == "failed (OSError: bad weights)"


def test_is_available_needs_every_file(monkeypatch, tmp_path):
    monkeypatch.setattr(pm.VITMATTE, "weight_dir", lambda: str(tmp_path))
    monkeypatch.setattr(pm.importlib.util, "find_spec", lambda name: object())
    assert not pm.VITMATTE.is_available()
    for f in pm.VITMATTE.WEIGHT_FILES:
        (tmp_path / f).write_bytes(b"x")
    assert pm.VITMATTE.is_available() and pm.VITMATTE.status() == "installed"
    assert [m.WEIGHT_SUBDIR for m in pm.STAGES.values()] == ["grounding_dino", "sam2", "vitmatte"]


# --- install.bat ----------------------------------------------------------------------------------------

def _install_bat() -> str:
    with open(os.path.join(_PROJECT_ROOT, "install.bat"), "rb") as f:
        return f.read().decode("ascii")


@pytest.mark.parametrize("name,var,repo", [("grounding_dino", "GD_URL", "IDEA-Research/grounding-dino-tiny"),
                                           ("sam2", "SAM_URL", "facebook/sam2.1-hiera-small"),
                                           ("vitmatte", "VM_URL", "hustvl/vitmatte-small-composition-1k")])
def test_install_bat_fetches_every_precise_file_pinned_and_hashed(name, var, repo):
    text = _install_bat()
    assert re.search(rf'set "{var}=https://huggingface\.co/{re.escape(repo)}/resolve/[0-9a-f]{{40}}"', text)
    for f in pm.STAGES[name].WEIGHT_FILES:
        assert re.search(rf'call :FETCH "!{var}!/{re.escape(f)}" "!MODELS_DIR!\\{name}\\{re.escape(f)}" [0-9a-f]{{64}}\r\n',
                         text), f


def test_install_bat_precise_step_is_one_optional_step():
    text = _install_bat()
    assert text.count("Precise selection (Grounding DINO + SAM 2 + ViTMatte, about 980 MB") == 1
    assert 'call :INSTALL_EXTRAS requirements-clipseg.txt "Precise selection"' in text
    assert "trust_remote_code" not in text


def test_install_bat_offers_ollama_pull_only_when_ollama_exists():
    text = _install_bat()
    i = text.index("where ollama")
    assert "ollama pull qwen3:8b" in text[i:] and text.count("where ollama") == 1
    assert "ollama.com/download" not in text and "OllamaSetup" not in text  # never installs Ollama
