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

from agent_fakes import (BOTTLE, PERSON, FakeEngine, FakeGdinoModel,  # noqa: E402
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
def voc(monkeypatch):
    """The VOC fallback sees one person rectangle on the left."""
    net = FakeSegNet([(PERSON, 0.05, 0.35)])
    monkeypatch.setattr(agent_tools, "get_segmentation_model", lambda *a, **k: net)
    return net


class _WireNet(FakeSegNet):
    """VOC logits with one 3-px horizontal 'bottle' band: a thin object in a long box."""

    def forward(self, x):
        _, _, h, w = x.shape
        out = torch.zeros(1, 21, h, w)
        out[:, 0] = 1.0
        out[:, BOTTLE, 100:103, :] = 5.0
        return {"out": out}


TWO_PEOPLE = [("person", 0.1, 0.2, 0.3, 0.9, 0.8), ("person. [SEP]", 0.7, 0.25, 0.9, 0.9, 0.7),  # raw span: cleaned to "person"
              ("person", 0.105, 0.2, 0.3, 0.9, 0.5),   # the left one found twice: merged by IoU
              ("person", 0.4, 0.1, 0.5, 0.2, 0.2)]      # below GDINO_BOX_THRESHOLD


# --- find: Grounding DINO ---------------------------------------------------------------------------

def test_two_people_give_two_regions_with_positions(monkeypatch):
    gp, sp = FakeGdinoProcessor(TWO_PEOPLE), FakeSamProcessor()
    _install(monkeypatch, gdino=(gp, FakeGdinoModel()), sam=(sp, FakeSamModel()))
    s = AgentSession(engine=FakeEngine(), image=_people())
    out = json.loads(s.detect_regions("Two People."))
    assert [(r["id"], r["label"], r["position"]) for r in out["regions"]] == [
        ("r1", "person", "left-middle"), ("r2", "person", "right-middle")]
    assert "note" not in out and s.detector == "grounding_dino"
    held = agent_tools.HELD_QUERY  # then one held-object pass per person; its "person" boxes are skipped
    assert gp.texts == ["two people.", held, held]  # sanitised, lowercased, one trailing "."
    assert gp.thresholds[0] == (agent_tools.GDINO_BOX_THRESHOLD, agent_tools.GDINO_TEXT_THRESHOLD) == (0.35, 0.25)
    assert len(sp.calls) == 2  # one SAM pass per instance, not per duplicate box


def test_gdino_regions_are_sam_outlines_not_boxes(monkeypatch):
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


def test_sam_runs_on_a_padded_crop_around_the_box(monkeypatch):
    sp = FakeSamProcessor()
    _install(monkeypatch, gdino=(FakeGdinoProcessor(TWO_PEOPLE[:1]), FakeGdinoModel()), sam=(sp, FakeSamModel()))
    AgentSession(engine=FakeEngine(), image=_people()).detect_regions("person")
    (cw, ch), (x0, y0, x1, y1), _ = sp.calls[0]
    assert x0 > 0 and y0 > 0 and x1 < cw and y1 < ch  # context around the box
    assert cw < 600 and ch <= 400                       # a crop, not the whole photo


def test_photo_context_and_auto_select_know_grounding_dino(monkeypatch):
    _install(monkeypatch, gdino=(FakeGdinoProcessor(TWO_PEOPLE), FakeGdinoModel()),
             sam=(FakeSamProcessor(), FakeSamModel()))
    s = AgentSession(engine=FakeEngine(), image=_people())
    assert "any named object, one region per instance" in photo_context(s)
    s.detect_regions("person")
    assert _fallback_select(s, "remove them") and s.selection.region_ids == ["r1", "r2"]


# --- outline: SAM 2 on VOC regions --------------------------------------------------------

def test_sam_outline_replaces_the_coarse_voc_mask(monkeypatch):
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


def test_thin_wire_keeps_refine_result_when_sam_misses_it(monkeypatch):
    """SAM on a long thin box can return a blob next to the wire: the 30 % coverage guard falls back."""
    net = _WireNet([])
    monkeypatch.setattr(agent_tools, "get_segmentation_model", lambda *a, **k: net)
    blob = lambda crop: np.pad(np.ones((4, 4), bool), ((0, crop.height - 4), (0, crop.width - 4)))  # noqa: E731
    _install(monkeypatch, sam=(FakeSamProcessor(segment=blob), FakeSamModel()))
    refined = []
    real = agent_tools.refine_region
    monkeypatch.setattr(agent_tools, "refine_region", lambda *a: refined.append(1) or real(*a))
    s = AgentSession(engine=FakeEngine(), image=Image.new("RGB", (352, 352)))
    out = json.loads(s.detect_regions())
    assert len(out["regions"]) == 1 and refined == [1]
    assert s.region_masks["r1"][101, 200] and s.region_masks["r1"][101, 20]  # the whole wire is still selected


# --- fallback order -----------------------------------------------------------------------------------

def test_gdino_missing_uses_voc_outlined_by_sam(monkeypatch, voc):
    _install(monkeypatch, sam=(FakeSamProcessor(), FakeSamModel()))
    s = AgentSession(engine=FakeEngine(), image=_people())
    out = json.loads(s.detect_regions("person"))
    assert s.detector == "voc" and out["regions"][0]["label"] == "person"
    assert "install the optional Precise selection step (install.bat)" in out["note"]


def test_gdino_load_failure_falls_back_to_voc_and_says_failed(monkeypatch, voc, caplog):
    _install(monkeypatch, sam=(FakeSamProcessor(), FakeSamModel()))
    monkeypatch.setattr(pm.GROUNDING_DINO, "is_available", lambda: True)
    monkeypatch.setattr(pm.GROUNDING_DINO, "_loader", lambda path: (_ for _ in ()).throw(OSError("corrupt")))
    s = AgentSession(engine=FakeEngine(), image=_people())
    s.detect_regions("sign")
    out = json.loads(s.detect_regions("sign"))
    assert s.detector == "voc" and pm.GROUNDING_DINO.status() == "failed (OSError: corrupt)"
    assert "Precise selection failed" in out["note"]
    assert sum("Could not load Grounding DINO" in r.message for r in caplog.records) == 1
    assert "any named object" not in photo_context(s)


def test_gdino_without_sam_is_not_used(monkeypatch, voc):
    _install(monkeypatch, gdino=(FakeGdinoProcessor(TWO_PEOPLE), FakeGdinoModel()))
    s = AgentSession(engine=FakeEngine(), image=_people())
    s.detect_regions("person")
    assert s.detector == "voc"  # boxes alone are not an outline
    assert "any named object" not in photo_context(s)


def test_nothing_installed_is_todays_voc_path_with_its_note(monkeypatch):
    net = FakeSegNet([(PERSON, 0.0, 0.3)])
    monkeypatch.setattr(agent_tools, "get_segmentation_model", lambda *a, **k: net)
    called = []
    real = agent_tools.refine_region
    monkeypatch.setattr(agent_tools, "refine_region", lambda *a: called.append(1) or real(*a))
    s = AgentSession(engine=FakeEngine(), image=_people())
    out = json.loads(s.detect_regions("person"))
    assert s.detector == "voc" and called == [1] and "Precise selection step (install.bat)" in out["note"]
    assert all(m.status() == "not installed" for m in pm.STAGES.values())


# --- soft edges: ViTMatte -----------------------------------------------------------------------------

def _left_person(monkeypatch, matte):
    _install(monkeypatch, gdino=(FakeGdinoProcessor(TWO_PEOPLE[:1]), FakeGdinoModel()),
             sam=(FakeSamProcessor(), FakeSamModel()), matte=matte)
    s = AgentSession(engine=FakeEngine(), image=_people(strand=True))
    s.detect_regions("person")
    return s.region_masks["r1"] > 0


def test_matte_keeps_a_one_pixel_strand_beyond_the_sam_edge(monkeypatch):
    vp = FakeVitMatteProcessor()
    with_matte = _left_person(monkeypatch, (vp, FakeVitMatteModel()))
    tri = vp.trimaps[0]
    assert set(np.unique(tri)) == {0, 128, 255}
    assert with_matte[76, 120] and not with_matte[76, 110]  # the strand 4 px above the head, not beside it
    assert not _left_person(monkeypatch, "missing")[76, 120]  # without ViTMatte the strand is missed (skip, no error)


def test_matte_never_runs_for_power_lines(monkeypatch):
    _install(monkeypatch, gdino=(FakeGdinoProcessor([("power lines", 0.0, 0.28, 1.0, 0.3, 0.9)]), FakeGdinoModel()),
             sam=(FakeSamProcessor(segment=lambda c: np.ones(c.size[::-1], bool)), FakeSamModel()))
    monkeypatch.setattr(pm.VITMATTE, "get", lambda: pytest.fail("ViTMatte must not load for power lines"))
    s = AgentSession(engine=FakeEngine(), image=Image.new("RGB", (352, 352)))
    assert json.loads(s.detect_regions("power lines"))["regions"][0]["label"] == "power lines"
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


def test_install_bat_has_no_stray_control_bytes():
    """A "\\t" or "\\v" expanded while writing the file once turned a models\\...\\tokenizer_config.json path into a TAB."""
    with open(os.path.join(_PROJECT_ROOT, "install.bat"), "rb") as f:
        raw = f.read()
    for n, line in enumerate(raw.split(b"\r\n"), start=1):
        body = line.lstrip(b" \t")  # TAB is only allowed as indentation
        bad = [b for b in body if b < 0x20 or b == 0x7F]
        assert not bad, f"install.bat:{n} has control byte(s) {bad}"


def test_install_bat_has_no_clipseg_step_and_pins_transformers():
    text = _install_bat()
    assert "clipseg" not in text.lower() and not os.path.exists(os.path.join(_PROJECT_ROOT, "requirements-clipseg.txt"))
    with open(os.path.join(_PROJECT_ROOT, "requirements-precise.txt")) as f:
        assert f.read().count("transformers==5.22.2") == 1


def test_install_bat_precise_step_is_one_optional_step():
    text = _install_bat()
    assert text.count("Precise selection (Grounding DINO + SAM 2 + ViTMatte, about 980 MB") == 1
    assert 'call :INSTALL_EXTRAS requirements-precise.txt "Precise selection"' in text
    assert "trust_remote_code" not in text


def test_install_bat_offers_ollama_pull_only_when_ollama_exists():
    text = _install_bat()
    i = text.index("where ollama")
    assert "ollama pull qwen3:8b" in text[i:] and text.count("where ollama") == 1
    assert "ollama.com/download" not in text and "OllamaSetup" not in text  # never installs Ollama


# --- a stage that raises (e.g. CUDA out of memory) falls back -----------------------------------------

class _Boom:
    def __init__(self):
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        raise RuntimeError("CUDA out of memory")


def test_sam_error_on_voc_regions_falls_back_to_refine_with_one_warning(monkeypatch, caplog):
    net = FakeSegNet([(PERSON, 0.05, 0.35), (PERSON, 0.65, 0.95)])
    monkeypatch.setattr(agent_tools, "get_segmentation_model", lambda *a, **k: net)
    _install(monkeypatch, sam=(_Boom(), FakeSamModel()))
    refined = []
    real = agent_tools.refine_region
    monkeypatch.setattr(agent_tools, "refine_region", lambda *a: refined.append(1) or real(*a))
    s = AgentSession(engine=FakeEngine(), image=_people())
    out = json.loads(s.detect_regions())
    assert len(out["regions"]) == 2 and refined == [1, 1]
    assert sum("SAM 2 failed, falling back" in r.message for r in caplog.records) == 1


def test_gdino_stage_error_falls_back_to_voc(monkeypatch, voc, caplog):
    _install(monkeypatch, gdino=(FakeGdinoProcessor(TWO_PEOPLE), FakeGdinoModel()), sam=(_Boom(), FakeSamModel()))
    s = AgentSession(engine=FakeEngine(), image=_people())
    out = json.loads(s.detect_regions("person"))
    assert s.detector == "voc" and out["regions"][0]["label"] == "person"
    assert "Precise selection failed" in out["note"]
    assert sum("Grounding DINO / SAM 2 failed" in r.message for r in caplog.records) == 1


def test_matte_error_keeps_the_outline_without_soft_edges(monkeypatch, caplog):
    m = _left_person(monkeypatch, (_Boom(), FakeVitMatteModel()))
    assert m[220, 120] and not m[76, 120]  # the SAM outline, no strand
    assert sum("ViTMatte failed, falling back" in r.message for r in caplog.records) == 1


# --- small objects: native input size, tiles, low size floor ---------------------------------------------

def _dark_box(label, min_px=0):
    """Fake Grounding DINO: the bounding box of the dark pixels in the view it is given, or nothing when the
    object spans fewer than `min_px` view pixels (too small for the real model to recognise)."""
    def find(view):
        dark_px = np.asarray(view.convert("L")) < 128
        if not dark_px.any():
            return []
        ys, xs = np.nonzero(dark_px)
        if min(xs.max() - xs.min(), ys.max() - ys.min()) + 1 < min_px:
            return []
        w, h = view.size
        return [(label, xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h, 0.9)]
    return find


def test_tiles_cover_large_photos_and_skip_small_ones():
    assert agent_tools._tiles(2000, 1500) == []
    w, h = 3820, 5730
    tiles = agent_tools._tiles(w, h)
    covered = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and max(x1 - x0, y1 - y0) <= agent_tools.GDINO_TILE
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_small_object_seen_only_at_tile_resolution_is_found_where_it_is(monkeypatch):
    """A 40 px sign on a 4000x3000 photo: about 13 px in the whole view (missed), 20 px in a tile (found)."""
    img = Image.new("RGB", (4000, 3000), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((3000, 2200, 3039, 2239), fill=(0, 0, 0))
    gp = FakeGdinoProcessor({"sign.": _dark_box("sign", min_px=18)})
    _install(monkeypatch, gdino=(gp, FakeGdinoModel()), sam=(FakeSamProcessor(), FakeSamModel()))
    s = AgentSession(engine=FakeEngine(), image=img)
    out = json.loads(s.detect_regions("sign"))
    assert [(r["label"], r["position"]) for r in out["regions"]] == [("sign", "right-bottom")]
    ys, xs = np.nonzero(s.region_masks["r1"])
    assert 2990 <= xs.min() and xs.max() <= 3050 and 2190 <= ys.min() and ys.max() <= 2250
    assert out["regions"][0]["area_share"] < agent_tools.MIN_REGION_SHARE  # below the old 0.2 % floor


def test_object_cut_by_tile_edges_comes_back_once(monkeypatch):
    """Every tile sees only part of a big object; those cut boxes are dropped and the whole view's box is kept."""
    img = Image.new("RGB", (4000, 3000), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((1000, 500, 3499, 2499), fill=(0, 0, 0))
    gp = FakeGdinoProcessor({"boat.": _dark_box("boat")})
    _install(monkeypatch, gdino=(gp, FakeGdinoModel()), sam=(FakeSamProcessor(), FakeSamModel()))
    s = AgentSession(engine=FakeEngine(), image=img)
    out = json.loads(s.detect_regions("boat"))
    assert len(out["regions"]) == 1 and out["regions"][0]["position"] == "centre-middle"
    assert gp.texts.count("boat.") == 1 + len(agent_tools._tiles(4000, 3000))
