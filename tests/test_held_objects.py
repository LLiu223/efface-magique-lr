"""Job 024: a selected person also gets what they hold, wear or carry. Fakes only: no weights, no network."""

import json
import os
import sys

import pytest
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

pytest.importorskip("langchain.agents")

import numpy as np  # noqa: E402

from agent_fakes import (PERSON, FakeEngine, FakeGdinoModel, FakeGdinoProcessor, FakeSamModel,  # noqa: E402
                         FakeSamProcessor, FakeSegNet, dark)
from companion.agent import tools as agent_tools  # noqa: E402
from companion.agent.harness import photo_context  # noqa: E402
from companion.agent.tools import HELD_QUERY, AgentSession  # noqa: E402
from companion.utils import precise_models as pm  # noqa: E402

RED = (220, 0, 0)  # held objects; dark() sees them (L ~ 66), so the fake SAM outlines them like the person
PERSON_BOX = ("person", 0.1, 0.2, 0.3, 0.9, 0.8)  # x 60..180, y 80..360: the black ellipse only
CARTON = ("rectangle", (181, 205, 205, 235))      # in the hand at the widest point, 1-2 px from the body


def _scene(*red_shapes):
    """A black 'person' ellipse (x 60..180, y 80..360, widest at y 220) on white, over red shapes."""
    img = Image.new("RGB", (600, 400), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for kind, xy in red_shapes:
        getattr(d, kind)(xy, fill=RED)
    d.ellipse((60, 80, 180, 360), fill=(0, 0, 0))
    return img


def _red(img):
    a = np.asarray(img.convert("RGB")).astype(int)
    return (a[..., 0] > 150) & (a[..., 1] < 60)


def red_finder(label):
    """A fake held-object detector: one box around the red pixels of the crop it is given."""
    def find(img):
        ys, xs = np.nonzero(_red(img))
        if not len(xs):
            return []
        w, h = img.size
        return [(label, xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h, 0.9)]
    return find


def _install(monkeypatch, boxes):
    gp, sp = FakeGdinoProcessor(boxes), FakeSamProcessor()
    for stage, value in ((pm.GROUNDING_DINO, (gp, FakeGdinoModel())), (pm.SAM2, (sp, FakeSamModel()))):
        monkeypatch.setattr(stage, "get", lambda v=value: v)
        monkeypatch.setattr(stage, "is_available", lambda: True)
    monkeypatch.setattr(pm.VITMATTE, "get", lambda: None)
    return gp, sp


def _person_mask():
    d = Image.new("L", (600, 400), 0)
    ImageDraw.Draw(d).ellipse((60, 80, 180, 360), fill=255)
    return np.asarray(d).copy()


def test_held_object_touching_the_hand_is_added(monkeypatch):
    img = _scene(CARTON)
    gp, sp = _install(monkeypatch, {"person.": [PERSON_BOX], HELD_QUERY: red_finder("carton")})
    s = AgentSession(engine=FakeEngine(), image=img)
    r1 = json.loads(s.detect_regions("person"))["regions"][0]
    assert r1["id"] == "r1" and r1["label"] == "person" and r1["with"] == ["carton"]
    assert gp.texts == ["person.", HELD_QUERY] and len(sp.calls) == 2  # one GDINO + one SAM call extra
    m = s.region_masks["r1"] > 0
    assert m[220, 200] and m[220, 120]  # the carton and the body
    assert np.count_nonzero(m & _red(img)) == np.count_nonzero(_red(img))
    share = np.count_nonzero(dark(img)) / dark(img).size  # the union before the edge margin
    assert abs(r1["area_share"] - share) < 0.002 and r1["position"] == "left-middle"
    assert "A person region also covers what the person holds" in photo_context(s)


def test_object_near_but_not_touching_is_not_added(monkeypatch):
    img = _scene(("rectangle", (185, 90, 205, 110)))  # inside the person's box, ~35 px from the body
    _, sp = _install(monkeypatch, {"person.": [PERSON_BOX], HELD_QUERY: red_finder("phone")})
    s = AgentSession(engine=FakeEngine(), image=img)
    r1 = json.loads(s.detect_regions("person"))["regions"][0]
    assert "with" not in r1 and len(sp.calls) == 2  # outlined, then rejected
    assert not (s.region_masks["r1"] > 0)[100, 195]


def test_second_person_touching_is_not_added(monkeypatch):
    img = _scene(("ellipse", (181, 170, 207, 290)))  # a child, red so the held detector sees it; touches at y ~ 220
    child_box = ("person", 181 / 600, 170 / 400, 208 / 600, 291 / 400, 0.7)
    # without the child as its own region the 'bag' passes every other test...
    _install(monkeypatch, {"person.": [PERSON_BOX], HELD_QUERY: red_finder("bag")})
    s = AgentSession(engine=FakeEngine(), image=img)
    assert json.loads(s.detect_regions("person"))["regions"][0]["with"] == ["bag"]
    # ...but here it lies on another person region of the same call
    _install(monkeypatch, {"person.": [PERSON_BOX, child_box], HELD_QUERY: red_finder("bag")})
    s = AgentSession(engine=FakeEngine(), image=img)
    regions = json.loads(s.detect_regions("person"))["regions"]
    assert [r["label"] for r in regions] == ["person", "person"] and not any("with" in r for r in regions)
    assert not (s.region_masks["r1"] > 0)[230, 200]  # r1 is the larger, first person


def test_a_person_label_from_the_held_query_is_never_outlined(monkeypatch):
    _, sp = _install(monkeypatch, {"person.": [PERSON_BOX], HELD_QUERY: [("woman", 0.0, 0.0, 1.0, 1.0, 0.9)]})
    s = AgentSession(engine=FakeEngine(), image=_scene())
    assert "with" not in json.loads(s.detect_regions("person"))["regions"][0] and len(sp.calls) == 1


@pytest.mark.parametrize("shape", [
    ("rectangle", (20, 40, 240, 400)),    # a wall behind the person: over HELD_MAX_AREA of the person
    ("rectangle", (178, 200, 400, 230)),  # a rail touching the hand, mostly outside the person's box
])
def test_large_background_candidate_is_rejected(monkeypatch, shape):
    img = _scene(shape)
    _, sp = _install(monkeypatch, {HELD_QUERY: red_finder("bag")})
    person = _person_mask()
    mask, labels = AgentSession(engine=FakeEngine(), image=img)._add_held(img, person, [])
    assert labels == [] and mask is person and len(sp.calls) == 1


def test_non_person_query_runs_no_extra_detection(monkeypatch):
    gp, sp = _install(monkeypatch, {"power lines.": [("power lines", 0.0, 0.28, 1.0, 0.3, 0.9)],
                                    HELD_QUERY: lambda img: pytest.fail("no held-object pass for power lines")})
    AgentSession(engine=FakeEngine(), image=_scene(CARTON)).detect_regions("power lines")
    assert gp.texts == ["power lines."] and len(sp.calls) == 1


def test_voc_person_also_gets_held_objects(monkeypatch):
    net = FakeSegNet([(PERSON, 0.1, 0.3)])
    monkeypatch.setattr(agent_tools, "get_segmentation_model", lambda *a, **k: net)
    _install(monkeypatch, {HELD_QUERY: red_finder("carton")})
    s = AgentSession(engine=FakeEngine(), image=_scene(CARTON))
    r1 = json.loads(s.detect_regions())["regions"][0]
    assert s.detector == "voc" and r1["label"] == "person" and r1["with"] == ["carton"]


def test_raising_held_stage_keeps_the_plain_mask_with_one_warning(monkeypatch, caplog):
    second = ("person", 0.7, 0.25, 0.9, 0.9, 0.7)
    _install(monkeypatch, {"person.": [PERSON_BOX, second]})
    plain = AgentSession(engine=FakeEngine(), image=_scene(CARTON))
    plain.detect_regions("person")

    def boom(img):
        raise RuntimeError("CUDA out of memory")
    _install(monkeypatch, {"person.": [PERSON_BOX, second], HELD_QUERY: boom})
    s = AgentSession(engine=FakeEngine(), image=_scene(CARTON))
    out = json.loads(s.detect_regions("person"))
    assert out["regions"] == list(plain.regions.values()) and not any("with" in r for r in out["regions"])
    assert all(np.array_equal(s.region_masks[r], plain.region_masks[r]) for r in plain.region_masks)
    assert sum("Held objects failed, falling back" in r.message for r in caplog.records) == 1


def test_missing_gdino_skips_silently(monkeypatch, caplog):
    _install(monkeypatch, {HELD_QUERY: red_finder("carton")})
    monkeypatch.setattr(pm.GROUNDING_DINO, "get", lambda: None)
    s = AgentSession(engine=FakeEngine(), image=_scene(CARTON))
    person = _person_mask()
    mask, labels = s._add_held(s.image, person, [])
    assert mask is person and labels == [] and not caplog.records
