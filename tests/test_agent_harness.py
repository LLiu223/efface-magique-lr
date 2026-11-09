"""Agent harness: limits (a), allow-list (b), scripted "remove the bin" (c), retry, region detection."""

import json
import os
import sys

import pytest
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

pytest.importorskip("langchain.agents")

import numpy as np  # noqa: E402

from agent_fakes import (BOTTLE, PERSON, FakeEngine, FakeGdinoModel, FakeGdinoProcessor,  # noqa: E402
                         FakeSamModel, FakeSamProcessor, FakeSegNet, call, say, scripted)
from companion.agent import tools as agent_tools  # noqa: E402
from companion.agent.harness import AgentWorker, is_transient, run_agent  # noqa: E402
from companion.agent.tools import TOOL_NAMES, AgentSession, make_tools, sanitize_text  # noqa: E402
from companion.agent.usage import UsageStore, make_usage_callback  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return UsageStore(str(tmp_path / "usage.sqlite3"), prices={})


@pytest.fixture
def seg(monkeypatch):
    """Install a fake segmentation net; returns a setter for its boxes."""
    net = FakeSegNet([])
    monkeypatch.setattr(agent_tools, "get_segmentation_model", lambda *a, **k: net)
    return net


def _session(engine=None):
    return AgentSession(engine=engine or FakeEngine(), image=Image.new("RGB", (300, 200), (90, 90, 90)))


def _run(llm, session, store, **kw):
    cb = make_usage_callback(store, "ollama", "fake")
    kw.setdefault("retry_delay", 0)
    return run_agent(llm, session, kw.pop("text", "remove the bin"), cb, **kw), cb


# --- (a) limits -----------------------------------------------------------

def test_stops_at_max_steps(store, seg):
    llm = scripted(*[call("list_models", i=i) for i in range(10)])
    s = _session()
    r, _ = _run(llm, s, store, max_steps=3)
    assert not r.ok and "limit of 3 steps" in r.reason
    assert r.steps == 3 and len(llm.seen) == 3
    assert s.calls == ["list_models", "list_models"]  # the 3rd model call's tools never ran


def test_stops_at_token_budget(store, seg):
    llm = scripted(*[call("list_models", i=i) for i in range(10)])
    r, cb = _run(llm, _session(), store, token_budget=250)  # 120 tokens per call
    assert not r.ok and "token budget" in r.reason
    assert cb.total_tokens == 360 and len(llm.seen) == 3


def test_stops_on_cancel_between_steps(store, seg):
    s = _session()
    llm = scripted(call("list_models"), say("never"), on_call=lambda n: s.cancel_event.set())
    r, _ = _run(llm, s, store)
    assert not r.ok and r.reason == "Cancelled."
    assert s.calls == [] and len(llm.seen) == 1


def test_stops_at_timeout(store, seg):
    times = iter([0.0, 500.0, 1000.0, 1500.0])
    llm = scripted(call("list_models"), say("late"))
    r, _ = _run(llm, _session(), store, timeout_s=60, clock=lambda: next(times))
    assert not r.ok and "Timed out after 60 s" in r.reason


# --- (b) allow-list -------------------------------------------------------

def test_tool_outside_allow_list_is_refused(store, seg):
    s = _session()
    llm = scripted(call("run_shell", {"cmd": "rm -rf /"}), say("done"))
    r, _ = _run(llm, s, store)
    assert not r.ok and "Refused tool call outside the allow-list: 'run_shell'" in r.reason
    assert s.calls == [] and r.selection is None


def test_tool_surface_is_exactly_the_allow_list():
    tools = make_tools(_session())
    assert tuple(t.name for t in tools) == TOOL_NAMES
    for t in tools:
        assert t.description
        for arg in t.args:  # no path / url / shell argument anywhere
            assert arg in {"name", "text", "region_ids", "query"}
    assert set(tools[3].args) == {"query"}
    assert set(tools[4].args) == {"region_ids"}


# --- (c) scripted conversation -------------------------------------------

def test_remove_the_bin_conversation(store, seg):
    seg.boxes = [(PERSON, 0.05, 0.15), (BOTTLE, 0.7, 0.95)]  # a "bin" is not VOC; the bottle stands in
    engine = FakeEngine()
    s = _session(engine)
    llm = scripted(
        call("select_model", {"name": "lama"}, 1),
        call("detect_regions", {}, 2),
        call("select_regions", {"region_ids": ["r1"]}, 3),
        say("Selected the bottle on the right. Adjust it and click Erase."),
    )
    r, _ = _run(llm, s, store)
    assert r.ok, r.reason
    assert r.tool_calls == ["select_model", "detect_regions", "select_regions"]
    regions = s.regions
    assert {v["label"] for v in regions.values()} == {"person", "bottle"}
    bottle_id = next(k for k, v in regions.items() if v["label"] == "bottle")
    assert regions[bottle_id]["position"] == "right-middle"
    # the region model output reached the model as a tool result
    assert any("bottle" in m for m in llm.seen[2])
    sel = r.selection
    assert sel is not None and sel.region_ids == [bottle_id] == ["r1"] and sel.labels == ["bottle"]
    assert engine.generated == [] and engine.model_name == "lama"  # the assistant never inpaints
    assert sel.mask.size == (300, 200) and sel.mask.getpixel((255, 100)) == 255 and sel.mask.getpixel((30, 100)) == 0
    assert "Selected. The user can adjust it with the brush and click Erase." in llm.seen[3][-1]
    assert "Erase" in r.text


def test_not_found_path_says_so(store, seg):
    seg.boxes = []
    engine = FakeEngine()
    s = _session(engine)
    llm = scripted(call("detect_regions"), say("I could not find a bin; only VOC objects can be located."))
    r, _ = _run(llm, s, store)
    assert r.ok and r.selection is None and engine.generated == []
    assert "No VOC objects found" in llm.seen[1][-1]
    assert "could not find" in r.text


def test_detect_regions_labels_positions_and_cap(seg):
    seg.boxes = [(PERSON, 0.0, 0.3), (PERSON, 0.4, 0.6), (BOTTLE, 0.8, 0.9)]
    s = _session()
    s.output_cap = 10_000
    out = json.loads(s.detect_regions())["regions"]
    assert [(r["label"], r["position"]) for r in out] == [
        ("person", "left-middle"), ("person", "centre-middle"), ("bottle", "right-middle")]
    assert out[0]["area_share"] > out[2]["area_share"] > 0
    s.output_cap = 60
    assert len(s.detect_regions()) <= 60


# --- named objects: Grounding DINO + SAM 2, else VOC -----------------------------------------------

@pytest.fixture
def precise(monkeypatch):
    """Fake Grounding DINO + SAM 2; set `gp.boxes` = [(label, x0, y0, x1, y1, score)] before calling detect."""
    from companion.utils import precise_models as pm
    gp, sp = FakeGdinoProcessor(), FakeSamProcessor(segment=lambda crop: np.ones(crop.size[::-1], bool))
    for stage, value in ((pm.GROUNDING_DINO, (gp, FakeGdinoModel())), (pm.SAM2, (sp, FakeSamModel()))):
        monkeypatch.setattr(stage, "get", lambda v=value: v)
        monkeypatch.setattr(stage, "is_available", lambda: True)
    return gp


def test_named_query_is_sanitised_lowercased_and_capped(precise):
    precise.boxes = [("sign", 0.1, 0.1, 0.5, 0.5, 0.9)]
    out = json.loads(_session().detect_regions("The\x00 Sign‮ on\n the left " + "x" * 200))
    text = precise.texts[0]
    assert text.startswith("the sign on the left x") and text.endswith(".") and len(text) == agent_tools.MAX_QUERY_LEN + 1
    assert out["regions"][0]["label"] == "sign"


def test_named_query_nothing_found_says_do_not_guess(precise):
    precise.boxes = [("rock", 0.1, 0.1, 0.5, 0.5, 0.2)]  # below the box threshold
    out = json.loads(_session().detect_regions("rock"))
    assert out["regions"] == [] and "nothing matching 'rock' found; do not guess a region" in out["note"]


def test_blank_query_runs_voc_like_no_query(seg, precise):
    seg.boxes = [(PERSON, 0.0, 0.3)]
    s = _session()
    assert s.detect_regions("  \x00 ") == s.detect_regions() == s.detect_regions(None)
    held = [agent_tools.HELD_QUERY]  # VOC found a person: only the fixed held-object query ran
    assert precise.texts == held * 3 and json.loads(s.detect_regions())["regions"][0]["label"] == "person"


def test_query_without_precise_selection_falls_back_to_voc_with_note(seg):
    seg.boxes = [(PERSON, 0.0, 0.3)]
    s = _session()
    plain = s.detect_regions()
    assert "note" not in json.loads(plain)
    out = json.loads(s.detect_regions("power lines"))
    assert [r["label"] for r in out["regions"]] == ["person"]
    assert "install the optional Precise selection step (install.bat)" in out["note"]
    assert json.loads(plain)["regions"] == out["regions"]


def test_query_after_precise_load_failure_says_failed(seg, monkeypatch):
    from companion.utils import precise_models as pm
    seg.boxes = []
    monkeypatch.setattr(pm.GROUNDING_DINO, "_error", "OSError: broken")
    out = json.loads(_session().detect_regions("sign"))
    assert out["regions"] == [] and "Precise selection failed" in out["note"] and "No VOC objects found" in out["note"]


def test_conversation_passes_query_through_the_tool(store, precise):
    precise.boxes = [("sign", 0.05, 0.4, 0.3, 0.9, 0.9)]
    s = _session()
    llm = scripted(call("detect_regions", {"query": "sign"}, 1),
                   call("select_regions", {"region_ids": ["r1"]}, 2), say("Selected the sign. Click Erase."))
    r, _ = _run(llm, s, store, text="remove the sign on the left")
    assert r.ok, r.reason
    assert precise.texts == ["sign."] and r.selection.labels == ["sign"]


def test_select_unknown_region_and_select_exact_match(seg):
    seg.boxes = [(PERSON, 0.0, 0.3)]
    s = _session()
    assert json.loads(s.select_regions(["r1"])) == {"ok": False, "error": "unknown region ids; call detect_regions first",
                                                    "unknown": ["r1"], "known": []}
    s.detect_regions()
    out = json.loads(s.select_regions(["r1", "r9"]))
    assert out["ok"] is False and out["unknown"] == ["r9"] and out["known"] == ["r1"] and s.selection is None
    assert json.loads(s.select_model("LaMa"))["ok"] is False  # exact match only
    assert json.loads(s.select_model("lama"))["ok"] is True


def test_sanitize_text_strips_controls():
    assert sanitize_text("a\x00b‮c\nd   e", 100) == "a b c d e"
    assert len(sanitize_text("x" * 500)) == 200


# --- retry --------------------------------------------------------------

def test_retries_once_on_transient_error(store, seg):
    llm = scripted(ConnectionError("blip"), say("ok"))
    r, _ = _run(llm, _session(), store)
    assert r.ok and r.text == "ok" and len(llm.seen) == 2


def test_gives_up_after_one_retry(store, seg):
    llm = scripted(TimeoutError("slow"), TimeoutError("slow"), say("never"))
    r, _ = _run(llm, _session(), store)
    assert not r.ok and r.reason.startswith("Assistant error: TimeoutError") and len(llm.seen) == 2


def test_permanent_error_is_not_retried(store, seg):
    llm = scripted(ValueError("bad request"), say("never"))
    r, _ = _run(llm, _session(), store)
    assert not r.ok and "ValueError" in r.reason and len(llm.seen) == 1


def test_is_transient():
    class RateLimitError(Exception):
        pass

    class E(Exception):
        status_code = 401

    assert is_transient(RateLimitError()) and is_transient(ConnectionError())
    assert not is_transient(E()) and not is_transient(ValueError())


def test_worker_emits_progress_and_result(qtbot, store, seg):
    s = _session()
    w = AgentWorker(lambda: scripted(call("detect_regions"), say("nothing")),
                    lambda: make_usage_callback(store, "ollama", "fake"), s, "remove the bin", retry_delay=0)
    msgs, results = [], []
    w.progress.connect(msgs.append)
    w.resultReady.connect(results.append)
    with qtbot.waitSignal(w.finished, timeout=20000):
        w.start()
    qtbot.waitUntil(lambda: bool(results), timeout=5000)
    assert "Thinking..." in msgs and "Detecting regions..." in msgs
    assert results[0].ok


def test_worker_reports_config_error_without_traceback(qtbot, monkeypatch, store):
    from companion.agent.llm import make_llm
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    w = AgentWorker(lambda: make_llm("openai", "gpt-4o-mini"), lambda: None, _session(), "hi")
    results = []
    w.resultReady.connect(results.append)
    with qtbot.waitSignal(w.finished, timeout=10000):
        w.start()
    qtbot.waitUntil(lambda: bool(results), timeout=5000)
    assert results[0].reason == "Set the OPENAI_API_KEY environment variable to use openai."


def test_langsmith_tracing_forced_off(monkeypatch):
    """A user env with tracing switched on must not upload requests to LangSmith."""
    from companion.agent import harness
    from langchain_core.tracers.context import _tracing_v2_is_enabled
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    assert _tracing_v2_is_enabled()  # the env really would turn tracing on
    monkeypatch.setattr(harness, "_run_agent", lambda *a, **k: _tracing_v2_is_enabled())
    assert harness.run_agent(None, None, "x", None) is False


def test_model_that_only_describes_still_selects_the_match(store, seg):
    """llama3.1 often says "one person found" and stops: the matching region is selected for it."""
    seg.boxes = [(PERSON, 0.4, 0.6), (BOTTLE, 0.8, 0.95)]
    s = _session()
    llm = scripted(call("detect_regions", {"query": "person"}), say("There is one person at the centre-bottom."))
    r, _ = _run(llm, s, store, text="select the man")
    assert r.ok and r.selection is not None
    assert r.selection.labels == ["person"] and r.tool_calls[-1] == "select_regions"


def test_fallback_never_selects_an_unrelated_voc_object(store, seg):
    seg.boxes = [(PERSON, 0.4, 0.6)]
    s = _session()
    llm = scripted(call("detect_regions", {"query": "power lines"}), say("No power lines found."))
    r, _ = _run(llm, s, store, text="select the power lines")
    assert r.ok and r.selection is None


def test_prompt_gives_photo_size_model_and_detectable_classes(store, seg):
    llm = scripted(say("ok"))
    _run(llm, _session(FakeEngine("migan")), store)
    system = llm.seen[0][0]
    assert "300x200" in system and "migan" in system and "person" in system and "select_regions" in system


def _figure(w=2400, h=1600):
    truth = np.zeros((h, w), np.uint8)
    truth[400:1400, 1000:1300] = 255   # the person
    truth[300:1400, 1330:1500] = 255   # a second person just beside them, 30 px away
    return truth


def test_refine_region_snaps_a_blobby_first_pass_to_the_object():
    truth = _figure()
    person = np.zeros_like(truth)
    person[400:1400, 1000:1300] = 255
    coarse = np.zeros_like(truth)
    coarse[380:1420, 940:1310] = 255   # first pass: shifted ~60 px left, as on the owner's photo
    seg = lambda crop: np.asarray(crop.convert("L")) > 127  # noqa: E731 - the "detector" sees the true shape
    img = Image.fromarray(truth).convert("RGB")
    out = agent_tools.refine_region(img, coarse, seg, cell=2400 / 512)
    spill = np.count_nonzero((out > 0) & (person == 0))
    assert spill < 0.1 * np.count_nonzero(coarse & ~person)       # most of the spill is gone
    assert np.count_nonzero((out > 0) & (person > 0)) > 0.97 * np.count_nonzero(person)
    assert not out[:, 1400:].any()                                 # the neighbour is not pulled in


def test_refine_region_keeps_first_pass_when_sharp_pass_finds_nothing():
    coarse = np.zeros((1600, 2400), np.uint8)
    coarse[400:1400, 1000:1300] = 255
    img = Image.new("RGB", (2400, 1600))
    out = agent_tools.refine_region(img, coarse, lambda crop: np.zeros(crop.size[::-1], bool), cell=2400 / 512)
    assert np.array_equal(out, coarse)


def test_refine_region_skips_small_photos():
    coarse = np.zeros((200, 300), np.uint8)
    coarse[50:150, 100:200] = 255
    called = []
    out = agent_tools.refine_region(Image.new("RGB", (300, 200)), coarse, lambda c: called.append(1), cell=1.0)
    assert out is coarse and not called
