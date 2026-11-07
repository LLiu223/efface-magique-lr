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

from agent_fakes import BOTTLE, PERSON, FakeEngine, FakeSegNet, call, say, scripted  # noqa: E402
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
    assert s.calls == [] and r.proposal is None


def test_tool_surface_is_exactly_the_allow_list():
    tools = make_tools(_session())
    assert tuple(t.name for t in tools) == TOOL_NAMES
    for t in tools:
        assert t.description
        for arg in t.args:  # no path / url / shell argument anywhere
            assert arg in {"name", "text", "region_ids", "num_variations"}
    assert set(tools[4].args) == {"region_ids", "num_variations"}


# --- (c) scripted conversation -------------------------------------------

def test_remove_the_bin_conversation(store, seg):
    seg.boxes = [(PERSON, 0.05, 0.15), (BOTTLE, 0.7, 0.95)]  # a "bin" is not VOC; the bottle stands in
    engine = FakeEngine()
    s = _session(engine)
    llm = scripted(
        call("select_model", {"name": "lama"}, 1),
        call("detect_regions", {}, 2),
        call("inpaint_region", {"region_ids": ["r1"], "num_variations": 1}, 3),
        say("Proposed erasing the bottle on the right. Click Apply to keep it."),
    )
    r, _ = _run(llm, s, store)
    assert r.ok, r.reason
    assert r.tool_calls == ["select_model", "detect_regions", "inpaint_region"]
    regions = s.regions
    assert {v["label"] for v in regions.values()} == {"person", "bottle"}
    bottle_id = next(k for k, v in regions.items() if v["label"] == "bottle")
    assert regions[bottle_id]["position"] == "right-middle"
    # the region model output reached the model as a tool result
    assert any("bottle" in m for m in llm.seen[2])
    p = r.proposal
    assert p is not None and p.region_ids == [bottle_id] == ["r1"] and p.labels == ["bottle"]
    assert len(engine.generated) == 1 and p.model == "lama"
    mask = engine.generated[0][1]
    assert mask.size == (300, 200) and mask.getpixel((255, 100)) == 255 and mask.getpixel((30, 100)) == 0
    assert "Apply" in r.text


def test_not_found_path_says_so(store, seg):
    seg.boxes = []
    engine = FakeEngine()
    s = _session(engine)
    llm = scripted(call("detect_regions"), say("I could not find a bin; only VOC objects can be located."))
    r, _ = _run(llm, s, store)
    assert r.ok and r.proposal is None and engine.generated == []
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


def test_inpaint_unknown_region_and_select_exact_match(seg):
    s = _session()
    assert json.loads(s.inpaint_region(["r9"]))["ok"] is False
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
