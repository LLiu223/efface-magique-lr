"""Usage log (d), key hygiene (e), provider factory (AC 3)."""

import logging
import os
import sqlite3
import sys
import time

import pytest
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.agent.llm import AgentConfigError, DEFAULT_OLLAMA_URL, is_local_url, make_llm, scrub  # noqa: E402
from companion.agent.usage import UsageStore, estimate_cost, load_prices  # noqa: E402

DUMMY_KEY = "sk-DUMMYKEY-0123456789abcdefXYZ"


# --- store (no LangChain needed) -------------------------------------------

def test_cost_table():
    assert estimate_cost("ollama", "llama3.1", 10_000, 10_000) == 0.0
    assert estimate_cost("openai", "no-such-model", 10, 10) is None
    assert estimate_cost("openai", "gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.5)


def test_price_override_file(tmp_path):
    (tmp_path / "agent_prices.json").write_text('{"openai/my-model": [1, 2]}', encoding="utf-8")
    prices = load_prices(str(tmp_path))
    assert prices["openai/my-model"] == (1.0, 2.0) and "openai/gpt-4o" in prices
    (tmp_path / "agent_prices.json").write_text("not json", encoding="utf-8")
    assert "openai/my-model" not in load_prices(str(tmp_path))


def test_summary_groups_by_day_provider_model(tmp_path):
    st = UsageStore(str(tmp_path / "u.db"), prices={})
    now = time.time()
    st.record(request_id="a", provider="ollama", model="m", input_tokens=10, output_tokens=5, latency_ms=100, ts=now)
    st.record(request_id="a", provider="ollama", model="m", input_tokens=30, output_tokens=5, latency_ms=300, ts=now)
    st.record(request_id="b", provider="openai", model="x", input_tokens=1, output_tokens=1, latency_ms=50,
              status="error: Boom", ts=now)
    st.record(request_id="old", provider="ollama", model="m", input_tokens=1, output_tokens=1, latency_ms=1, ts=now - 86400 * 10)
    rows = {(r["provider"], r["model"]): r for r in st.summary(since=now - 3600)}
    assert rows[("ollama", "m")]["calls"] == 2 and rows[("ollama", "m")]["input_tokens"] == 40
    assert rows[("ollama", "m")]["cost"] == 0 and rows[("ollama", "m")]["avg_latency_ms"] == 200
    assert rows[("openai", "x")]["cost"] is None and rows[("openai", "x")]["errors"] == 1
    assert len(st.summary(0)) == 3
    st.clear()
    assert st.recent() == []


def test_default_path_is_app_data(monkeypatch, tmp_path):
    from PyQt6.QtCore import QStandardPaths
    QStandardPaths.setTestModeEnabled(True)
    try:
        from companion.agent.usage import default_db_path
        path = default_db_path()
        assert path.startswith(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation))
    finally:
        QStandardPaths.setTestModeEnabled(False)


# --- provider factory -------------------------------------------------------

def test_make_llm_rejects_unknown_provider():
    with pytest.raises(AgentConfigError, match="Unknown assistant provider 'mistral'"):
        make_llm("mistral", "x")


@pytest.mark.parametrize("provider,env", [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY"),
                                          ("gemini", "GOOGLE_API_KEY")])
def test_missing_key_is_one_line(monkeypatch, provider, env):
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(AgentConfigError) as ei:
        make_llm(provider, "m")
    assert str(ei.value) == f"Set the {env} environment variable to use {provider}." and "\n" not in str(ei.value)


def test_ollama_defaults_to_localhost():
    pytest.importorskip("langchain_ollama")
    assert make_llm("ollama", "llama3.1").base_url == DEFAULT_OLLAMA_URL == "http://localhost:11434"
    assert is_local_url(DEFAULT_OLLAMA_URL) and not is_local_url("http://10.0.0.5:11434")


@pytest.mark.parametrize("provider,env", [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY"),
                                          ("gemini", "GOOGLE_API_KEY")])
def test_cloud_clients_hide_key_in_repr(monkeypatch, provider, env):
    pytest.importorskip("langchain_" + {"gemini": "google_genai"}.get(provider, provider))
    monkeypatch.setenv(env, DUMMY_KEY)
    llm = make_llm(provider, "m")
    assert DUMMY_KEY not in repr(llm) and DUMMY_KEY not in str(llm)


def test_scrub_removes_env_keys_and_key_shapes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "plainsecretvalue")
    out = scrub("auth failed for plainsecretvalue and sk-abcdefghijklmnop and AIza" + "B" * 30 + "\nline2")
    assert "plainsecretvalue" not in out and "sk-abc" not in out and "AIzaB" not in out and "\n" not in out


# --- (d) + (e) through the real harness and callback ---------------------

def test_usage_rows_written_with_token_sums(tmp_path, monkeypatch):
    pytest.importorskip("langchain.agents")
    from agent_fakes import FakeEngine, call, say, scripted
    from companion.agent.harness import run_agent
    from companion.agent.tools import AgentSession
    from companion.agent.usage import make_usage_callback

    st = UsageStore(str(tmp_path / "u.db"), prices={})
    cb = make_usage_callback(st, "ollama", "llama3.1", request_id="req1")
    s = AgentSession(engine=FakeEngine(), image=Image.new("RGB", (40, 30)))
    r = run_agent(scripted(call("list_models"), say("done")), s, "what models?", cb, retry_delay=0)
    assert r.ok
    rows = st.recent()
    assert len(rows) == 2 and {x["request_id"] for x in rows} == {"req1"}
    assert sum(x["input_tokens"] for x in rows) == 200 and sum(x["output_tokens"] for x in rows) == 40
    assert cb.total_tokens == 240 == r.tokens
    assert all(x["cost"] == 0 and x["status"] == "ok" and x["prompt_text"] is None for x in rows)
    assert all(x["latency_ms"] >= 0 for x in rows)


def test_prompt_text_only_with_opt_in(tmp_path):
    pytest.importorskip("langchain.agents")
    from agent_fakes import FakeEngine, say, scripted
    from companion.agent.harness import run_agent
    from companion.agent.tools import AgentSession
    from companion.agent.usage import make_usage_callback

    st = UsageStore(str(tmp_path / "u.db"), prices={})
    s = AgentSession(engine=FakeEngine(), image=Image.new("RGB", (40, 30)))
    run_agent(scripted(say("hi")), s, "remove the dog", make_usage_callback(st, "ollama", "m", store_prompts=True))
    assert "remove the dog" in st.recent()[0]["prompt_text"]


def test_dummy_key_never_leaks(tmp_path, monkeypatch, caplog):
    pytest.importorskip("langchain.agents")
    from agent_fakes import FakeEngine, scripted
    from companion.agent.harness import run_agent
    from companion.agent.tools import AgentSession
    from companion.agent.usage import make_usage_callback

    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)

    class AuthenticationError(Exception):
        status_code = 401

    db = tmp_path / "u.db"
    st = UsageStore(str(db), prices={})
    caplog.set_level(logging.DEBUG)
    s = AgentSession(engine=FakeEngine(), image=Image.new("RGB", (40, 30)))
    boom = AuthenticationError(f"Incorrect API key provided: {DUMMY_KEY}. Request headers: Bearer {DUMMY_KEY}")
    cb = make_usage_callback(st, "openai", "gpt-4o-mini", store_prompts=True)
    r = run_agent(scripted(boom), s, f"my key is {DUMMY_KEY}, remove the dog", cb, retry_delay=0)

    assert not r.ok and "AuthenticationError" in r.reason
    assert DUMMY_KEY not in r.reason  # text emitted to the UI
    assert DUMMY_KEY not in caplog.text  # log output
    with sqlite3.connect(db) as con:
        con.execute("PRAGMA wal_checkpoint")
    assert DUMMY_KEY.encode() not in db.read_bytes()  # DB file
    rows = st.recent()
    assert rows and rows[0]["status"] == "error: AuthenticationError"
    assert "***" in (rows[0]["prompt_text"] or "") and DUMMY_KEY not in str(rows)
