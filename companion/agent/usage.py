"""Local SQLite usage log for assistant LLM calls (one row per call).

No prompts, images or keys are stored unless the "store prompt text" opt-in is set.
"""

import csv
import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

from companion.agent.llm import scrub

# Editable price table: USD per million (input, output) tokens. Ollama is always 0;
# a model missing here gets cost NULL (never a guess). Override or extend it with
# agent_prices.json in the app data folder: {"provider/model": [in_per_mtok, out_per_mtok]}.
PRICES: Dict[str, Tuple[float, float]] = {
    "anthropic/claude-opus-4-1": (15.0, 75.0),
    "anthropic/claude-sonnet-4-5": (3.0, 15.0),
    "anthropic/claude-haiku-4-5": (1.0, 5.0),
    "openai/gpt-4o": (2.5, 10.0),
    "openai/gpt-4o-mini": (0.15, 0.6),
    "gemini/gemini-2.5-flash": (0.3, 2.5),
    "gemini/gemini-2.5-pro": (1.25, 10.0),
}

_SCHEMA = """CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    cost REAL,
    status TEXT NOT NULL,
    prompt_text TEXT
)"""


def app_data_dir() -> str:
    from PyQt6.QtCore import QStandardPaths
    path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    os.makedirs(path, exist_ok=True)
    return path


def default_db_path() -> str:
    return os.path.join(app_data_dir(), "agent_usage.sqlite3")


def load_prices(folder: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    prices = dict(PRICES)
    path = os.path.join(folder or app_data_dir(), "agent_prices.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                prices.update({str(k): (float(v[0]), float(v[1])) for k, v in json.load(f).items()})
        except (OSError, ValueError, TypeError, IndexError, AttributeError):
            pass  # a broken override must not break the assistant; built-in table still applies
    return prices


def estimate_cost(provider: str, model: str, in_tok: int, out_tok: int, prices=None) -> Optional[float]:
    if provider == "ollama":
        return 0.0
    price = (prices if prices is not None else PRICES).get(f"{provider}/{model}")
    if price is None:
        return None
    return (in_tok * price[0] + out_tok * price[1]) / 1_000_000


class UsageStore:
    """Thread-safe SQLite store (rows are written from the agent worker thread)."""

    def __init__(self, path: Optional[str] = None, prices=None):
        self.path = path or default_db_path()
        self.prices = prices if prices is not None else load_prices(os.path.dirname(self.path))
        self._lock = threading.Lock()
        with self._connect() as con:
            con.execute(_SCHEMA)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def record(self, *, request_id, provider, model, input_tokens, output_tokens, latency_ms,
               status="ok", prompt_text=None, ts=None):
        cost = estimate_cost(provider, model, input_tokens, output_tokens, self.prices)
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO llm_calls (ts, request_id, provider, model, input_tokens, output_tokens,"
                " latency_ms, cost, status, prompt_text) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts or time.time(), request_id, provider, model, int(input_tokens), int(output_tokens),
                 int(latency_ms), cost, status, prompt_text),
            )

    def summary(self, since: float = 0.0) -> List[dict]:
        """Totals per day / provider / model since the given unix time."""
        q = ("SELECT date(ts, 'unixepoch', 'localtime') AS day, provider, model, COUNT(*) AS calls,"
             " SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,"
             " SUM(cost) AS cost, AVG(latency_ms) AS avg_latency_ms,"
             " SUM(status != 'ok') AS errors"
             " FROM llm_calls WHERE ts >= ? GROUP BY day, provider, model ORDER BY day DESC, provider, model")
        return self._query(q, (since,))

    def recent(self, limit: int = 50) -> List[dict]:
        return self._query("SELECT * FROM llm_calls ORDER BY ts DESC, id DESC LIMIT ?", (limit,))

    def _query(self, q, args) -> List[dict]:
        with self._lock, self._connect() as con:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute(q, args)]

    def export_csv(self, path: str) -> int:
        rows = self._query("SELECT * FROM llm_calls ORDER BY ts", ())
        cols = ["id", "ts", "request_id", "provider", "model", "input_tokens", "output_tokens",
                "latency_ms", "cost", "status", "prompt_text"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return len(rows)

    def clear(self):
        with self._lock, self._connect() as con:
            con.execute("DELETE FROM llm_calls")


def make_usage_callback(store: UsageStore, provider: str, model: str, request_id: Optional[str] = None,
                        store_prompts: bool = False):
    """Return a LangChain BaseCallbackHandler that logs each LLM call and keeps a running token total."""
    from langchain_core.callbacks import BaseCallbackHandler

    class UsageCallback(BaseCallbackHandler):
        def __init__(self):
            self.request_id = request_id or uuid.uuid4().hex[:12]
            self.total_tokens = 0
            self.calls = 0
            self._starts: Dict[object, Tuple[float, Optional[str]]] = {}

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            prompt = None
            if store_prompts:
                # The user request only: the system prompt and tool results are not stored
                prompt = " ".join(str(m.content) for batch in messages for m in batch if m.type == "human")
            self._starts[run_id] = (time.perf_counter(), prompt)

        def on_llm_end(self, response, *, run_id, **kwargs):
            in_tok, out_tok = _tokens(response)
            self.total_tokens += in_tok + out_tok
            self._write(run_id, in_tok, out_tok, "ok")

        def on_llm_error(self, error, *, run_id, **kwargs):
            # Type name only: provider error text can carry request details or keys
            self._write(run_id, 0, 0, f"error: {type(error).__name__}")

        def _write(self, run_id, in_tok, out_tok, status):
            self.calls += 1
            start, prompt = self._starts.pop(run_id, (time.perf_counter(), None))
            store.record(request_id=self.request_id, provider=provider, model=model,
                         input_tokens=in_tok, output_tokens=out_tok,
                         latency_ms=int((time.perf_counter() - start) * 1000),
                         status=status, prompt_text=scrub(prompt, 4000) if prompt else None)

    return UsageCallback()


def _tokens(response) -> Tuple[int, int]:
    """Input/output tokens from an LLMResult (message usage_metadata, else llm_output token_usage)."""
    in_tok = out_tok = 0
    for gens in response.generations or []:
        for g in gens:
            um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
            in_tok += int(um.get("input_tokens", 0) or 0)
            out_tok += int(um.get("output_tokens", 0) or 0)
    if not (in_tok or out_tok) and response.llm_output:
        tu = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
        in_tok = int(tu.get("prompt_tokens", tu.get("input_tokens", 0)) or 0)
        out_tok = int(tu.get("completion_tokens", tu.get("output_tokens", 0)) or 0)
    return in_tok, out_tok
