"""Chat-model factory for the assistant. Keys come only from environment variables."""

import os
import re
from typing import Optional
from urllib.parse import urlparse

PROVIDERS = ("ollama", "anthropic", "openai", "gemini")
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY"}
DEFAULT_OLLAMA_URL = "http://localhost:11434"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
# Common provider key shapes, in case a key that is not in our env vars shows up in error text
_KEY_PATTERNS = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,})")


class AgentConfigError(ValueError):
    """User-facing, one-line configuration problem (bad provider, missing key)."""


# shortcut: a hand-kept list of model ids, will date; the picker stays editable for any other id
CLOUD_MODELS = {
    "anthropic": ["claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
}


def is_local_url(url: str) -> bool:
    return (urlparse(url).hostname or "").lower() in _LOCAL_HOSTS


def _fetch_tags(url: str, timeout: float) -> bytes:
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - local URL, checked by the caller
        return resp.read(1_000_000)


def list_ollama_models(base_url: Optional[str] = None, timeout: float = 1.0) -> list:
    """Names of the models installed in a local Ollama (GET /api/tags). Raises OSError/ValueError when unreachable."""
    base = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
    if not is_local_url(base):
        raise ValueError(f"{base} is not localhost; the model list is only read from a local Ollama")
    import json
    data = json.loads(_fetch_tags(base + "/api/tags", timeout))
    return sorted({str(m["name"]) for m in data.get("models", []) if isinstance(m, dict) and m.get("name")})


def scrub(text: object, max_len: int = 300) -> str:
    """Remove API keys from text before it is logged, stored or shown; collapse to one line."""
    s = str(text)
    for env in KEY_ENV.values():
        key = os.environ.get(env)
        if key and len(key) >= 4:
            s = s.replace(key, "***")
    s = _KEY_PATTERNS.sub("***", s)
    return " ".join(s.split())[:max_len]


def make_llm(provider: str, model: str, *, base_url: Optional[str] = None, timeout: float = 60.0, **opts):
    """Return a LangChain chat model for ollama / anthropic / openai / gemini.

    SDK-level retries are disabled: the harness owns the single retry with backoff.
    """
    provider = (provider or "").strip().lower()
    if provider not in PROVIDERS:
        raise AgentConfigError(f"Unknown assistant provider {provider!r}; use one of: {', '.join(PROVIDERS)}.")
    if not model:
        raise AgentConfigError("Enter an assistant model name first.")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=base_url or DEFAULT_OLLAMA_URL,
                          client_kwargs={"timeout": timeout}, **opts)
    env = KEY_ENV[provider]
    key = os.environ.get(env, "").strip()
    if not key:
        raise AgentConfigError(f"Set the {env} environment variable to use {provider}.")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=key, timeout=timeout, max_retries=0, **opts)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=key, timeout=timeout, max_retries=0, **opts)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, api_key=key, request_timeout=timeout, retries=0, **opts)
