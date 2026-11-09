"""Optional natural-language assistant (job 015).

Everything here is optional: langchain/langgraph are imported only inside functions,
so the app imports and runs unchanged when requirements-agent.txt is not installed.
"""

MISSING_TOOLTIP = "install requirements-agent.txt"

# QSettings keys (store: companion.app._open_settings())
KEY_PROVIDER = "agent/provider"
KEY_MODEL = "agent/model"
KEY_OLLAMA_URL = "agent/ollama_url"
KEY_TOKEN_BUDGET = "agent/token_budget"
KEY_STORE_PROMPTS = "agent/store_prompts"
DEFAULT_PROVIDER = "ollama"
DEFAULT_AGENT_MODEL = "qwen3:8b"
# Listed first in the Ollama picker even before they are pulled: (name, note shown next to it)
RECOMMENDED_OLLAMA_MODELS = (("qwen3:8b", "recommended, 5.2 GB"), ("qwen3.5:4b", "light, 3.3 to 4 GB"))


def agent_available() -> bool:
    """True when the optional LangChain stack is installed (spec lookup only; no slow import at startup)."""
    import importlib.util
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("langchain", "langchain_core", "langgraph"))
    except Exception:  # ImportError/ValueError from a broken install or an import blocker
        return False
