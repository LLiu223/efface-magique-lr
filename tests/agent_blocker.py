"""Pytest plugin / importable module that makes the optional LangChain stack look uninstalled.

Run the whole suite as if requirements-agent.txt were absent:
    PYTHONPATH=tests python -m pytest -p agent_blocker tests -q
"""

import sys

BLOCKED = ("langchain", "langchain_core", "langgraph", "langchain_ollama", "langchain_anthropic",
           "langchain_openai", "langchain_google_genai")


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"{name} blocked (simulating: requirements-agent.txt not installed)")
        return None


def install():
    for mod in [m for m in sys.modules if m.split(".")[0] in BLOCKED]:
        del sys.modules[mod]
    if not any(isinstance(f, _Blocker) for f in sys.meta_path):
        sys.meta_path.insert(0, _Blocker())


install()
