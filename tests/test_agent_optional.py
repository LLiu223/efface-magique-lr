"""(f) The app imports, starts and passes existing tests with LangChain blocked."""

import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS = os.path.join(_PROJECT_ROOT, "tests")

_SCRIPT = r"""
import os, sys
sys.path[:0] = [ROOT, TESTS]
import agent_blocker  # noqa: F401  (installs the sys.meta_path blocker)
from PyQt6.QtWidgets import QApplication
app = QApplication([])
import companion.app as capp
import companion.agent.harness, companion.agent.tools, companion.agent.usage, companion.agent.usage_dialog
from companion.agent import agent_available, MISSING_TOOLTIP
from companion.agent.llm import make_llm
assert not agent_available()
assert not any(m.split(".")[0] in agent_blocker.BLOCKED for m in sys.modules), "LangChain imported at module level"
w = capp.MainWindow()
for widget in (w.agent_input, w.combo_agent_provider, w.ed_agent_model, w.btn_agent_ask):
    assert not widget.isEnabled() and widget.toolTip() == MISSING_TOOLTIP
w._on_agent_ask()  # no-op, no crash
assert w.worker is None
try:
    make_llm("ollama", "x")
    raise SystemExit("make_llm should fail without langchain_ollama")
except ImportError:
    pass
w.close()
print("BLOCKED-OK")
""".replace("ROOT", repr(_PROJECT_ROOT)).replace("TESTS", repr(_TESTS))


def _env():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env["PYTHONPATH"] = os.pathsep.join([_TESTS, _PROJECT_ROOT, env.get("PYTHONPATH", "")])
    return env


def test_app_starts_with_langchain_blocked():
    r = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=300,
                       cwd=_PROJECT_ROOT, env=_env())
    assert "BLOCKED-OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]


def test_existing_gui_tests_pass_with_langchain_blocked():
    # A representative slice; the full suite is run the same way in build.md (-p agent_blocker tests)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "agent_blocker", "-q", "-p", "no:cacheprovider",
         "tests/test_photo_switching.py", "tests/test_model_registry.py", "tests/test_agent_usage.py",
         "tests/test_agent_dialog.py"],
        capture_output=True, text=True, timeout=600, cwd=_PROJECT_ROOT, env=_env())
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert " passed" in r.stdout
