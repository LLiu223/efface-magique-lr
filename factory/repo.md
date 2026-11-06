# Repo context (read by every agent)
- Project: Efface Magique LR — local AI eraser for Lightroom Classic. Python 3 + PyQt6 companion (`companion/`), Lua plugin (`plugin/`).
- Install: `pip install -r requirements.txt -r requirements-dev.txt` (a `.venv` exists in the repo root).
- Tests: `python -m pytest tests -q` (pytest + pytest-qt, GUI tests run headless). `python run_tests.py` runs the full suite with coverage.
- Layout: `companion/` app code, `companion/utils/` helpers, `plugin/` Lua LiveBridge (HTTP IPC to the companion), `tests/test_*.py`.
- Conventions: conventional commits (`feat:`, `fix:`); architecture notes in `gemini.md` and `README.md`; never commit images (`*.tif`, `*.png` ignored) or `.venv`.
- Never touch: `dist/`, `.venv/`, release scripts unless the spec says so.
