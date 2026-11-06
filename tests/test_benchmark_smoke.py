"""Smoke test for tools/benchmark.py at a tiny size (the real 12/24/60 MP run is manual)."""

import os
import subprocess
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.model_engine import EngineMode  # noqa: E402
from tools import benchmark  # noqa: E402


def test_benchmark_stub_runs_and_labels_stub():
    out = subprocess.run(
        [sys.executable, os.path.join(_PROJECT_ROOT, "tools", "benchmark.py"), "--stub", "--sizes", "0.3"],
        cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert "STUB ENGINE" in out.stdout
    rows = [ln for ln in out.stdout.splitlines() if ln.startswith("| STUB |")]
    stages = {ln.split("|")[3].strip() for ln in rows}
    assert {"dilate", "crop", "inpaint", "blend", "paste", "save_tiff8", "save_tiff16", "total"} <= stages
    assert "peak tracemalloc MiB" in out.stdout
    assert all(ln.startswith("| STUB |") for ln in out.stdout.splitlines() if ln.startswith("| ") and "MP (" in ln)


def test_missing_weights_without_stub_exits_nonzero():
    env = dict(os.environ, LAMA_MODEL=os.path.join(_PROJECT_ROOT, "no-such-dir", "big-lama.pt"))
    out = subprocess.run(
        [sys.executable, os.path.join(_PROJECT_ROOT, "tools", "benchmark.py"), "--sizes", "0.3"],
        cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=300, env=env,
    )
    assert out.returncode != 0
    assert "weights not found" in out.stderr and "--stub" in out.stderr
    assert "| STUB |" not in out.stdout


def test_run_stages_matches_generate_variations():
    """The timed stages must produce exactly what the real engine path produces."""
    image, mask = benchmark.make_inputs(*benchmark.size_for(0.3))
    for mode in EngineMode:
        engine = benchmark.make_engine(stub=True, mode=mode)
        staged, _ = benchmark.run_stages(engine, image, mask)
        real = engine.generate_variations(image, mask, num_variations=1, base_seed=benchmark.SEED)[0]
        assert np.array_equal(np.asarray(staged), np.asarray(real))
