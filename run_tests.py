"""
run_tests.py
Unified QA & Software Reliability Test Runner for Efface Magique LR.

Executes all test suites:
- tests/test_engine.py  (AI Inpainting & Blending Engine)
- tests/test_gui.py     (Headless PyQt6 Canvas & Interaction logic)
- tests/test_plugin.py  (Lua static syntax validation & SDK mocking)
- tests/test_e2e.py     (End-to-end CLI headless pipeline)

Outputs formatted reports and code coverage metrics.
"""

import os
import sys
import time
import subprocess
import argparse

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run Efface Magique LR Automated Test Suite")
    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        choices=["all", "engine", "gui", "plugin", "e2e", "firefly", "subject", "live", "layers", "perf", "opt"],
        help="Test suite to run (default: all)",
    )
    parser.add_argument(
        "--no-cov",
        action="store_true",
        help="Disable code coverage generation",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Run pytest in verbose mode (default: True)",
    )
    args = parser.parse_args()

    # Map suite to test paths
    suite_map = {
        "all": [
            "tests/test_engine.py",
            "tests/test_gui.py",
            "tests/test_plugin.py",
            "tests/test_e2e.py",
            "tests/test_firefly_pipeline.py",
            "tests/test_subject_detector.py",
            "tests/test_live_bridge.py",
            "tests/test_layers.py",
            "tests/test_optimizations.py",
            "tests/test_pipeline_perf.py",
        ],
        "engine":  ["tests/test_engine.py"],
        "gui":     ["tests/test_gui.py"],
        "plugin":  ["tests/test_plugin.py"],
        "e2e":     ["tests/test_e2e.py"],
        "firefly": ["tests/test_firefly_pipeline.py"],
        "subject": ["tests/test_subject_detector.py"],
        "live":    ["tests/test_live_bridge.py"],
        "layers":  ["tests/test_layers.py"],
        "opt":     ["tests/test_optimizations.py"],
        "perf":    ["tests/test_pipeline_perf.py"],
    }

    test_targets = suite_map[args.suite]

    print("=" * 80)
    print("      EFFACE MAGIQUE LR - AUTOMATED TEST SUITE & QUALITY GATE")
    print("=" * 80)
    print(f"[*] Python Interpreter : {sys.executable}")
    print(f"[*] Project Root       : {_PROJECT_ROOT}")
    print(f"[*] Target Suite       : {args.suite.upper()} ({', '.join(test_targets)})")
    print(f"[*] Coverage Report    : {'Disabled' if args.no_cov else 'Enabled (companion/)'}")
    print("=" * 80)
    print()

    pytest_cmd = [sys.executable, "-m", "pytest"]
    if args.verbose:
        pytest_cmd.append("-v")

    # Add coverage flags if enabled
    if not args.no_cov:
        pytest_cmd.extend([
            "--cov=companion",
            "--cov-report=term-missing",
        ])

    pytest_cmd.extend(test_targets)

    # Reconfigure stdout for utf-8 if possible
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    start_time = time.time()
    exit_code = subprocess.call(pytest_cmd, cwd=_PROJECT_ROOT)
    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    if exit_code == 0:
        print(f" [PASS] QUALITY GATE PASSED: All tests succeeded in {elapsed:.2f}s!")
    else:
        print(f" [FAIL] QUALITY GATE FAILED: Pytest exited with code {exit_code} after {elapsed:.2f}s.")
    print("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
