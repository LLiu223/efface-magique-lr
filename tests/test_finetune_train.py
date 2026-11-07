"""tools/finetune/train (job 016): dry-run command and overrides, GPU checks, CLI --help for every script."""

import json
import os
import subprocess
import sys
import types

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.finetune import train  # noqa: E402

_DRY_RUN = """
import runpy, sys
sys.argv = ["train"] + sys.argv[1:]
try:
    runpy.run_module("tools.finetune.train", run_name="__main__")
except SystemExit as e:
    code = e.code
print("TORCH_IMPORTED=" + str("torch" in sys.modules))
sys.exit(code)
"""


@pytest.fixture
def setup_dirs(tmp_path):
    lama, big, data = tmp_path / "lama", tmp_path / "big-lama", tmp_path / "curated"
    (lama / "bin").mkdir(parents=True)
    (lama / "bin" / "train.py").write_text("")
    (big / "models").mkdir(parents=True)
    (big / "models" / "best.ckpt").write_bytes(b"")
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps({"counts": {"train": 40}, "images": []}))
    return ["--lama-dir", str(lama), "--big-lama", str(big), "--data", str(data), "--work", str(tmp_path / "work")]


def _dry_run(argv):
    """Run train --dry-run in a fresh interpreter; return (process, torch_was_not_imported)."""
    p = subprocess.run([sys.executable, "-c", _DRY_RUN, *argv, "--dry-run"], cwd=_ROOT, capture_output=True, text=True)
    return p, "TORCH_IMPORTED=False" in p.stdout


def test_dry_run_prints_upstream_command_without_torch(setup_dirs, tmp_path):
    p, no_torch = _dry_run(setup_dirs + ["--name", "mine", "--ckpt-every", "5"])
    assert p.returncode == 0, p.stderr
    assert no_torch
    out = p.stdout
    assert "bin/train.py -cn big-lama" in out
    for o in ("location.pretrained_models=", "optimizers.generator.lr=1e-05", "optimizers.discriminator.lr=1e-06",
              "trainer.kwargs.max_epochs=3", "data.batch_size=4", "data.train.out_size=256",
              "trainer.kwargs.val_check_interval=5", "trainer.kwargs.limit_train_batches=10",
              "+trainer.kwargs.resume_from_checkpoint=", "init.ckpt", "hydra.run.dir=", "run_title=mine"):
        assert o in out, o
    assert "nothing written" in out
    assert not (tmp_path / "work").exists()


def test_build_command_resume_and_clamp(setup_dirs, tmp_path):
    a = train.build_parser().parse_args(setup_dirs + ["--name", "r1", "--lr", "2e-5", "--batch-size", "8"])
    argv, env = train.build_command(a, n_train=100)
    assert argv[1:4] == ["bin/train.py", "-cn", "big-lama"]
    assert "trainer.kwargs.val_check_interval=13" in argv  # 500 clamped to one epoch of 13 batches
    assert "optimizers.generator.lr=2e-05" in argv
    assert env["TORCH_HOME"] == env["PYTHONPATH"]
    assert any(x.endswith("/r1/init.ckpt") for x in argv)
    last = tmp_path / "work" / "r1" / "models" / "last.ckpt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"")
    argv, _ = train.build_command(a, n_train=100)
    assert any(x.startswith("+trainer.kwargs.resume_from_checkpoint=") and x.endswith("/r1/models/last.ckpt")
               for x in argv)


def test_dry_run_reports_missing_paths(tmp_path):
    p, _ = _dry_run(["--lama-dir", str(tmp_path), "--big-lama", str(tmp_path), "--data", str(tmp_path),
                     "--work", str(tmp_path / "w"), "--name", "bad name"])
    assert p.returncode == 2
    for msg in ("bin/train.py not found", "best.ckpt not found", "manifest.json not found", "[A-Za-z0-9_-]"):
        assert msg in p.stderr


def _fake_torch(cuda, gb=0):
    props = types.SimpleNamespace(total_memory=int(gb * 1024 ** 3))
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda,
                                                            get_device_properties=lambda i: props))


def test_check_gpu_messages(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    msg = train.check_gpu(12)
    assert "No CUDA GPU" in msg and "--batch-size" in msg and "--crop" in msg
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, 8))
    msg = train.check_gpu(12)
    assert "8.0 GB" in msg and "--batch-size" in msg
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, 24))
    assert train.check_gpu(12) is None


def test_real_run_fails_early_without_gpu(setup_dirs, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    monkeypatch.setattr(train, "prepare_data", lambda a: pytest.fail("must fail before preparing data"))
    assert train.main(setup_dirs) == 3
    assert "No CUDA GPU" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["", "café", "١٢", "a.b", "a/b", "ok\n"])
def test_name_uses_registry_charset(setup_dirs, name):
    errors, _ = train.validate(train.build_parser().parse_args(setup_dirs + ["--name", name]))
    assert any("--name" in e for e in errors)


def test_invalid_manifest_is_a_clean_error(setup_dirs, tmp_path):
    (tmp_path / "curated" / "manifest.json").write_text("{not json")
    p, _ = _dry_run(setup_dirs)
    assert p.returncode == 2 and "manifest.json is unreadable" in p.stderr and "Traceback" not in p.stderr


@pytest.mark.parametrize("name", ["curate", "masks", "train", "export", "eval"])
def test_help(name):
    p = subprocess.run([sys.executable, "-m", f"tools.finetune.{name}", "--help"], cwd=_ROOT,
                       capture_output=True, text=True)
    assert p.returncode == 0 and "usage:" in p.stdout, p.stderr
