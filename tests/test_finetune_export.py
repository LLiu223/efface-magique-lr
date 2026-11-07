"""tools/finetune/export (job 016): meta.json schema, base-model guard, TorchScript round trip (torch, opt-in)."""

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.finetune import export  # noqa: E402

needs_torch = pytest.mark.skipif(os.environ.get("FINETUNE_TORCH_TESTS") != "1",
                                 reason="torch export test; set FINETUNE_TORCH_TESTS=1 to run")


def test_meta_schema():
    meta = export.build_meta("mine", "big-lama/models/best.ckpt", "ab" * 32, 750, 1e-5, date="2026-11-07T10:00:00")
    assert tuple(meta) == export.META_KEYS
    assert json.loads(json.dumps(meta)) == meta
    for bad in ({**meta, "steps": "750"}, {**meta, "metrics": []}, {k: v for k, v in meta.items() if k != "lr"}, [],
                {**meta, "steps": True}):
        with pytest.raises(ValueError):
            export.validate_meta(bad)


def test_meta_keys_match_app():
    from companion.models import tuned
    assert tuned.META_KEYS == export.META_KEYS


def test_refuses_base_model(tmp_path):
    base = tmp_path / "hub" / "big-lama.pt"
    base.parent.mkdir()
    base.write_bytes(b"base")
    assert export.refuses_base(base, base)  # OUT_DIR given as the file itself
    assert not export.refuses_base(tmp_path / "out", base)
    link = tmp_path / "linked"
    link.mkdir()
    try:
        os.link(base, link / "model.pt")  # OUT_DIR/model.pt is the same file under another name
    except OSError:
        pytest.skip("hard links not supported here")
    assert export.refuses_base(link, base)
    with pytest.raises(SystemExit, match="Refusing"):
        export.main([str(tmp_path / "x.ckpt"), str(link), "--lama-dir", ".", "--manifest", "m", "--base-pt", str(base)])
    assert base.read_bytes() == b"base"


@needs_torch
def test_export_round_trip_and_failure_leaves_nothing(tmp_path):
    import torch
    from companion.models import tuned

    class Gen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 3, 3, padding=1)

        def forward(self, x):
            return torch.sigmoid(self.conv(x))

    meta = export.build_meta("t", "base", "0" * 64, 1, 1e-5)
    root = tmp_path / "finetuned"
    export.export_generator(Gen(), root / "t", meta, tmp_path / "big-lama.pt")
    assert sorted(p.name for p in (root / "t").iterdir()) == ["meta.json", "model.pt"]
    [cls] = tuned.discover(root)
    assert cls.is_available(), cls.unavailable_reason()

    with pytest.raises(Exception):
        export.export_generator(torch.nn.Identity(), root / "bad", meta, tmp_path / "big-lama.pt")  # 4-channel output
    assert sorted(p.name for p in root.iterdir()) == ["t"]  # no OUT_DIR and no temp folder left behind
