"""tools/finetune/curate (job 016): rules, manifest, split, dry-run, originals untouched."""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.finetune import curate  # noqa: E402


def _noise(seed, h=600, w=800, lo=20, hi=230):
    return np.random.default_rng(seed).integers(lo, hi, (h, w, 3), dtype=np.uint8)


def _write(path, arr):
    cv2.imencode(path.suffix, arr)[1].tofile(str(path))


@pytest.fixture
def photos(tmp_path):
    d = tmp_path / "in"
    (d / "sub").mkdir(parents=True)
    _write(d / "good1.png", _noise(1))
    _write(d / "good2.jpg", _noise(2))
    _write(d / "sub" / "good3.png", _noise(3))
    dup = np.clip(_noise(1).astype(int) + 3, 0, 255).astype(np.uint8)  # brightened copy of good1
    _write(d / "zz_dup.png", dup)
    _write(d / "tiny.png", _noise(4, 300, 400))
    smooth = np.tile(np.linspace(60, 190, 800, dtype=np.uint8)[None, :, None], (600, 1, 3))
    _write(d / "blurry.png", cv2.GaussianBlur(smooth, (0, 0), 8))
    clipped = _noise(5)
    clipped[:300] = 255
    _write(d / "clipped.png", clipped)
    (d / "broken.jpg").write_bytes(b"not an image")
    _write(d / "deep16.tif", (_noise(6).astype(np.uint16) * 257))
    (d / "notes.txt").write_text("ignored")
    return d


def _run(*argv):
    return curate.curate(curate.build_parser().parse_args([str(a) for a in argv]))


def _by_file(manifest):
    return {r["file"]: r for r in manifest["images"]}


def test_rules_reasons_and_outputs(photos, tmp_path, capsys):
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in photos.rglob("*") if p.is_file()}
    out = tmp_path / "out"
    m = _run(photos, out)
    recs = _by_file(m)
    assert {f for f, r in recs.items() if r["status"] == "kept"} == {"good1.png", "good2.jpg", "sub/good3.png", "deep16.tif"}
    assert recs["tiny.png"]["reason"].startswith("tiny")
    assert recs["blurry.png"]["reason"].startswith("blurry")
    assert recs["clipped.png"]["reason"].startswith("clipped highlights")
    assert recs["broken.jpg"]["reason"] == "unreadable"
    assert recs["zz_dup.png"]["reason"].startswith("near-duplicate of good1.png")
    assert "notes.txt" not in recs
    for r in recs.values():
        assert len(r["sha256"]) == 64 and r["status"] in ("kept", "dropped") and r["reason"]
    assert set(recs["good1.png"]["scores"]) >= {"width", "height", "blur", "highlight_share", "shadow_share", "phash"}

    on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["counts"]["kept"] == 4 and on_disk["thresholds"]["min_side"] == 512
    for r in on_disk["images"]:
        if r["status"] == "kept":
            assert (out / r["out"]).is_file() and r["out"].startswith(r["split"] + "/")
    # 16-bit TIFF -> 8-bit PNG in the output only
    deep = recs["deep16.tif"]
    assert deep["bit_depth"] == 16 and deep["out"].endswith(".png")
    arr = cv2.imdecode(np.fromfile(str(out / deep["out"]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert arr.dtype == np.uint8 and arr.shape == (600, 800, 3)
    # originals unchanged
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in photos.rglob("*") if p.is_file()}
    assert after == before
    assert "fewer than 200" in capsys.readouterr().err


def test_dry_run_writes_nothing(photos, tmp_path):
    out = tmp_path / "out"
    m = _run(photos, out, "--dry-run")
    assert m["counts"]["kept"] == 4
    assert not out.exists()


def test_refuses_overlap_and_non_empty_out(photos, tmp_path):
    with pytest.raises(SystemExit, match="overlaps"):
        _run(photos, photos / "out")
    with pytest.raises(SystemExit, match="overlaps"):
        _run(photos / "sub", photos)
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        _run(photos, full)


def test_thresholds_are_flags():
    out = subprocess.run([sys.executable, "-m", "tools.finetune.curate", "--help"], cwd=_ROOT,
                         env={**os.environ, "COLUMNS": "250"}, capture_output=True, text=True, check=True).stdout
    for flag, default in (("--min-side", "512"), ("--blur", "50.0"), ("--max-highlight", "0.1"),
                          ("--max-shadow", "0.25"), ("--dup-distance", "6"), ("--seed", "0")):
        assert flag in out and f"(default: {default})" in out


def test_flags_change_rules(photos, tmp_path):
    recs = _by_file(_run(photos, tmp_path / "o", "--min-side", "200", "--max-highlight", "0.6", "--dry-run"))
    assert recs["tiny.png"]["status"] == "kept" and recs["clipped.png"]["status"] == "kept"


def test_split_disjoint_reproducible_by_image():
    names = [f"img{i:03d}.jpg" for i in range(100)]
    a, b = curate.split_names(names, 7), curate.split_names(list(reversed(names)), 7)
    assert a == b
    assert [len(a[s]) for s in ("train", "val", "test")] == [80, 10, 10]
    sets = [set(a[s]) for s in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set().union(*sets) == set(names)
    assert curate.split_names(names, 8) != a


def test_phash_and_hamming():
    g = cv2.cvtColor(_noise(1), cv2.COLOR_RGB2GRAY)
    assert curate.hamming(curate.phash(g), curate.phash(g)) == 0
    other = cv2.cvtColor(_noise(2), cv2.COLOR_RGB2GRAY)
    assert curate.hamming(curate.phash(g), curate.phash(other)) > 6
