"""tools/finetune/masks (job 016): seeded, reproducible, coverage per regime, LaMa mask format."""

import os
import sys

import numpy as np
import pytest
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.finetune import masks, train, eval as ft_eval  # noqa: E402
from tools.finetune.masks import generate_mask  # noqa: E402

SIZES = [(512, 512), (300, 700), (777, 513)]


@pytest.mark.parametrize("regime", ["normal", "small"])
def test_same_seed_byte_identical(regime):
    for h, w in SIZES:
        a, b = generate_mask(h, w, 42, regime), generate_mask(h, w, 42, regime)
        assert a.tobytes() == b.tobytes()
        assert a.shape == (h, w) and a.dtype == np.uint8 and set(np.unique(a)) <= {0, 255}
    assert generate_mask(512, 512, 1, regime).tobytes() != generate_mask(512, 512, 2, regime).tobytes()


def test_coverage_ranges():
    normal = [generate_mask(h, w, s, "normal").mean() / 255 for s in range(15) for h, w in SIZES]
    small = [generate_mask(h, w, s, "small").mean() / 255 for s in range(15) for h, w in SIZES]
    assert 0.05 <= min(normal) and max(normal) <= 0.40
    assert max(small) < 0.05 and min(small) > 0
    assert np.mean(small) < np.mean(normal)


def test_lama_mask_format(tmp_path):
    m = generate_mask(64, 96, 0)
    path = tmp_path / masks.mask_filename("abc")
    masks.write_mask(path, m)
    assert path.name == "abc_mask000.png"
    with Image.open(path) as im:
        assert im.mode == "L" and im.size == (96, 64)
        assert np.array_equal(np.asarray(im), m)


def test_train_and_eval_share_the_function():
    assert ft_eval.generate_mask is masks.generate_mask
    assert "generate_mask" in open(train.__file__, encoding="utf-8").read()
