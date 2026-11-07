"""tools/finetune/eval (job 016): metric functions on known inputs and the verdict rule."""

import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.finetune import eval as ft_eval  # noqa: E402


def _img(seed=0):
    return np.random.default_rng(seed).integers(30, 220, (64, 80, 3), dtype=np.uint8)


def test_psnr_known_values():
    a = _img()
    assert ft_eval.psnr(a, a) == float("inf")
    b = (a.astype(int) + 10).astype(np.uint8)
    assert math.isclose(ft_eval.psnr(a, b), 20 * math.log10(255 / 10), rel_tol=1e-9)


def test_psnr_in_mask_only_counts_masked_pixels():
    a = _img()
    b = a.copy()
    mask = np.zeros(a.shape[:2], np.uint8)
    mask[:, :40] = 255
    b[:, 40:] = 0  # damage outside the mask only
    assert ft_eval.psnr(a, b, mask) == float("inf")
    assert ft_eval.psnr(a, b) < 20


def test_ssim_known_values():
    a = _img()
    assert math.isclose(ft_eval.ssim(a, a), 1.0, rel_tol=1e-9)
    assert ft_eval.ssim(a, _img(1)) < 0.2
    mask = np.zeros(a.shape[:2], np.uint8)
    mask[20:40, 20:40] = 255
    b = a.copy()
    b[:5] = 0
    assert math.isclose(ft_eval.ssim(a, b, mask), 1.0, rel_tol=1e-9)
    assert ft_eval.ssim(a, b) < 1.0


def _summary(b_lp, c_lp, b_psnr, c_psnr, wins):
    return {"baseline": {"lpips_mask": b_lp, "psnr_mask": b_psnr},
            "candidate": {"lpips_mask": c_lp, "psnr_mask": c_psnr}, "wins": wins}


def test_verdict_rule():
    assert ft_eval.verdict(_summary(0.20, 0.19, 25.0, 25.0, 6), 10)[0] == "ADOPT"
    assert ft_eval.verdict(_summary(0.20, 0.198, 25.0, 25.0, 6), 10)[0] == "KEEP BASELINE"  # 1 % gain < 2 %
    assert ft_eval.verdict(_summary(0.20, 0.19, 25.0, 24.7, 6), 10)[0] == "KEEP BASELINE"  # PSNR -0.3 dB
    assert ft_eval.verdict(_summary(0.20, 0.19, 25.0, 25.0, 5), 10)[0] == "KEEP BASELINE"  # no majority
    decision, reasons = ft_eval.verdict(_summary(0.20, 0.19, 25.0, 25.0, 6), 10, min_lpips_gain=0.1)
    assert decision == "KEEP BASELINE" and reasons[0].startswith("FAIL")


def test_summarize_and_reports(tmp_path):
    keys = {f"{who}_{m}_{reg}": 1.0 for who in ("baseline", "candidate") for m in ft_eval.METRICS for reg in ("mask", "full")}
    per = [{"file": "a.png", **keys, "candidate_wins": True}, {"file": "b.png", **keys, "candidate_wins": False}]
    s = ft_eval.summarize(per)
    assert s["wins"] == 1 and s["baseline"]["lpips_mask"] == 1.0
    args = ft_eval.argparse.Namespace(baseline="big-lama.pt", candidate="model.pt", data="test", seed=1000,
                                      regime="normal", min_lpips_gain=0.02, max_psnr_drop=0.2)
    decision, reasons = ft_eval.verdict(s, 2)
    ft_eval.write_reports(tmp_path, args, per, s, decision, reasons)
    assert (tmp_path / "report.md").read_text(encoding="utf-8").rstrip().endswith("**Verdict: KEEP BASELINE**")
    assert '"verdict": "KEEP BASELINE"' in (tmp_path / "report.json").read_text(encoding="utf-8")
    ft_eval.contact_sheet([(_img(), _img(1), _img(2))], tmp_path / "sheet.jpg")
    assert (tmp_path / "sheet.jpg").stat().st_size > 0
