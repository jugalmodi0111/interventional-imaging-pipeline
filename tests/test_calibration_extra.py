"""Stage 2.5 calibration: reliability curve, temperature scaling, OOD-AUROC. Pure numpy."""
import numpy as np

from src.eval.calibration import (apply_temperature, auroc, ece, ood_auroc,
                                   reliability_curve, temperature_scale, uncertainty_score)


def test_reliability_curve_bins_cover_and_count():
    p = np.array([0.05, 0.2, 0.9, 0.95])
    y = np.array([0, 0, 1, 1])
    rows = reliability_curve(p, y, n_bins=10)
    assert len(rows) == 10
    assert sum(r["count"] for r in rows) == 4               # every sample counted once
    empty = [r for r in rows if r["count"] == 0]
    assert all(r["conf"] is None and r["acc"] is None for r in empty)


def test_temperature_scaling_lowers_ece_on_overconfident_logits():
    rng = np.random.default_rng(1)
    z = rng.normal(size=5000) * 2.0
    y = (rng.uniform(size=5000) < 1 / (1 + np.exp(-z))).astype(int)
    over = z * 3.0                                           # deliberately over-confident
    T = temperature_scale(over, y)
    ece_pre = ece(1 / (1 + np.exp(-over)), y)
    ece_post = ece(apply_temperature(over, T), y)
    assert T > 1.0                                          # must cool down over-confidence
    assert ece_post < ece_pre                               # calibration improved
    assert ece_post < 0.05                                  # Stage 2.5 exit gate


def test_temperature_of_one_is_noop():
    z = np.array([-2.0, 0.0, 2.0])
    assert np.allclose(apply_temperature(z, 1.0), 1 / (1 + np.exp(-z)))


def test_auroc_perfect_and_random():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0     # perfectly separable
    assert auroc([1.0, 1.0, 1.0, 1.0], [0, 1, 0, 1]) == 0.5     # all-tie -> chance
    assert np.isnan(auroc([0.1, 0.2], [0, 0]))                  # one class -> undefined


def test_ood_auroc_separates_uncertain_ood_from_confident_id():
    rng = np.random.default_rng(2)
    id_p = rng.uniform(0.9, 1.0, size=400)                  # confident in-distribution
    ood_p = rng.uniform(0.45, 0.55, size=400)               # unsure OOD
    a = ood_auroc(uncertainty_score(id_p), uncertainty_score(ood_p))
    assert a > 0.95                                         # defer path clearly flags OOD


def test_uncertainty_score_two_sided():
    # 0.5 -> maximally unsure (1.0); 0/1 -> certain (0.0); symmetric around 0.5
    assert np.isclose(uncertainty_score([0.5])[0], 1.0)
    assert np.isclose(uncertainty_score([0.0])[0], 0.0)
    assert np.isclose(uncertainty_score([1.0])[0], 0.0)
    assert np.isclose(uncertainty_score([0.2])[0], uncertainty_score([0.8])[0])
