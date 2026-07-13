"""CLGeoDice loss + metric (Chen et al. 2026). Pure-config tests always run; the numpy metric
needs skimage/scipy and the torch loss needs torch — both importorskip so a torch-free box still
passes (mirrors the repo's other heavy-dep tests)."""
import numpy as np
import pytest


def _bar(H=64, row=None, cols=(8, 56)):
    """A horizontal 2px 'vessel' bar — a simple thin tubular structure."""
    m = np.zeros((H, H), np.uint8)
    r = H // 2 if row is None else row
    m[r:r + 2, cols[0]:cols[1]] = 1
    return m


def test_distill_kwargs_reads_clgeodice_weight():
    # torch-free: the config plumbing must expose the weight, default OFF for backward compat.
    from src.train.train_seg import distill_kwargs
    kw = distill_kwargs({"distill": {"clgeodice_weight": 0.5, "clgeodice_r_th": 6}})
    assert kw["clgeo_weight"] == 0.5 and kw["clgeo_r_th"] == 6
    assert distill_kwargs({})["clgeo_weight"] == 0.0            # KD-only unless enabled


def test_distance_heatmap_peaks_on_skeleton_and_decays():
    pytest.importorskip("skimage"); pytest.importorskip("scipy")
    from src.eval.metrics import vessel_distance_heatmap
    d, s = vessel_distance_heatmap(_bar(), r_th=8)
    assert 0.0 <= d.min() and d.max() <= 1.0
    assert d[s > 0].mean() > d[s == 0].mean()                  # heatmap higher on the centerline
    dz, sz = vessel_distance_heatmap(np.zeros((32, 32), np.uint8))
    assert dz.sum() == 0 and sz.sum() == 0                     # empty GT -> zeros


def test_clgeodice_metric_perfect_disjoint_empty():
    pytest.importorskip("skimage"); pytest.importorskip("scipy")
    from src.eval.metrics import clgeodice
    g = _bar()
    assert clgeodice(g, g) > 0.9                               # perfect overlap -> high
    far = _bar(row=4)                                          # a bar shoved to the edge
    assert clgeodice(far, g) < clgeodice(g, g)                 # geometry penalty
    assert np.isnan(clgeodice(np.zeros_like(g), np.zeros_like(g)))   # empty GT -> NaN (nanmean-able)


def test_clgeodice_penalises_in_lumen_drift_clDice_blindspot():
    # The whole point vs clDice: a small centerline shift should not score equal-to-perfect.
    pytest.importorskip("skimage"); pytest.importorskip("scipy")
    from src.eval.metrics import clgeodice
    g = _bar()
    assert clgeodice(g, g) >= clgeodice(np.roll(g, 2, axis=0), g)


def test_clgeodice_loss_runs_and_backprops():
    # skip on the absent deps FIRST so a torch-only box doesn't import torch (would pollute the
    # `torch not in sys.modules` guardrail in test_train_seg).
    pytest.importorskip("skimage"); pytest.importorskip("scipy")
    torch = pytest.importorskip("torch")
    from src.models.clgeodice import clgeodice_loss
    gt = torch.zeros(2, 1, 64, 64); gt[:, :, 31:33, 8:56] = 1
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    loss = clgeodice_loss(pred, gt, r_th=8)
    assert 0.0 <= float(loss) <= 1.0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()   # gradients flow through pred
