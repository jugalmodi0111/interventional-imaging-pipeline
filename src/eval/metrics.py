"""Segmentation metrics: Dice, clDice (connectivity), HD95. Functional.

scipy/skimage are imported lazily inside the functions that need them so that the
numpy-only paths (dice, and the empty-GT NaN early-returns) import and run without
those heavy deps installed."""
import numpy as np

def dice(pred, gt, eps=1e-6, exclude_empty=True):
    """Dice overlap. Empty GT is ambiguous (0/0 -> a fake ~1.0 via eps that biases
    the mean upward), so with exclude_empty=True (default) we return NaN when the GT
    mask has no positive pixels; callers should aggregate with np.nanmean to drop
    those frames. Pass exclude_empty=False for the legacy eps-smoothed value.
    Non-empty-GT behaviour is unchanged (incl. empty-pred-vs-non-empty-GT -> ~0.0)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if exclude_empty and gt.sum() == 0:
        return float("nan")
    return (2 * (pred & gt).sum() + eps) / (pred.sum() + gt.sum() + eps)

def _tprec(s, v):  # skeleton fraction inside volume
    s, v = s.astype(bool), v.astype(bool)
    return (s & v).sum() / (s.sum() + 1e-6) if s.sum() else 1.0

def cldice(pred, gt, exclude_empty=True):
    """Centreline-Dice (topology-aware). Empty GT is ambiguous: empty skeletons make
    _tprec return 1.0, so empty-pred + empty-GT scores a fake ~1.0. With
    exclude_empty=True (default) we return NaN when the GT mask has no positive
    pixels so callers can np.nanmean it out. Pass exclude_empty=False for the legacy
    value. Non-empty-GT behaviour is unchanged (incl. empty-pred vs non-empty-GT -> ~0.0)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if exclude_empty and gt.sum() == 0:
        return float("nan")
    from skimage.morphology import skeletonize
    sp, sg = skeletonize(pred), skeletonize(gt)
    tprec, tsens = _tprec(sp, gt), _tprec(sg, pred)
    return 2 * tprec * tsens / (tprec + tsens + 1e-6)

def hd95(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    from scipy.ndimage import distance_transform_edt as edt
    dg, dp = edt(~gt), edt(~pred)
    d = np.concatenate([dg[pred], dp[gt]])
    return float(np.percentile(d, 95))

if __name__ == "__main__":
    g = np.zeros((64, 64), int); g[20:40, 30:33] = 1
    p = g.copy(); p[35:40, 30:33] = 0
    print("dice", round(dice(p, g), 3), "clDice", round(cldice(p, g), 3), "hd95", round(hd95(p, g), 2))
