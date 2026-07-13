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

def vessel_distance_heatmap(gt, r_th=8):
    """GT binary mask -> (D_gt heatmap in [0,1], GT skeleton as float). D_gt is 1 on the skeleton and
    decays LINEARLY to 0 at r_th px away (CLGeoDice paper Eq.1). Empty GT -> zeros. skimage/scipy lazy."""
    from skimage.morphology import skeletonize
    from scipy.ndimage import distance_transform_edt as edt
    gt = np.asarray(gt).astype(bool)
    if gt.sum() == 0:
        z = np.zeros(gt.shape, np.float32)
        return z, z
    sk = skeletonize(gt)
    dgt = np.clip(1.0 - edt(~sk) / float(r_th), 0.0, 1.0).astype(np.float32)
    return dgt, sk.astype(np.float32)

def clgeodice(pred, gt, r_th=8, eps=1e-6, exclude_empty=True):
    """CLGeoDice score (Chen et al. 2026) — geometry+topology-aware, a clDice successor that rewards
    the predicted centerline for lying NEAR the GT centerline via a continuous distance heatmap
    (clDice only rewards exact skeleton overlap and is blind to in-lumen drift). Higher = better.
    Empty GT -> NaN (aggregate with np.nanmean), like dice/cldice. skimage/scipy lazy."""
    from skimage.morphology import skeletonize
    pred_b, gt_b = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    if exclude_empty and gt_b.sum() == 0:
        return float("nan")
    dgt, s_true = vessel_distance_heatmap(gt_b, r_th)
    s_pred = skeletonize(pred_b).astype(np.float32)
    tgp = (s_pred * dgt).sum() / (s_pred.sum() + eps)          # geometric topological precision
    tsn = (s_true * pred_b.astype(np.float32)).sum() / (s_true.sum() + eps)   # topological sensitivity
    return float(2 * tgp * tsn / (tgp + tsn + eps))

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
