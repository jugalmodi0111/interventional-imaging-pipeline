"""Segmentation metrics: Dice, clDice (connectivity), HD95. Functional."""
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from skimage.morphology import skeletonize

def dice(pred, gt, eps=1e-6):
    pred, gt = pred.astype(bool), gt.astype(bool)
    return (2 * (pred & gt).sum() + eps) / (pred.sum() + gt.sum() + eps)

def _tprec(s, v):  # skeleton fraction inside volume
    s, v = s.astype(bool), v.astype(bool)
    return (s & v).sum() / (s.sum() + 1e-6) if s.sum() else 1.0

def cldice(pred, gt):
    sp, sg = skeletonize(pred.astype(bool)), skeletonize(gt.astype(bool))
    tprec, tsens = _tprec(sp, gt), _tprec(sg, pred)
    return 2 * tprec * tsens / (tprec + tsens + 1e-6)

def hd95(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    dg, dp = edt(~gt), edt(~pred)
    d = np.concatenate([dg[pred], dp[gt]])
    return float(np.percentile(d, 95))

if __name__ == "__main__":
    g = np.zeros((64, 64), int); g[20:40, 30:33] = 1
    p = g.copy(); p[35:40, 30:33] = 0
    print("dice", round(dice(p, g), 3), "clDice", round(cldice(p, g), 3), "hd95", round(hd95(p, g), 2))
