"""CLGeoDice — geometry+topology-aware loss for thin-vessel (coronary) segmentation.

Chen et al., HNNDL 2026 (doi 10.1145/3795892.3795903). A clDice / cbDice successor: it swaps
clDice's DISCRETE skeleton-overlap for a CONTINUOUS distance heatmap, so the gradient is smooth
and geometry-aware (it penalises the in-lumen centerline drift clDice is blind to) while keeping
clDice's topological-sensitivity term for thin-vessel recall.

Training-only -> GPU / build side; the shipped CoreML student is unchanged (this only changes the
distillation objective). Paper Eqs 1-4:
  D_gt(x)     = max(0, 1 - dist(x, S_true)/r_th)            # inverted distance heatmap of the GT skeleton
  T_geo_prec  = sum(S_pred * D_gt) / (sum(S_pred) + eps)    # pred soft-skeleton lands on the GT centerline
  T_sens      = sum(S_true * P)   / (sum(S_true) + eps)     # GT skeleton recalled by pred prob (clDice term)
  L           = 1 - 2*T_geo_prec*T_sens / (T_geo_prec + T_sens + eps)

torch is imported lazily; the eval metric + distance heatmap (numpy) live in `src.eval.metrics`
and are unit-testable without torch. Paper's recommended weight in the total loss is 0.5 (with
CE 1.0, Dice 0.5) and r_th=8.
"""


def _soft_skeletonize(img, iters, F):
    """Differentiable soft-skeleton (Shit et al., clDice) via iterative morphological opening.
    img: (B,1,H,W) in [0,1]. Erosion = -maxpool(-x); dilation = maxpool(x); opening = dilate(erode)."""
    def erode(x):   return -F.max_pool2d(-x, 3, 1, 1)
    def dilate(x):  return F.max_pool2d(x, 3, 1, 1)
    def opening(x): return dilate(erode(x))
    img1 = opening(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = erode(img)
        img1 = opening(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)         # accumulate skeleton without double counting
    return skel


def clgeodice_loss(pred_prob, gt_mask, r_th=8, eps=1e-6, skel_iters=10):
    """Differentiable CLGeoDice loss. `pred_prob`:(B,1,H,W) in [0,1] (sigmoid of student logits),
    `gt_mask`:(B,1,H,W) binary. The GT skeleton + distance heatmap are built per-sample in numpy
    and DETACHED (GT carries no gradient), so gradients flow only through the prediction. Returns a
    scalar tensor (mean over batch).

    NOTE: `skeletonize` per batch is the slow path; the paper precomputes D_gt / S_true offline. If
    training time dominates, cache them per stem instead of recomputing each step."""
    import torch
    import torch.nn.functional as F
    from src.eval.metrics import vessel_distance_heatmap
    dev = pred_prob.device
    gt_np = gt_mask.detach().cpu().numpy()
    dgt = torch.zeros_like(pred_prob)
    st = torch.zeros_like(pred_prob)
    for b in range(pred_prob.shape[0]):
        d, s = vessel_distance_heatmap(gt_np[b, 0] > 0.5, r_th)
        dgt[b, 0] = torch.from_numpy(d).to(dev)
        st[b, 0] = torch.from_numpy(s).to(dev)
    s_pred = _soft_skeletonize(pred_prob.clamp(0, 1), skel_iters, F)
    tgp = (s_pred * dgt).flatten(1).sum(1) / (s_pred.flatten(1).sum(1) + eps)   # geometric topo precision
    tsn = (st * pred_prob).flatten(1).sum(1) / (st.flatten(1).sum(1) + eps)     # topological sensitivity
    return (1 - 2 * tgp * tsn / (tgp + tsn + eps)).mean()
