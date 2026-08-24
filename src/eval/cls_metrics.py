"""Binary classification metrics for Model One triage (B3 vocabulary: sensitivity/specificity).

Pure numpy, no torch: importable by eval scripts, the trainer, and tests without the heavy stack.
AUROC and ECE deliberately live in src/eval/calibration.py already -- import them from there, this
module only adds what the repo lacked (confusion vocabulary, operating-point selection, CIs).

Operating-point rule (B3 'low confidence defaults to uncertain, never false normal'): choose the
threshold FROM a sensitivity target -- the highest cut that still catches the required fraction of
positives -- then report the specificity that follows. A None target means the clinical floor is
not signed off (configs/avf_fistulography.yaml target: null); callers fall back to 0.5 and must
mark the result unsigned rather than invent a floor.
"""
import numpy as np


def confusion_counts(probs, labels, thr):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = probs >= thr
    pos = labels == 1
    return {"tp": int(np.sum(pred & pos)), "fp": int(np.sum(pred & ~pos)),
            "tn": int(np.sum(~pred & ~pos)), "fn": int(np.sum(~pred & pos))}


def sensitivity(probs, labels, thr):
    c = confusion_counts(probs, labels, thr)
    denom = c["tp"] + c["fn"]
    return c["tp"] / denom if denom else 0.0


def specificity(probs, labels, thr):
    c = confusion_counts(probs, labels, thr)
    denom = c["tn"] + c["fp"]
    return c["tn"] / denom if denom else 0.0


def threshold_at_sensitivity(probs, labels, target):
    """Highest threshold with sensitivity >= target, scanning candidate cuts at the observed
    positive probabilities (any threshold between two observed values behaves identically).
    None target -> 0.5 (floor unsigned); unreachable target -> 0.0 (call everything positive)."""
    if target is None:
        return 0.5
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    for thr in sorted(np.unique(probs[labels == 1]), reverse=True):
        if sensitivity(probs, labels, thr) >= target:
            return float(thr)
    return 0.0


def bootstrap_ci(metric_fn, probs, labels, n_boot=1000, seed=0, alpha=0.05, thr=0.5):
    """Percentile bootstrap CI for metric_fn(probs, labels, thr). Deterministic under a seed so
    metrics.json is reproducible run-to-run."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(probs)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(probs[idx], labels[idx], thr))
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))
