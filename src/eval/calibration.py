"""Calibration & abstention metrics. 'Wrong but confident' is the dangerous mode."""
import numpy as np

def ece(probs, labels, n_bins=15):
    """Expected Calibration Error for binary/foreground confidence."""
    probs, labels = np.asarray(probs).ravel(), np.asarray(labels).ravel()
    n = len(probs)
    if n == 0:
        return float("nan")  # no samples -> undefined, not perfectly calibrated (0.0)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (probs >= lo) & (probs <= hi) if i == 0 else (probs > lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        conf, acc = probs[m].mean(), labels[m].mean()
        e += (m.sum() / n) * abs(conf - acc)
    return float(e)

def brier(probs, labels):
    probs, labels = np.asarray(probs).ravel(), np.asarray(labels).ravel()
    return float(np.mean((probs - labels) ** 2))

def coverage_risk(probs, labels, thresholds=None):
    """Risk (error) vs coverage as the abstention threshold sweeps. Drives the defer-to-human path."""
    probs, labels = np.asarray(probs).ravel(), np.asarray(labels).ravel()
    conf = np.maximum(probs, 1 - probs)
    preds = (probs >= 0.5).astype(int)
    thresholds = thresholds if thresholds is not None else np.linspace(0.5, 0.99, 10)
    out = []
    for t in thresholds:
        keep = conf >= t
        cov = keep.mean()
        if keep.sum() == 0:
            # Nothing kept at this threshold: risk is undefined, NOT zero. Flag it
            # (risk=None) so the risk-coverage curve isn't anchored to a fake 0.0 point.
            out.append({"threshold": round(float(t), 3), "coverage": round(float(cov), 3),
                        "risk": None})
            continue
        risk = (preds[keep] != labels[keep]).mean()
        out.append({"threshold": round(float(t), 3), "coverage": round(float(cov), 3),
                    "risk": round(float(risk), 3)})
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000); y = (rng.uniform(size=2000) < p).astype(int)
    print("ECE", round(ece(p, y), 4), "Brier", round(brier(p, y), 4))
    print("coverage-risk[0], [-1]:", coverage_risk(p, y)[0], coverage_risk(p, y)[-1])
# TODO: reliability diagram plot; temperature scaling; OOD-AUROC vs held-out vendor.
