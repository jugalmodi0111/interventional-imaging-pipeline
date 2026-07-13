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

def reliability_curve(probs, labels, n_bins=15):
    """Per-bin (confidence, accuracy, count) for a reliability diagram. Pure — no plotting.
    A perfectly calibrated model sits on conf==acc (the diagonal)."""
    probs, labels = np.asarray(probs).ravel(), np.asarray(labels).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (probs >= lo) & (probs <= hi) if i == 0 else (probs > lo) & (probs <= hi)
        c = int(m.sum())
        rows.append({"bin_lo": round(float(lo), 3), "bin_hi": round(float(hi), 3),
                     "conf": float(probs[m].mean()) if c else None,
                     "acc": float(labels[m].mean()) if c else None, "count": c})
    return rows


def save_reliability_diagram(probs, labels, path, n_bins=15):
    """Write a reliability diagram PNG (guarded: matplotlib optional). Returns path or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping reliability diagram")
        return None
    rows = [r for r in reliability_curve(probs, labels, n_bins) if r["count"]]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot([r["conf"] for r in rows], [r["acc"] for r in rows], "o-", label="model")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability (ECE={ece(probs, labels, n_bins):.3f})"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def _bce(probs, labels, eps=1e-7):
    p = np.clip(np.asarray(probs).ravel(), eps, 1 - eps)
    y = np.asarray(labels).ravel()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def temperature_scale(logits, labels, lo=0.05, hi=10.0, iters=60):
    """Post-hoc temperature T>0 that minimizes BCE of sigmoid(logits/T) vs labels (Platt-style,
    single scalar). Pure 1-D golden-section search (no torch). Returns the optimal T."""
    logits, labels = np.asarray(logits, float).ravel(), np.asarray(labels).ravel()
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    nll = lambda T: _bce(sig(logits / T), labels)
    gr = (np.sqrt(5) - 1) / 2                      # golden ratio
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = nll(d)
    return float((a + b) / 2)


def apply_temperature(logits, T):
    """Calibrated probabilities sigmoid(logits/T)."""
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, float) / T))


def auroc(scores, labels):
    """AUROC that `scores` rank positives (label==1) above negatives. Pure, tie-averaged
    (Mann–Whitney U). Returns nan if only one class present."""
    scores, labels = np.asarray(scores, float).ravel(), np.asarray(labels).ravel()
    pos, neg = (labels == 1), (labels == 0)
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties so the U statistic is exact
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    auc = (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return float(auc)


def ood_auroc(id_scores, ood_scores):
    """Separability of an OOD score between in-distribution and OOD inputs, as AUROC.
    Pass a score that is HIGHER for OOD (e.g. predictive entropy, or 1 - max(p,1-p)).
    ~0.5 = the defer path can't tell OOD apart; →1.0 = clean separation."""
    id_scores, ood_scores = np.asarray(id_scores, float).ravel(), np.asarray(ood_scores, float).ravel()
    scores = np.concatenate([ood_scores, id_scores])
    labels = np.concatenate([np.ones(len(ood_scores)), np.zeros(len(id_scores))])
    return auroc(scores, labels)


def uncertainty_score(probs):
    """Two-sided uncertainty in [0,1] for the defer/OOD path: 1 - |2p - 1|. 0.5 -> 1.0 (max unsure),
    0 or 1 -> 0.0 (certain). A monotone rescale of coverage_risk's confidence max(p,1-p), so it
    ranks the same for OOD-AUROC while spanning the full [0,1]."""
    p = np.asarray(probs, float).ravel()
    return 1.0 - np.abs(2.0 * p - 1.0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000); y = (rng.uniform(size=2000) < p).astype(int)
    print("ECE", round(ece(p, y), 4), "Brier", round(brier(p, y), 4))
    print("coverage-risk[0], [-1]:", coverage_risk(p, y)[0], coverage_risk(p, y)[-1])
    # temp-scaling demo on deliberately over-confident logits
    z = rng.normal(size=3000) * 3; yb = (rng.uniform(size=3000) < 1 / (1 + np.exp(-z))).astype(int)
    zt = z * 2.5                                       # inflate -> miscalibrated
    T = temperature_scale(zt, yb)
    print("T*", round(T, 3),
          "ECE pre", round(ece(1 / (1 + np.exp(-zt)), yb), 4),
          "post", round(ece(apply_temperature(zt, T), yb), 4))
    # OOD demo: OOD inputs are less confident -> higher uncertainty
    print("OOD-AUROC", round(ood_auroc(uncertainty_score(p),
                                       uncertainty_score(rng.uniform(0.4, 0.6, size=500))), 3))
