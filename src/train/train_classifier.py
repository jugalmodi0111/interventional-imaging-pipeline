"""Train Model One: frozen-backbone binary classifier over the de-identified AVF frame store.

Consumes exactly what src/ingest/ produces: frames/<stem_prefix>/f%05d.png and a labels JSONL of
{"key": <stem_prefix>, "label": 0|1} rows (src.ingest.labels join output). Split is by PATIENT
(io_utils.group_key) -- the F1 0.885->0.214 leakage incident is why this module refuses to split
any other way, and why group overlap is an assertion, not a log line (B5/B6).
"""
import hashlib
import json
from pathlib import Path

from src.data_prep.io_utils import group_key
from src.ingest.extract import frame_stem
from src.ingest.manifest import read_jsonl


def _patient_group(stem):
    """Patient group for a labels-JSONL ``key``, which is a SERIES stem (``avf_<site>_<pid>_s01``).

    ``group_key`` is written against FRAME stems -- ``_AVF_RE`` is anchored ``..._s\\d+_\\d+$`` and
    will not match a bare series prefix, which would hand back the series and split one patient's
    two studies across train and val (audit P0.2 / A1a). So reconstruct the frame stem the ingest
    writer itself would emit (``extract.frame_stem``) and group THAT. If no rule in ``group_key``
    fires, the reconstruction comes back unchanged: fall back to the series key, so an unknown
    stem grammar degrades to one group per series and never to one group per frame -- the shape
    that passes a group-overlap audit trivially while leaking (PROJECT_TRACKER 2026-08-16, CADICA).
    """
    probe = frame_stem(stem, 0)
    collapsed = group_key(probe)
    return stem if collapsed == probe else collapsed


def load_examples(frames_root, labels_path):
    frames_root = Path(frames_root)
    examples, missing = [], 0
    for row in read_jsonl(labels_path):
        stem, label = row.get("key"), row.get("label")
        if stem is None or label not in (0, 1):
            continue
        d = frames_root / stem
        pngs = sorted(d.glob("f*.png")) if d.is_dir() else []
        if not pngs:
            missing += 1
            continue
        group = _patient_group(stem)
        for p in pngs:
            examples.append({"path": str(p), "stem": stem, "group": group, "label": int(label)})
    if missing:
        print(f"[train_cls] skipped {missing} labeled stems with no frames on disk")
    return examples


def grouped_split(examples, val_frac=0.2, seed=0):
    """Deterministic per-GROUP assignment: hash(group|seed) -> [0,1) < val_frac => val."""
    train, val = [], []
    for e in examples:
        h = hashlib.sha256(f"{e['group']}|{seed}".encode()).hexdigest()
        (val if int(h[:8], 16) / 0xFFFFFFFF < val_frac else train).append(e)
    overlap = {e["group"] for e in train} & {e["group"] for e in val}
    assert not overlap, f"patient groups in BOTH splits (leakage): {sorted(overlap)[:5]}"
    return train, val


#: Repo-root-anchored, like src/ingest/extract.DEFAULT_CLEARANCE_PATH: the defer band is a
#: clinical parameter (B3) and must not change meaning with the process cwd.
DEFAULT_CLS_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "avf_fistulography.yaml"


def _load_gray(path, imgsz):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    return img.astype("float32") / 255.0


def _tensorize(examples, imgsz):
    import numpy as np
    import torch
    x = np.stack([_load_gray(e["path"], imgsz) for e in examples])[:, None]   # [N,1,H,W]
    y = np.array([e["label"] for e in examples], dtype="float32")
    return torch.from_numpy(x), torch.from_numpy(y)


def _defer_band(cfg_path=None):
    """B3 defer band from configs/avf_fistulography.yaml. Unreadable config is a warning, not a
    crash -- but the fallback is the SAME band the config ships, never a wider (more permissive)
    one: a missing file must not quietly turn abstention off."""
    cfg_path = DEFAULT_CLS_CONFIG if cfg_path is None else cfg_path
    try:
        import yaml
        with open(cfg_path) as f:
            return list(yaml.safe_load(f)["defer"]["band"])
    except Exception:
        print(f"[train_cls] warning: could not read defer.band from {cfg_path}; using [0.3, 0.6]")
        return [0.3, 0.6]


def train(frames_root, labels_path, out_dir, *, backbone="test-tiny", imgsz=32, epochs=2,
          lr=1e-2, val_frac=0.2, seed=0, target_sensitivity=None):
    """Fit the linear head, calibrate on val, pick an operating point, write head.pt+metrics.json.

    target_sensitivity None means the clinical floor is NOT signed off
    (configs/avf_fistulography.yaml target.sensitivity: null): the threshold falls back to 0.5 and
    metrics.json says so via threshold_policy='threshold-unsigned'. Never invent a floor.
    """
    import torch
    from src.eval.calibration import apply_temperature, auroc, ece, temperature_scale
    from src.eval.cls_metrics import (bootstrap_ci, sensitivity, specificity,
                                      threshold_at_sensitivity)
    from src.models.frozen_backbone import FrozenBackboneClassifier

    torch.manual_seed(seed)
    examples = load_examples(frames_root, labels_path)
    train_ex, val_ex = grouped_split(examples, val_frac=val_frac, seed=seed)
    assert train_ex and val_ex, "both splits must be non-empty (add patients or adjust val_frac)"
    xt, yt = _tensorize(train_ex, imgsz)
    xv, yv = _tensorize(val_ex, imgsz)

    model = FrozenBackboneClassifier(backbone, imgsz=imgsz)
    opt = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(xv).numpy()
    labels_np = yv.numpy().astype(int)
    T = float(temperature_scale(val_logits, labels_np))
    probs = apply_temperature(val_logits, T)

    thr = threshold_at_sensitivity(probs, labels_np, target_sensitivity)
    policy = "threshold-unsigned" if target_sensitivity is None else "at-target-sensitivity"
    band = _defer_band()
    metrics = {
        "n_train_frames": len(train_ex), "n_val_frames": len(val_ex),
        "n_train_groups": len({e["group"] for e in train_ex}),
        "n_val_groups": len({e["group"] for e in val_ex}),
        "backbone": backbone, "temperature": T, "threshold": float(thr),
        "threshold_policy": policy,
        "sensitivity": sensitivity(probs, labels_np, thr),
        "specificity": specificity(probs, labels_np, thr),
        "auroc": float(auroc(probs, labels_np)), "ece": float(ece(probs, labels_np)),
        "sensitivity_ci": bootstrap_ci(sensitivity, probs, labels_np, n_boot=200,
                                       seed=seed, thr=thr),
        "specificity_ci": bootstrap_ci(specificity, probs, labels_np, n_boot=200,
                                       seed=seed, thr=thr),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": backbone, "imgsz": imgsz, "head_state": model.head.state_dict(),
                "temperature": T, "threshold": float(thr), "defer_band": band}, out / "head.pt")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[train_cls] " + json.dumps({k: metrics[k] for k in
                                       ("auroc", "sensitivity", "specificity", "threshold_policy")}))
    return metrics


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument("--frames", required=True, help="frame store root: <root>/<stem_prefix>/f%%05d.png")
    p.add_argument("--labels", required=True, help="labels JSONL of {'key','label'} rows")
    p.add_argument("--out", required=True, help="output dir for head.pt + metrics.json")
    p.add_argument("--backbone", default="test-tiny",
                   help="timm model name; 'test-tiny' is the OFFLINE TEST backbone -- never a real run")
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-sensitivity", type=float, default=None,
                   help="clinical sensitivity floor; omitted => threshold-unsigned (B7 not signed)")
    a = p.parse_args(argv)
    train(a.frames, a.labels, a.out, backbone=a.backbone, imgsz=a.imgsz, epochs=a.epochs,
          lr=a.lr, val_frac=a.val_frac, seed=a.seed, target_sensitivity=a.target_sensitivity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
