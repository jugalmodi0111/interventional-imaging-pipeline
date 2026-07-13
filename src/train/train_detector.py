"""Train YOLO11n stenosis detector (+ optional pseudo-label SSL). Importable; run on GPU.

Library entrypoint `train(cfg)` — call it from the Colab/Kaggle notebook. Heavy (GPU); do not
run on a laptop CPU. Saves best weights to runs/stenosis/ and reports F1/mAP.

SSL cold-start: set `ssl.seed: gdino` to seed round-0 pseudo-labels with an open-vocabulary
Grounding DINO pass on the unlabeled frames (a better cold start than a YOLO trained on scarce
labels), then the usual pseudo-label self-training refines. GD deps (torch/transformers) load
only inside `_gdino_seed_round`; this module imports fine without them.
"""
import argparse, glob, os, yaml

from src.data_prep.autolabel_gdino import DEFAULT_PROMPTS, dino_boxes_to_yolo_lines


def _detector(cfg):
    return cfg.get("model") or cfg.get("detector")           # stenosis uses 'model', catheter 'detector'


def _data_yaml(cfg):
    task = cfg.get("task", "")
    sub = "catheter" if "catheter" in task else "stenosis"
    return f"data/processed/{sub}/data.yaml"


def ssl_seed(cfg):
    """The SSL cold-start seed, e.g. 'gdino', or None (from cfg['ssl']['seed'])."""
    return cfg.get("ssl", {}).get("seed")


def ssl_enabled(cfg):
    """(do_pseudo, do_gdino): whether each SSL mode is allowed to run.

    Both pseudo-label and the gdino cold-start copy every ssl.unlabeled_dir frame into images/train,
    so they're only safe with a disjoint unlabeled set explicitly attached. With no usable
    cfg['ssl']['unlabeled_dir'] (None/empty) both modes are forced off so a CLI/notebook run can't
    silently re-leak val patients into train. Pure (config-only, no filesystem) — train() adds the
    on-disk existence check.
    """
    ssl = cfg.get("ssl") or {}
    if not ssl.get("unlabeled_dir"):                         # None/empty -> no disjoint set -> both off
        return (False, False)
    return (bool(ssl.get("pseudo_label")), ssl.get("seed") == "gdino")


def seed_prompt_and_classes(cfg):
    """(open-vocab text prompt, {label_string: yolo_class_id}) for the detector's task.

    Catheter pulls its class names from the config; stenosis is single-class. Prompt words must
    match the class names so Grounding DINO labels map cleanly onto YOLO ids.
    """
    task = cfg.get("task", "")
    if "catheter" in task:
        names = cfg.get("datasets", {}).get("cathaction", {}).get("classes", ["catheter", "guidewire"])
        return DEFAULT_PROMPTS["catheter"], {n: i for i, n in enumerate(names)}
    return DEFAULT_PROMPTS["stenosis"], {"stenosis": 0}


def boxes_labels_to_yolo_lines(boxes_xyxy, labels, class_map, W, H):
    """Pixel boxes + label strings -> YOLO lines, mapping labels via class_map (unknown skipped)."""
    lines = []
    for box, lab in zip(boxes_xyxy, labels):
        cid = class_map.get(lab)
        if cid is not None:
            lines += dino_boxes_to_yolo_lines([box], W, H, cid)
    return lines


def run_tag(cfg):
    """Stable run name encoding data+model+imgsz+epochs, e.g. 'arcade_yolo11n_640_e150'.
    Use it as the ultralytics run folder so different configs don't clobber each other's outputs."""
    m = _detector(cfg) or {}
    names = [k.replace("_stenosis", "").replace("_detection", "") for k in cfg.get("datasets", {})]
    data = "+".join(sorted(names)) or "data"
    return f"{data}_{m.get('name', 'yolo')}_{m.get('imgsz', 640)}_e{cfg.get('train', {}).get('epochs', 100)}"


def train_kwargs(cfg):
    """Ultralytics train() kwargs incl. speed knobs. Fast, quality-neutral defaults:
    cache (RAM/disk dataset cache), amp (mixed precision), and patience (early-stop once the
    val metric plateaus — keeps the same best.pt, skips wasted epochs)."""
    m, tr = _detector(cfg) or {}, cfg.get("train", {})
    return {"imgsz": m.get("imgsz", 640),
            "epochs": tr.get("epochs", 100), "batch": tr.get("batch", 16),
            "lr0": tr.get("lr", 1e-3),
            "cache": tr.get("cache", True), "workers": tr.get("workers", 8),
            "patience": tr.get("patience", 30), "amp": tr.get("amp", True)}


def best_f1_from_pr(precision_arr, recall_arr):
    """Max F1 = 2PR/(P+R) over paired precision/recall points (div-by-zero -> 0.0).

    Accepts the ultralytics per-class p/r arrays, or [mean_p]/[mean_r] passed as single-element lists
    for the scalar fallback. Empty -> 0.0. Pure/torch-free so the F1 floor can be unit-tested with no
    GPU val run.
    """
    best = 0.0
    for p, r in zip(precision_arr, recall_arr):
        p, r = float(p), float(r)
        denom = p + r
        f1 = (2.0 * p * r / denom) if denom > 0 else 0.0     # guard P=R=0
        if f1 > best:
            best = f1
    return best


def qualifies_det(scores, cfg):
    """True iff scores['f1'] clears the F1 floor cfg['target']['f1'] (default 0.57, inclusive) AND,
    when cfg['target']['recall'] is set, scores['recall'] clears that recall floor too (inclusive).

    Recall is the clinically costly axis for stenosis — a missed lesion is the dangerous error — so a
    config can demand a minimum recall on TOP of F1. An unset target.recall leaves the F1-only
    behaviour untouched. Mirrors train_seg.qualifies (Dice floor + optional second gate). PURE
    (config-only), torch-free so the floor is unit-tested with no GPU val run."""
    tgt = cfg.get("target") or {}
    if float(scores.get("f1") or 0.0) < float(tgt.get("f1", 0.57)):
        return False
    rfloor = tgt.get("recall")
    if rfloor is not None and float(scores.get("recall") or 0.0) < float(rfloor):
        return False
    return True


def det_scores(precision, recall, map50, map=None):
    """Recall-first score dict from an ultralytics val result's box means (val.box.mp/mr/map50/map).

    Returns {'precision','recall','f1','map50'[,'map']} with F1 = 2PR/(P+R), guarded so P+R==0 -> 0.0
    (a model that fires nowhere scores F1 0, not a div error). Recall is surfaced deliberately because
    for stenosis a missed lesion (low recall) is the clinically costly error. PURE/torch-free so the
    F1 math + the gate are unit-tested with no GPU val run."""
    p, r = float(precision or 0.0), float(recall or 0.0)
    denom = p + r
    f1 = (2.0 * p * r / denom) if denom > 0 else 0.0         # guard P=R=0
    out = {"precision": p, "recall": r, "f1": f1, "map50": float(map50 or 0.0)}
    if map is not None:
        out["map"] = float(map)
    return out


def val_kwargs(cfg):
    """Kwargs for the final ultralytics .val() call. Exposes a LOW default conf so recall isn't
    throttled at eval: ultralytics' default val conf (0.001) is fine, but if a config over-raises it
    the low-confidence true stenoses get dropped and recall (the costly axis) tanks. cfg['val']['conf']
    (default 0.001) and optional cfg['val']['iou'] flow through. PURE/torch-free."""
    v = cfg.get("val") or {}
    kw = {"conf": v.get("conf", 0.001)}
    if v.get("iou") is not None:
        kw["iou"] = v.get("iou")
    return kw


def pretrained_ckpt(cfg):
    """Path to an SSL/angiography-pretrained backbone checkpoint (cfg['model']['pretrained_weights']),
    or None. The drop-in point for a backbone pretrained GPU-side (e.g. self-supervised on unlabeled
    XCA) — set the key and train() loads it (strict=False) before training, no flow changes needed.
    PURE/torch-free."""
    return (cfg.get("model") or {}).get("pretrained_weights")


def train(cfg, project=None, data_yaml=None, device=0):
    # device=0 forces GPU (fails loud if none); pass 'cpu' to override.
    from ultralytics import YOLO
    m = _detector(cfg)
    data_yaml = data_yaml or _data_yaml(cfg)
    project = project or f"runs/{'catheter' if 'catheter' in cfg.get('task','') else 'stenosis'}"
    tk = train_kwargs(cfg)                                    # imgsz/epochs/batch/lr0 + cache/workers/patience/amp
    model = YOLO(m["name"] + ".pt")
    _load_pretrained_backbone(model, cfg)                    # optional SSL/angiography-pretrained init
    model.train(data=data_yaml, project=project, name="base", exist_ok=True, device=device, **tk)
    best = os.path.join(project, "base", "weights", "best.pt")

    ssl = cfg.get("ssl") or {}
    udir = ssl.get("unlabeled_dir")                          # gate SSL on a disjoint, on-disk unlabeled set
    has_unlabeled = bool(udir and os.path.isdir(udir)
                         and glob.glob(os.path.join(udir, "**", "*.png"), recursive=True))
    do_pseudo, do_gdino = ssl_enabled(cfg)                   # config intent (None/empty dir -> both off)

    if ssl_seed(cfg) == "gdino":                              # open-vocab cold start before self-training
        if do_gdino and has_unlabeled:
            best = _gdino_seed_round(cfg, project, data_yaml, device=device)
        else:
            print("SSL gdino skipped: no disjoint ssl.unlabeled_dir")

    if ssl.get("pseudo_label"):
        if do_pseudo and has_unlabeled:
            best = _pseudo_label_round(best, cfg, project, data_yaml, device=device)
        else:
            print("SSL pseudo-label skipped: no disjoint ssl.unlabeled_dir")

    val = YOLO(best).val(data=data_yaml, device=device, **val_kwargs(cfg))   # low val conf: don't throttle recall
    box = val.box
    p_arr, r_arr = getattr(box, "p", None), getattr(box, "r", None)
    if p_arr is not None and r_arr is not None and len(p_arr) and len(r_arr):
        f1 = best_f1_from_pr(p_arr, r_arr)                   # best F1 over the per-class PR points
    else:                                                    # fall back to mean-P/mean-R scalars
        f1 = best_f1_from_pr([float(getattr(box, "mp", 0.0) or 0.0)],
                             [float(getattr(box, "mr", 0.0) or 0.0)])
    # recall-first report from the val box means; gate on the F1-maximizing operating point (the
    # best-over-PR-curve F1 when it beats the mean-P/mean-R F1 — that's the point the floor is on).
    scores = det_scores(getattr(box, "mp", 0.0), getattr(box, "mr", 0.0),
                        getattr(box, "map50", 0.0), map=getattr(box, "map", None))
    scores["f1"] = max(scores["f1"], f1)
    floor = (cfg.get("target") or {}).get("f1", 0.57)
    rfloor = (cfg.get("target") or {}).get("recall")
    ok = qualifies_det(scores, cfg)
    print("best:", best, "| scores:", {k: round(v, 4) for k, v in scores.items()})
    print("qualifies_det:", ok)
    print(f"[{'PASS' if ok else 'FAIL'}] recall {round(scores['recall'], 4)}"
          + (f" {'>=' if scores['recall'] >= rfloor else '<'} {rfloor}" if rfloor is not None else " (no floor)")
          + f" | F1 {round(scores['f1'], 4)} {'>=' if scores['f1'] >= floor else '<'} floor {floor}")
    return best


def _load_pretrained_backbone(model, cfg):
    """Load cfg['model']['pretrained_weights'] into the YOLO model (strict=False) BEFORE training —
    the drop-in point for an SSL/angiography-pretrained backbone produced GPU-side. No-op when the key
    is unset. Lazy torch (module stays importable without it); prints matched/total key counts; a
    missing file or shape mismatch is reported, not fatal (training just continues from default init)."""
    ckpt = pretrained_ckpt(cfg)
    if not ckpt:
        return
    if not os.path.exists(ckpt):
        print(f"pretrained_weights: {ckpt} not found; using default init")
        return
    import torch
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict):                                 # unwrap common checkpoint envelopes
        sd = sd.get("state_dict", sd.get("model", sd))
    if hasattr(sd, "state_dict"):                            # a pickled nn.Module (e.g. a YOLO ckpt)
        sd = sd.state_dict()
    tgt = model.model                                        # underlying torch nn.Module
    tsd = tgt.state_dict()
    matched = {k: v for k, v in sd.items() if k in tsd and tsd[k].shape == v.shape}
    tsd.update(matched)
    tgt.load_state_dict(tsd, strict=False)
    print(f"pretrained_weights: loaded {len(matched)}/{len(tsd)} matching keys from {ckpt} (strict=False)")


def _gdino_seed_round(cfg, project, data_yaml, unlabeled_dir=None, device=0):
    """Round-0 cold start: Grounding DINO open-vocab labels the unlabeled frames -> YOLO labels in
    the train split, then retrain from scratch. Heavy: GD (torch/transformers) loads lazily here."""
    import cv2
    from ultralytics import YOLO
    from src.data_prep import io_utils as io
    from src.data_prep.autolabel_gdino import detect, filter_detections
    ssl = cfg.get("ssl", {})
    conf = ssl.get("conf", 0.4)
    size = (_detector(cfg) or {}).get("imgsz", 640)
    prompt, class_map = seed_prompt_and_classes(cfg)
    unlabeled_dir = unlabeled_dir or ssl.get("unlabeled_dir", "data/raw/xcad")
    imgs = glob.glob(os.path.join(unlabeled_dir, "**", "*.png"), recursive=True)
    if not imgs:
        print("GD-seed: no unlabeled frames found; skipping")
        return os.path.join(project, "base", "weights", "best.pt")
    proc = os.path.dirname(data_yaml)
    out_i, out_l = os.path.join(proc, "images/train"), os.path.join(proc, "labels/train")
    os.makedirs(out_i, exist_ok=True); os.makedirs(out_l, exist_ok=True)
    kept = 0
    for ip in imgs:
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        H, W = g.shape
        boxes, scores, labels = detect(g, prompt, box_thresh=conf, device=device)
        boxes, scores, labels = filter_detections(boxes, scores, labels, conf)
        lines = boxes_labels_to_yolo_lines(boxes, labels, class_map, W, H)
        if not lines:
            continue
        stem = "gd_" + os.path.splitext(os.path.basename(ip))[0]
        cv2.imwrite(os.path.join(out_i, stem + ".png"),           # CLAHE+resize to match base/inference
                    cv2.resize(io.clahe_unsharp(g), (size, size)))
        open(os.path.join(out_l, stem + ".txt"), "w").write("\n".join(lines))
        kept += 1
    print(f"GD-seed: added {kept} Grounding-DINO-labeled frames; retraining")
    model = YOLO(_detector(cfg)["name"] + ".pt")
    model.train(data=data_yaml, project=project, name="gdino", exist_ok=True, device=device,
                **train_kwargs(cfg))
    return os.path.join(project, "gdino", "weights", "best.pt")


def _pseudo_label_round(weights, cfg, project, data_yaml, unlabeled_dir=None, device=0):
    """Predict on unlabeled frames >= conf, write YOLO pseudo-labels into the train split, retrain.
    Frames are CLAHE+resized before predict AND on disk so the pseudo-labels match the base-train and
    inference preprocessing; the predicted class id is kept (not forced to 0) for multi-class tasks."""
    import cv2
    from ultralytics import YOLO
    from src.data_prep import io_utils as io
    conf = cfg["ssl"].get("conf", 0.4)
    size = (_detector(cfg) or {}).get("imgsz", 640)
    unlabeled_dir = unlabeled_dir or cfg.get("ssl", {}).get("unlabeled_dir", "data/raw/xcad")
    imgs = glob.glob(os.path.join(unlabeled_dir, "**", "*.png"), recursive=True)
    if not imgs:
        print("SSL: no unlabeled frames found; skipping")
        return weights
    model = YOLO(weights)
    proc = os.path.dirname(data_yaml)
    out_i = os.path.join(proc, "images/train")
    out_l = os.path.join(proc, "labels/train")
    os.makedirs(out_i, exist_ok=True); os.makedirs(out_l, exist_ok=True)
    kept = 0
    for ip in imgs:
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        frame = cv2.resize(io.clahe_unsharp(g), (size, size))    # match base/inference preprocessing
        b = model.predict(frame, conf=conf, verbose=False, device=device)[0].boxes
        if b is None or len(b) == 0:
            continue
        stem = "pl_" + os.path.splitext(os.path.basename(ip))[0]
        cv2.imwrite(os.path.join(out_i, stem + ".png"), frame)
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                 for c, (x, y, w, h) in zip(b.cls.tolist(), b.xywhn.tolist())]
        open(os.path.join(out_l, stem + ".txt"), "w").write("\n".join(lines))
        kept += 1
    print(f"SSL: added {kept} pseudo-labeled frames; retraining")
    model = YOLO(_detector(cfg)["name"] + ".pt")
    model.train(data=data_yaml, project=project, name="ssl", exist_ok=True, device=device,
                **train_kwargs(cfg))
    return os.path.join(project, "ssl", "weights", "best.pt")


def main(cfg):
    return train(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    main(yaml.safe_load(open(ap.parse_args().config)))
