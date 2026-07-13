"""Recall-first detector reporting + gate: det_scores (F1 math), val_kwargs (don't throttle recall),
qualifies_det's recall floor, and the pretrained-backbone hook path (pretrained_ckpt).

Torch-free: every helper is pure, so the F1 math, the low-conf eval knob, the recall gate, and the
SSL-backbone drop-in point are all unit-tested without ultralytics/torch or a GPU val run.
"""
from src.train.train_detector import det_scores, val_kwargs, qualifies_det, pretrained_ckpt


# ---- det_scores: recall-first dict, F1 = 2PR/(P+R) guarded -------------------------------------

def test_det_scores_f1_math_known_value():
    s = det_scores(0.6, 0.6, 0.5)
    assert round(s["f1"], 6) == 0.6                          # P=R=0.6 -> F1 0.6
    assert s["precision"] == 0.6 and s["recall"] == 0.6 and s["map50"] == 0.5


def test_det_scores_f1_asymmetric():
    s = det_scores(0.9, 0.1, 0.4)                            # 2*0.9*0.1/1.0 = 0.18
    assert round(s["f1"], 6) == 0.18
    assert s["recall"] == 0.1                                # recall surfaced verbatim


def test_det_scores_p_plus_r_zero_is_zero_no_div_error():
    s = det_scores(0.0, 0.0, 0.0)                            # denom==0 -> guarded 0.0, no ZeroDivision
    assert s["f1"] == 0.0


def test_det_scores_perfect_pr_is_one():
    assert det_scores(1.0, 1.0, 1.0)["f1"] == 1.0


def test_det_scores_map_optional():
    assert "map" not in det_scores(0.5, 0.5, 0.5)           # omitted by default
    assert det_scores(0.5, 0.5, 0.5, map=0.33)["map"] == 0.33


def test_det_scores_none_inputs_coerce_to_zero():
    s = det_scores(None, None, None)                         # missing box attrs -> 0.0, F1 0.0
    assert s == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "map50": 0.0}


# ---- val_kwargs: low eval conf so recall isn't throttled ---------------------------------------

def test_val_kwargs_default_conf_is_low():
    assert val_kwargs({})["conf"] == 0.001                  # low default: keep low-conf true stenoses
    assert "iou" not in val_kwargs({})                      # iou only flows when set


def test_val_kwargs_reads_cfg_val():
    kw = val_kwargs({"val": {"conf": 0.01, "iou": 0.6}})
    assert kw["conf"] == 0.01 and kw["iou"] == 0.6


def test_val_kwargs_iou_only_when_set():
    assert val_kwargs({"val": {"conf": 0.005}}) == {"conf": 0.005}


# ---- qualifies_det: recall floor on top of the F1 floor ----------------------------------------

def test_qualifies_det_recall_floor_gates_when_set():
    cfg = {"target": {"f1": 0.57, "recall": 0.5}}
    assert qualifies_det({"f1": 0.60, "recall": 0.55}, cfg) is True   # both clear
    assert qualifies_det({"f1": 0.60, "recall": 0.40}, cfg) is False  # F1 ok, recall short
    assert qualifies_det({"f1": 0.50, "recall": 0.90}, cfg) is False  # recall ok, F1 short


def test_qualifies_det_recall_floor_inclusive():
    cfg = {"target": {"f1": 0.57, "recall": 0.5}}
    assert qualifies_det({"f1": 0.57, "recall": 0.5}, cfg) is True    # both exactly at floor


def test_qualifies_det_no_recall_floor_is_f1_only():
    cfg = {"target": {"f1": 0.57}}                          # recall unset -> ignored
    assert qualifies_det({"f1": 0.60, "recall": 0.01}, cfg) is True
    assert qualifies_det({"f1": 0.60}, cfg) is True         # recall key absent entirely


def test_qualifies_det_missing_recall_score_treated_as_zero():
    cfg = {"target": {"f1": 0.10, "recall": 0.3}}           # F1 easily cleared
    assert qualifies_det({"f1": 0.60}, cfg) is False        # no recall in scores -> 0.0 < 0.3


# ---- pretrained_ckpt: SSL-backbone drop-in point -----------------------------------------------

def test_pretrained_ckpt_returns_path_when_set():
    cfg = {"model": {"name": "yolo11s", "pretrained_weights": "runs/ssl/backbone.pt"}}
    assert pretrained_ckpt(cfg) == "runs/ssl/backbone.pt"


def test_pretrained_ckpt_none_when_unset():
    assert pretrained_ckpt({"model": {"name": "yolo11s"}}) is None
    assert pretrained_ckpt({}) is None
    assert pretrained_ckpt({"model": None}) is None
