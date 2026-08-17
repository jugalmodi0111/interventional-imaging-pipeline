"""Detector trainer: SSL seed selection (Grounding DINO cold-start) + speed/train kwargs.

The `_pseudo_label_round` section at the bottom runs the real SSL round on tiny synthetic PNGs with
a FAKE `ultralytics` module injected into sys.modules — no weights, no GPU, no ultralytics install
(the module's `from ultralytics import YOLO` is lazy, inside the function, exactly so this works).
"""
import glob
import os
import sys
import types

import cv2
import numpy as np
import pytest
import yaml

from src.train import train_detector as td
from src.train.train_detector import (
    ssl_seed, seed_prompt_and_classes, boxes_labels_to_yolo_lines, train_kwargs, run_tag,
    ssl_max_background_frac, cap_background_frames, ssl_restart_from, ssl_restart_weights,
)


def test_run_tag_arcade_only_nano():
    cfg = {"datasets": {"arcade_stenosis": {}},
           "model": {"name": "yolo11n", "imgsz": 640}, "train": {"epochs": 150}}
    assert run_tag(cfg) == "arcade_yolo11n_640_e150"


def test_run_tag_arcade_plus_danilov_s768():
    cfg = {"datasets": {"arcade_stenosis": {}, "danilov": {}},
           "model": {"name": "yolo11s", "imgsz": 768}, "train": {"epochs": 150}}
    assert run_tag(cfg) == "arcade+danilov_yolo11s_768_e150"


def test_ssl_seed_reads_cfg():
    assert ssl_seed({"ssl": {"seed": "gdino"}}) == "gdino"
    assert ssl_seed({"ssl": {}}) is None
    assert ssl_seed({}) is None


def test_seed_prompt_and_classes_stenosis():
    prompt, cmap = seed_prompt_and_classes({"task": "stenosis_detection"})
    assert prompt == "stenosis"
    assert cmap == {"stenosis": 0}


def test_seed_prompt_and_classes_catheter_uses_config_classes():
    cfg = {"task": "catheter_guidewire_tracking",
           "datasets": {"cathaction": {"classes": ["catheter", "guidewire"]}}}
    prompt, cmap = seed_prompt_and_classes(cfg)
    assert prompt == "catheter . guidewire"
    assert cmap == {"catheter": 0, "guidewire": 1}


def test_boxes_labels_to_yolo_lines_maps_known_skips_unknown():
    boxes = [[0, 0, 100, 100], [10, 10, 30, 30]]
    labels = ["stenosis", "background_noise"]
    lines = boxes_labels_to_yolo_lines(boxes, labels, {"stenosis": 0}, 100, 100)
    # only the known label survives; center-normalized, matches io_utils 6-dp format
    assert lines == ["0 0.500000 0.500000 1.000000 1.000000"]


def test_train_kwargs_fast_defaults():
    k = train_kwargs({})
    assert k["cache"] is True          # dataset caching = big I/O speedup
    assert k["amp"] is True            # mixed precision
    assert k["patience"] >= 1          # early stop once converged (quality-neutral)
    assert k["workers"] >= 1
    assert k["imgsz"] == 640


def test_train_kwargs_config_overrides():
    cfg = {"model": {"name": "yolo11n", "imgsz": 512},
           "train": {"epochs": 50, "batch": 8, "lr": 5e-4,
                     "cache": False, "workers": 2, "patience": 10}}
    k = train_kwargs(cfg)
    assert k["imgsz"] == 512 and k["epochs"] == 50 and k["batch"] == 8
    assert k["lr0"] == 5e-4 and k["cache"] is False and k["workers"] == 2 and k["patience"] == 10


def test_train_kwargs_no_augment_block_unchanged():
    # No 'augment' key -> base defaults only, no extra keys leak in.
    k = train_kwargs({})
    assert set(k.keys()) == {"imgsz", "epochs", "batch", "lr0", "cache", "workers", "patience", "amp"}
    assert k["imgsz"] == 640 and k["epochs"] == 100 and k["batch"] == 16
    assert k["lr0"] == 1e-3 and k["cache"] is True and k["workers"] == 8
    assert k["patience"] == 30 and k["amp"] is True


def test_train_kwargs_augment_block_passes_through():
    cfg = {"augment": {"mosaic": 0.0, "scale": 0.2, "erasing": 0.0, "box": 9.0, "cos_lr": True}}
    k = train_kwargs(cfg)
    assert k["mosaic"] == 0.0
    assert k["scale"] == 0.2
    assert k["erasing"] == 0.0
    assert k["box"] == 9.0
    assert k["cos_lr"] is True


def test_train_kwargs_augment_none_value_skipped():
    # A None-valued augment key leaves the base dict alone so ultralytics' own default stands.
    k = train_kwargs({"augment": {"mosaic": None}})
    assert "mosaic" not in k


def test_train_kwargs_augment_overrides_base_key():
    # augment can collide with a base key (e.g. imgsz) -- override wins.
    cfg = {"model": {"imgsz": 512}, "augment": {"imgsz": 1024}}
    k = train_kwargs(cfg)
    assert k["imgsz"] == 1024


# ---- ssl.max_background_frac + the deterministic background cap (pure, no cv2/ultralytics) -----

def test_ssl_max_background_frac_defaults_to_half():
    assert ssl_max_background_frac({}) == 0.5
    assert ssl_max_background_frac({"ssl": {}}) == 0.5
    assert ssl_max_background_frac({"ssl": {"max_background_frac": None}}) == 0.5


def test_ssl_max_background_frac_reads_cfg():
    assert ssl_max_background_frac({"ssl": {"max_background_frac": 0.1}}) == 0.1
    assert ssl_max_background_frac({"ssl": {"max_background_frac": 0}}) == 0.0


def test_cap_background_frames_default_frac_allows_one_bg_per_positive():
    paths = [f"f{i}.png" for i in range(10)]
    assert len(cap_background_frames(paths, 3, 0.5)) == 3       # bg/(pos+bg) = 3/6 = 0.5


def test_cap_background_frames_respects_a_tighter_frac():
    paths = [f"f{i}.png" for i in range(20)]
    # 10% backgrounds: 9 positives buy exactly 1 background (1/10 == 0.1)
    assert len(cap_background_frames(paths, 9, 0.1)) == 1


def test_cap_background_frames_zero_positives_adds_nothing():
    # The flood guard: a detector that fires on NOTHING must not turn the train split into
    # pure background (which would collapse training just as surely as zero negatives did).
    assert cap_background_frames([f"f{i}.png" for i in range(50)], 0, 0.5) == []


def test_cap_background_frames_frac_zero_is_off():
    assert cap_background_frames(["a.png", "b.png"], 5, 0.0) == []


def test_cap_background_frames_frac_one_is_uncapped():
    paths = ["b.png", "a.png"]
    assert cap_background_frames(paths, 0, 1.0) == ["a.png", "b.png"]     # sorted, all kept


def test_cap_background_frames_fewer_candidates_than_cap_keeps_all():
    assert cap_background_frames(["b.png", "a.png"], 10, 0.5) == ["a.png", "b.png"]


def test_cap_background_frames_selection_is_deterministic_and_evenly_spaced():
    paths = [f"f{i:02d}.png" for i in range(9)]
    kept = cap_background_frames(paths, 3, 0.5)                 # 3 of 9, sorted + evenly spaced
    assert kept == cap_background_frames(list(reversed(paths)), 3, 0.5)   # input order irrelevant
    assert kept == ["f00.png", "f04.png", "f08.png"]            # endpoints included, no RNG


# ---- ssl.restart_from: WHICH checkpoint an SSL round retrains from (pure, config-only) ---------
# Both rounds re-initialised from the stock `<name>.pt` instead of continuing from the weights that
# had just produced the pseudo-labels (docs/STENOSIS_ARCHITECTURE_AUDIT.md A3). That restart is a
# defensible confirmation-bias guard, but it was undocumented and unconfigurable, and it throws away
# everything the base run learned except what survives in the pseudo-labels. ssl.restart_from makes
# the choice explicit; the DEFAULT stays 'pretrained' so no existing run changes behaviour.

def test_ssl_restart_from_defaults_to_pretrained():
    assert ssl_restart_from({}) == "pretrained"
    assert ssl_restart_from({"ssl": {}}) == "pretrained"
    assert ssl_restart_from({"ssl": {"restart_from": None}}) == "pretrained"


def test_ssl_restart_from_reads_cfg():
    assert ssl_restart_from({"ssl": {"restart_from": "current"}}) == "current"
    assert ssl_restart_from({"ssl": {"restart_from": "pretrained"}}) == "pretrained"


def test_ssl_restart_from_rejects_an_unknown_mode_and_lists_the_valid_options():
    # Loud, not a silent fall back to the default: a typo'd knob would otherwise look honoured.
    with pytest.raises(ValueError) as e:
        ssl_restart_from({"ssl": {"restart_from": "warm_start"}})
    msg = str(e.value)
    assert "restart_from" in msg and "warm_start" in msg
    assert "pretrained" in msg and "current" in msg


def test_ssl_restart_weights_defaults_to_the_stock_pretrained_checkpoint():
    cfg = {"model": {"name": "yolo11s", "imgsz": 768}}
    assert ssl_restart_weights(cfg, "runs/stenosis/base/weights/best.pt") == "yolo11s.pt"


def test_ssl_restart_weights_pretrained_is_explicit_and_matches_the_default():
    cfg = {"model": {"name": "yolo11s"}, "ssl": {"restart_from": "pretrained"}}
    assert ssl_restart_weights(cfg, "best.pt") == ssl_restart_weights({"model": {"name": "yolo11s"}},
                                                                      "best.pt") == "yolo11s.pt"


def test_ssl_restart_weights_current_returns_the_weights_passed_in():
    cfg = {"model": {"name": "yolo11s"}, "ssl": {"restart_from": "current"}}
    assert ssl_restart_weights(cfg, "runs/stenosis/base/weights/best.pt") == \
        "runs/stenosis/base/weights/best.pt"


def test_ssl_restart_weights_uses_the_catheter_detector_key_and_an_explicit_override():
    assert ssl_restart_weights({"detector": {"name": "yolo11n"}}, "best.pt") == "yolo11n.pt"
    assert ssl_restart_weights({"model": {"name": "yolo11s"}}, "best.pt",
                               detector_name="yolo11m") == "yolo11m.pt"


def test_ssl_restart_weights_rejects_an_unknown_mode():
    with pytest.raises(ValueError) as e:
        ssl_restart_weights({"model": {"name": "yolo11s"}, "ssl": {"restart_from": "coco"}}, "b.pt")
    assert "pretrained" in str(e.value) and "current" in str(e.value)


def test_ssl_restart_weights_current_without_weights_fails_loudly():
    # 'current' with nothing to continue from is a config error, not a silent restart from COCO.
    with pytest.raises(ValueError) as e:
        ssl_restart_weights({"model": {"name": "yolo11s"}, "ssl": {"restart_from": "current"}}, None)
    assert "current" in str(e.value)


def test_shipped_config_pins_restart_from_pretrained():
    cfg = yaml.safe_load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "stenosis_yolo.yaml")))
    assert cfg["ssl"]["restart_from"] == "pretrained"           # behaviour unchanged unless opted out


# ---- _pseudo_label_round: zero-detection frames become BACKGROUNDS, not discards ----------------

class _Vals(list):
    """Stands in for the ultralytics box tensors, which expose .tolist()."""

    def tolist(self):
        return [list(v) if isinstance(v, (list, tuple)) else v for v in self]


class _FakeBoxes:
    def __init__(self, rows):                                   # rows: [(cls, xn, yn, wn, hn), ...]
        self.cls = _Vals([r[0] for r in rows])
        self.xywhn = _Vals([list(r[1:]) for r in rows])
        self._n = len(rows)

    def __len__(self):
        return self._n


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def _install_fake_yolo(monkeypatch, script):
    """Inject a fake `ultralytics` so the lazy `from ultralytics import YOLO` needs no install/GPU.

    `script[i]` is the detection list returned for the i-th predict() call, and the round walks the
    unlabeled frames in SORTED order, so script[i] belongs to the i-th sorted frame. An entry past
    the end of the script (or an empty list) = the detector fired on nothing = a background frame.
    train() is a no-op that just records it was called.

    `state['constructed']` is every checkpoint YOLO() was built from, in order, and
    `state['trained_from']` is the one the RETRAIN started from — which is what ssl.restart_from
    selects (stock '<name>.pt' vs the weights that produced the pseudo-labels).
    """
    state = {"n_predict": 0, "trained": None, "constructed": [], "trained_from": None}

    class _YOLO:
        def __init__(self, weights):
            self.weights = weights
            state["constructed"].append(weights)

        def predict(self, frame, **kw):
            i = state["n_predict"]
            state["n_predict"] += 1
            return [_FakeResult(_FakeBoxes(script[i] if i < len(script) else []))]

        def train(self, **kw):
            state["trained"] = kw
            state["trained_from"] = self.weights

    mod = types.ModuleType("ultralytics")
    mod.YOLO = _YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return state


def _run_round(root, monkeypatch, script, n_frames, ssl_cfg=None, state_out=None):
    """Run the real _pseudo_label_round over `n_frames` synthetic PNGs; return (images, labels) dirs.

    Pass a dict as `state_out` to receive the fake-YOLO state (see `_install_fake_yolo`) once the
    round has finished — e.g. state_out['trained_from'], the checkpoint the retrain started from."""
    udir, proc = root / "unlabeled", root / "proc"
    udir.mkdir(parents=True); proc.mkdir(parents=True)
    for i in range(n_frames):
        cv2.imwrite(str(udir / f"f{i:03d}.png"), np.full((32, 32), 20 + 7 * i, np.uint8))
    state = _install_fake_yolo(monkeypatch, script)
    cfg = {"task": "stenosis_detection", "model": {"name": "yolo11n", "imgsz": 32},
           "ssl": dict({"pseudo_label": True, "conf": 0.4}, **(ssl_cfg or {}))}
    try:
        td._pseudo_label_round("best.pt", cfg, str(root / "runs"), str(proc / "data.yaml"),
                               unlabeled_dir=str(udir), device="cpu")
    finally:
        if state_out is not None:
            state_out.update(state)
    return proc / "images" / "train", proc / "labels" / "train"


def _stems(d, ext):
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(str(d / ("*" + ext))))


def test_pseudo_label_round_writes_a_background_for_a_zero_detection_frame(tmp_path, monkeypatch):
    # f000 -> one box (positive), f001 -> nothing. The zero-detection frame is the BACKGROUND
    # example the corpus never had; it must be written, not dropped.
    img, lbl = _run_round(tmp_path, monkeypatch, [[(0, 0.5, 0.5, 0.2, 0.2)], []], 2)
    assert (img / "pl_f001.png").exists()
    assert (lbl / "pl_f001.txt").exists()
    assert (lbl / "pl_f001.txt").read_text().strip() == ""       # EMPTY == background to ultralytics


def test_pseudo_label_round_still_writes_boxes_for_a_detected_frame(tmp_path, monkeypatch):
    img, lbl = _run_round(tmp_path, monkeypatch, [[(0, 0.5, 0.5, 0.2, 0.2)], []], 2)
    assert (img / "pl_f000.png").exists()
    assert (lbl / "pl_f000.txt").read_text() == "0 0.500000 0.500000 0.200000 0.200000"


def test_pseudo_label_round_keeps_multi_class_ids_and_multiple_boxes(tmp_path, monkeypatch):
    script = [[(1, 0.25, 0.25, 0.1, 0.1), (0, 0.75, 0.75, 0.4, 0.4)]]
    _, lbl = _run_round(tmp_path, monkeypatch, script, 1)
    assert (lbl / "pl_f000.txt").read_text().splitlines() == [
        "1 0.250000 0.250000 0.100000 0.100000",
        "0 0.750000 0.750000 0.400000 0.400000",
    ]


def test_max_background_frac_caps_the_backgrounds_added(tmp_path, monkeypatch):
    # 1 positive + 4 zero-detection frames; default frac 0.5 -> only 1 background may join.
    img, lbl = _run_round(tmp_path, monkeypatch, [[(0, 0.5, 0.5, 0.2, 0.2)]], 5)
    empties = [p for p in glob.glob(str(lbl / "*.txt")) if not open(p).read().strip()]
    assert len(empties) == 1
    assert len(_stems(img, ".png")) == 2                        # 1 positive + 1 background


def test_max_background_frac_is_configurable(tmp_path, monkeypatch):
    # 3 positives, frac 0.25 -> 1 background (1/4 of the 4 frames added).
    script = [[(0, 0.5, 0.5, 0.2, 0.2)]] * 3
    img, lbl = _run_round(tmp_path, monkeypatch, script, 9, ssl_cfg={"max_background_frac": 0.25})
    empties = [p for p in glob.glob(str(lbl / "*.txt")) if not open(p).read().strip()]
    assert len(empties) == 1
    assert len(_stems(img, ".png")) == 4


def test_pseudo_label_round_with_no_detections_at_all_adds_nothing(tmp_path, monkeypatch):
    # Flood guard: 0 positives -> 0 backgrounds, so a dead detector can't fill train with background.
    img, lbl = _run_round(tmp_path, monkeypatch, [], 6)
    assert _stems(img, ".png") == [] and _stems(lbl, ".txt") == []


def test_background_selection_is_deterministic_across_runs(tmp_path, monkeypatch):
    # 2 positives + 6 zero-detection frames -> cap 2; the SAME 2 backgrounds every run (no RNG).
    script = [[(0, 0.5, 0.5, 0.2, 0.2)]] * 2
    a_img, _ = _run_round(tmp_path / "a", monkeypatch, script, 8)
    b_img, _ = _run_round(tmp_path / "b", monkeypatch, script, 8)
    assert _stems(a_img, ".png") == _stems(b_img, ".png")
    assert len(_stems(a_img, ".png")) == 4                      # 2 positives + 2 of the 6 backgrounds


def test_pseudo_label_round_reports_positives_and_backgrounds_separately(tmp_path, monkeypatch,
                                                                        capsys):
    _run_round(tmp_path, monkeypatch, [[(0, 0.5, 0.5, 0.2, 0.2)], []], 2)
    out = capsys.readouterr().out
    assert "1 positives" in out and "1 backgrounds" in out


# --- ssl.restart_from, end to end through the pseudo-label round --------------------------------

def test_pseudo_label_round_retrains_from_the_stock_pretrained_checkpoint_by_default(tmp_path,
                                                                                     monkeypatch):
    st = {}
    _run_round(tmp_path, monkeypatch, [[(0, 0.5, 0.5, 0.2, 0.2)], []], 2, state_out=st)
    assert st["trained_from"] == "yolo11n.pt"                   # A3's documented current behaviour
    assert st["constructed"][0] == "best.pt"                    # ...but predicting still uses `weights`


def test_pseudo_label_round_restart_from_current_continues_from_the_round_weights(tmp_path,
                                                                                  monkeypatch):
    st = {}
    _run_round(tmp_path, monkeypatch, [[(0, 0.5, 0.5, 0.2, 0.2)], []], 2,
               ssl_cfg={"restart_from": "current"}, state_out=st)
    assert st["trained_from"] == "best.pt"                      # the weights that made the labels


def test_pseudo_label_round_prints_the_restart_mode(tmp_path, monkeypatch, capsys):
    _run_round(tmp_path, monkeypatch, [[(0, 0.5, 0.5, 0.2, 0.2)], []], 2)
    assert "restart_from=pretrained" in capsys.readouterr().out


def test_pseudo_label_round_prints_the_restart_mode_when_continuing(tmp_path, monkeypatch, capsys):
    _run_round(tmp_path, monkeypatch, [[(0, 0.5, 0.5, 0.2, 0.2)], []], 2,
               ssl_cfg={"restart_from": "current"})
    assert "restart_from=current" in capsys.readouterr().out


def test_pseudo_label_round_rejects_an_unknown_restart_mode(tmp_path, monkeypatch):
    with pytest.raises(ValueError) as e:
        _run_round(tmp_path, monkeypatch, [[(0, 0.5, 0.5, 0.2, 0.2)]], 1,
                   ssl_cfg={"restart_from": "warm"})
    assert "pretrained" in str(e.value) and "current" in str(e.value)


# --- gdino seed round: the SAME zero-negative defect, third instance -----------------------------
# `_pseudo_label_round` was fixed 2026-08-16 to write zero-detection frames as backgrounds instead
# of discarding them. `_gdino_seed_round` still had the identical `if not lines: continue`, so the
# open-vocab cold start re-created the very corpus the fix removed. Same bug, same fix, same cap.

def _install_fake_gdino(monkeypatch, script):
    """Fake the lazy GD imports so no torch/transformers/GPU is needed. script[i] -> i-th sorted
    frame's boxes; [] means GD found nothing = a background frame."""
    state = {"n": 0, "trained": None, "constructed": [], "trained_from": None}
    ag = types.ModuleType("src.data_prep.autolabel_gdino")
    ag.detect = lambda g, prompt, **kw: (None, None, None)
    ag.filter_detections = lambda b, s, l, c: (b, s, l)
    monkeypatch.setitem(sys.modules, "src.data_prep.autolabel_gdino", ag)

    def fake_lines(boxes, labels, class_map, W, H):
        i = state["n"]; state["n"] += 1
        return script[i] if i < len(script) else []
    monkeypatch.setattr(td, "boxes_labels_to_yolo_lines", fake_lines, raising=False)
    monkeypatch.setattr(td, "seed_prompt_and_classes", lambda cfg: ("stenosis", {0: 0}), raising=False)

    class _YOLO:
        def __init__(self, w):
            self.weights = w
            state["constructed"].append(w)

        def train(self, **kw):
            state["trained"] = kw
            state["trained_from"] = self.weights                # ssl.restart_from picks this ckpt
    mod = types.ModuleType("ultralytics"); mod.YOLO = _YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return state


def _run_gdino(tmp_path, monkeypatch, script, n_frames, ssl_cfg=None, weights=None, state_out=None):
    root = tmp_path
    udir = root / "unlabeled"; udir.mkdir()
    proc = root / "proc"; (proc / "images" / "train").mkdir(parents=True)
    (proc / "labels" / "train").mkdir(parents=True)
    (proc / "data.yaml").write_text("nc: 1\n")
    for i in range(n_frames):
        cv2.imwrite(str(udir / f"f{i:03d}.png"), np.full((32, 32), 120, np.uint8))
    state = _install_fake_gdino(monkeypatch, script)
    cfg = {"task": "stenosis_detection", "model": {"name": "yolo11n", "imgsz": 32},
           "ssl": dict({"seed": "gdino", "conf": 0.4}, **(ssl_cfg or {}))}
    try:
        td._gdino_seed_round(cfg, str(root / "runs"), str(proc / "data.yaml"), weights=weights,
                             unlabeled_dir=str(udir), device="cpu")
    finally:
        if state_out is not None:
            state_out.update(state)
    return proc / "images" / "train", proc / "labels" / "train"


def test_gdino_seed_writes_a_background_for_a_zero_detection_frame(tmp_path, monkeypatch):
    # GD finds nothing on frame 1 -> it must become a BACKGROUND (image + empty label), not vanish.
    imgs, lbls = _run_gdino(tmp_path, monkeypatch,
                            script=[["0 0.5 0.5 0.1 0.1"], []], n_frames=2)
    bg = [p for p in lbls.glob("*.txt") if p.read_text().strip() == ""]
    assert len(bg) == 1, f"zero-detection frame was discarded, not written as background: {list(lbls.glob('*.txt'))}"
    assert (imgs / (bg[0].stem + ".png")).exists(), "background label written without its image"


def test_gdino_seed_still_writes_boxes_for_a_detected_frame(tmp_path, monkeypatch):
    imgs, lbls = _run_gdino(tmp_path, monkeypatch,
                            script=[["0 0.5 0.5 0.1 0.1"], []], n_frames=2)
    pos = [p for p in lbls.glob("*.txt") if p.read_text().strip()]
    assert len(pos) == 1 and pos[0].read_text().strip() == "0 0.5 0.5 0.1 0.1"


def test_gdino_seed_respects_max_background_frac(tmp_path, monkeypatch):
    # 1 positive, 4 zero-detection frames, frac 0.5 -> at most 1 background survives.
    imgs, lbls = _run_gdino(tmp_path, monkeypatch,
                            script=[["0 0.5 0.5 0.1 0.1"], [], [], [], []], n_frames=5,
                            ssl_cfg={"max_background_frac": 0.5})
    bg = [p for p in lbls.glob("*.txt") if p.read_text().strip() == ""]
    assert len(bg) == 1, f"expected the cap to keep 1 background, got {len(bg)}"


def test_gdino_seed_zero_positives_adds_no_background(tmp_path, monkeypatch):
    # GD fired on nothing at all -> must not flood train with pure background.
    imgs, lbls = _run_gdino(tmp_path, monkeypatch, script=[[], [], []], n_frames=3)
    assert list(lbls.glob("*.txt")) == []


# --- ssl.restart_from, end to end through the Grounding-DINO seed round -------------------------

def test_gdino_seed_retrains_from_the_stock_pretrained_checkpoint_by_default(tmp_path, monkeypatch):
    st = {}
    _run_gdino(tmp_path, monkeypatch, script=[["0 0.5 0.5 0.1 0.1"], []], n_frames=2, state_out=st)
    assert st["trained_from"] == "yolo11n.pt"


def test_gdino_seed_restart_from_current_continues_from_the_weights_passed_in(tmp_path, monkeypatch):
    st = {}
    _run_gdino(tmp_path, monkeypatch, script=[["0 0.5 0.5 0.1 0.1"], []], n_frames=2,
               ssl_cfg={"restart_from": "current"}, weights="runs/stenosis/base/weights/best.pt",
               state_out=st)
    assert st["trained_from"] == "runs/stenosis/base/weights/best.pt"


def test_gdino_seed_restart_from_current_defaults_to_the_base_run_weights(tmp_path, monkeypatch):
    # Called with no explicit weights (round 0), 'current' means the base run's best.pt.
    st = {}
    _run_gdino(tmp_path, monkeypatch, script=[["0 0.5 0.5 0.1 0.1"], []], n_frames=2,
               ssl_cfg={"restart_from": "current"}, state_out=st)
    assert st["trained_from"] == os.path.join(str(tmp_path / "runs"), "base", "weights", "best.pt")


def test_gdino_seed_prints_the_restart_mode(tmp_path, monkeypatch, capsys):
    _run_gdino(tmp_path, monkeypatch, script=[["0 0.5 0.5 0.1 0.1"], []], n_frames=2)
    assert "restart_from=pretrained" in capsys.readouterr().out


def test_gdino_seed_rejects_an_unknown_restart_mode(tmp_path, monkeypatch):
    with pytest.raises(ValueError) as e:
        _run_gdino(tmp_path, monkeypatch, script=[["0 0.5 0.5 0.1 0.1"]], n_frames=1,
                   ssl_cfg={"restart_from": "finetune"})
    assert "pretrained" in str(e.value) and "current" in str(e.value)
