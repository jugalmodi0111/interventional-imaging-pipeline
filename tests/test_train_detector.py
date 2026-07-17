"""Detector trainer: SSL seed selection (Grounding DINO cold-start) + speed/train kwargs."""
from src.train.train_detector import (
    ssl_seed, seed_prompt_and_classes, boxes_labels_to_yolo_lines, train_kwargs, run_tag,
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
