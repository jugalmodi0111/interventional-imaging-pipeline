"""TDD for the build-side Grounding DINO auto-labeler (pure, model-free helpers).

Only numpy/cv2/pytest are exercised here: the heavy detect()/SAM paths lazy-import
torch/transformers on the GPU build side and are NOT run locally. Importing the module
(and src.models.grounded_sam) must succeed with no torch installed.
"""
import numpy as np

from src.data_prep.autolabel_gdino import (
    DEFAULT_PROMPTS,
    dino_boxes_to_yolo_lines,
    filter_detections,
    dino_to_coco,
)


# --- module import must not need torch ---------------------------------------

def test_modules_import_without_torch():
    import importlib
    importlib.import_module("src.data_prep.autolabel_gdino")
    importlib.import_module("src.models.grounded_sam")  # lazy heavy imports


# --- dino_boxes_to_yolo_lines ------------------------------------------------

def test_full_frame_box_maps_to_center_normalized_line():
    W, H = 640, 480
    assert dino_boxes_to_yolo_lines([[0, 0, W, H]], W, H) == [
        "0 0.500000 0.500000 1.000000 1.000000"
    ]


def test_boxes_to_yolo_uses_box_center_and_six_decimals():
    # x:[100,300] y:[100,200] in 400x400 -> center (200,150), wh (200,100)
    assert dino_boxes_to_yolo_lines([[100, 100, 300, 200]], 400, 400, class_id=2) == [
        "2 0.500000 0.375000 0.500000 0.250000"
    ]


def test_boxes_to_yolo_matches_io_utils_arithmetic():
    from src.data_prep import io_utils  # smoke: io_utils importable w/o torch
    assert hasattr(io_utils, "split_of")
    W, H, box = 512, 512, [10, 20, 110, 120]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    expect = f"0 {(x1 + w / 2) / W:.6f} {(y1 + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
    assert dino_boxes_to_yolo_lines([box], W, H) == [expect]


def test_boxes_to_yolo_handles_numpy_and_multiple_boxes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 20, 20]], dtype=float)
    lines = dino_boxes_to_yolo_lines(boxes, 20, 20, class_id=1)
    assert len(lines) == 2 and lines[0].startswith("1 ") and lines[1].startswith("1 ")


def test_boxes_to_yolo_empty_returns_empty():
    assert dino_boxes_to_yolo_lines([], 100, 100) == []


# --- filter_detections -------------------------------------------------------

def test_filter_detections_keeps_scores_at_or_above_threshold():
    boxes = [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]]
    fb, fs, fl = filter_detections(boxes, [0.9, 0.2, 0.35], ["a", "b", "c"], box_thresh=0.35)
    assert list(fl) == ["a", "c"]
    assert np.allclose(fs, [0.9, 0.35])
    assert fb.shape == (2, 4)


def test_filter_detections_empty_input():
    fb, fs, fl = filter_detections([], [], [], box_thresh=0.5)
    assert fb.shape == (0, 4) and len(fs) == 0 and len(fl) == 0


# --- dino_to_coco ------------------------------------------------------------

def test_dino_to_coco_shapes_and_bbox_is_xywh():
    dets = [
        {"file_name": "a.png", "width": 100, "height": 100,
         "boxes": [[10, 20, 30, 60]], "scores": [0.8], "labels": ["stenosis"]},
        {"file_name": "b.png", "width": 50, "height": 50,
         "boxes": [], "scores": [], "labels": []},
    ]
    coco = dino_to_coco(dets, ["stenosis", "catheter"])
    assert set(coco) == {"images", "annotations", "categories"}
    assert [c["name"] for c in coco["categories"]] == ["stenosis", "catheter"]
    assert [c["id"] for c in coco["categories"]] == [1, 2]        # 1-based COCO ids
    assert len(coco["images"]) == 2
    ann = coco["annotations"][0]
    assert ann["bbox"] == [10, 20, 20, 40]                        # [x,y,w,h] top-left + wh
    assert ann["category_id"] == 1                               # 'stenosis'
    assert ann["image_id"] == coco["images"][0]["id"]
    assert ann["area"] == 20 * 40
    assert all(isinstance(a["id"], int) for a in coco["annotations"])


def test_dino_to_coco_maps_label_strings_to_category_ids():
    dets = [{"file_name": "x.png", "width": 10, "height": 10,
             "boxes": [[0, 0, 5, 5], [1, 1, 2, 2]], "scores": [0.9, 0.9],
             "labels": ["catheter", "stenosis"]}]
    coco = dino_to_coco(dets, ["stenosis", "catheter"])
    assert {a["category_id"] for a in coco["annotations"]} == {1, 2}


def test_dino_to_coco_skips_unknown_labels():
    dets = [{"file_name": "x.png", "width": 10, "height": 10,
             "boxes": [[0, 0, 5, 5]], "scores": [0.9], "labels": ["unknown_thing"]}]
    assert dino_to_coco(dets, ["stenosis"])["annotations"] == []


def test_dino_to_coco_is_json_serializable():
    import json
    dets = [{"file_name": "a.png", "width": 100, "height": 100,
             "boxes": np.array([[10.0, 20.0, 30.0, 60.0]]), "scores": np.array([0.8]),
             "labels": ["stenosis"]}]
    json.dumps(dino_to_coco(dets, ["stenosis"]))  # must not raise on numpy scalars


# --- DEFAULT_PROMPTS ---------------------------------------------------------

def test_default_prompts_are_dino_dot_separated():
    assert set(DEFAULT_PROMPTS) >= {"stenosis", "catheter", "coronary"}
    assert " . " in DEFAULT_PROMPTS["catheter"]                   # DINO class separator
    assert all(isinstance(v, str) for v in DEFAULT_PROMPTS.values())
