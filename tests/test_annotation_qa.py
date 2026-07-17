"""TDD for src.eval.annotation_qa: per-source box-geometry stats on the merged stenosis YOLO
dataset (Stage 2 Phase 2, P2.1). The diagnostic saw mAP50 0.209 vs mAP50-95 0.080 (a 2.6x
collapse) -- consistent with loosely-localized boxes, i.e. the three merged sources
(ARCADE/CADICA/Danilov) drawing/sizing boxes differently. This tool measures that mismatch
straight from label files, so these tests must stay torch-free (repo invariant: src/* and
tests/* import without torch/ultralytics/cv2 installed).
"""
import os

from src.eval.annotation_qa import parse_yolo_label, _pct, box_stats, summarize


# --- parse_yolo_label: text -> list of (cls, cx, cy, w, h) ------------------------------------

def test_parse_yolo_label_valid_lines():
    text = "0 0.5 0.5 0.1 0.2\n1 0.25 0.75 0.05 0.05\n"
    boxes = parse_yolo_label(text)
    assert boxes == [
        (0, 0.5, 0.5, 0.1, 0.2),
        (1, 0.25, 0.75, 0.05, 0.05),
    ]


def test_parse_yolo_label_skips_malformed_and_blank_lines():
    text = "0 0.5 0.5 0.1 0.2\n\n0 0.5 0.5\nnot five fields here at all\n1 0.1 0.1 0.02 0.02\n"
    boxes = parse_yolo_label(text)
    assert len(boxes) == 2
    assert boxes[0] == (0, 0.5, 0.5, 0.1, 0.2)
    assert boxes[1] == (1, 0.1, 0.1, 0.02, 0.02)


def test_parse_yolo_label_empty_text_returns_empty_list():
    assert parse_yolo_label("") == []


# --- _pct: manual percentile via linear interpolation -----------------------------------------

def test_pct_known_values():
    values = [0, 10, 20, 30, 40]
    assert _pct(values, 50) == 20
    assert _pct(values, 0) == 0
    assert _pct(values, 100) == 40


def test_pct_interpolates_between_points():
    # n=5 -> rank at q=25 is 0.25*4=1.0 exactly on index 1 (value 10); q=10 -> rank 0.4 -> 0 + 0.4*(10-0)=4
    values = [0, 10, 20, 30, 40]
    assert _pct(values, 10) == 4


def test_pct_unsorted_input_is_sorted_first():
    assert _pct([30, 0, 40, 10, 20], 50) == 20


def test_pct_empty_returns_none():
    assert _pct([], 50) is None


def test_pct_single_value():
    assert _pct([7.0], 50) == 7.0


# --- box_stats: geometry stats on a pooled box list -------------------------------------------

def _box(w, h, cls=0, cx=0.5, cy=0.5):
    return (cls, cx, cy, w, h)


def test_box_stats_empty_boxes_returns_zeros_sensibly():
    stats = box_stats([])
    assert stats["n_boxes"] == 0
    assert stats["tiny_frac"] == 0.0
    assert stats["w_p50"] is None
    assert stats["area_p50"] is None


def test_box_stats_n_boxes_and_percentile_ordering():
    boxes = [_box(w, w) for w in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.20, 0.30, 0.40, 0.50]]
    stats = box_stats(boxes)
    assert stats["n_boxes"] == 10
    assert stats["w_p10"] <= stats["w_p50"] <= stats["w_p90"]
    assert stats["area_p10"] <= stats["area_p50"] <= stats["area_p90"]


def test_box_stats_tiny_frac_straddles_threshold():
    # sqrt(area) < 0.05 is "tiny". Use square boxes so sqrt(area) == side length.
    # 3 tiny (side 0.02, 0.03, 0.049) vs 2 not-tiny (side 0.05 exactly is NOT < 0.05, and 0.2).
    boxes = [
        _box(0.02, 0.02),   # sqrt(area)=0.02 -> tiny
        _box(0.03, 0.03),   # sqrt(area)=0.03 -> tiny
        _box(0.049, 0.049), # sqrt(area)=0.049 -> tiny
        _box(0.05, 0.05),   # sqrt(area)=0.05 -> not tiny (strict <)
        _box(0.20, 0.20),   # sqrt(area)=0.20 -> not tiny
    ]
    stats = box_stats(boxes)
    assert stats["n_boxes"] == 5
    assert stats["tiny_frac"] == 0.6  # 3/5


def test_box_stats_area_is_w_times_h():
    boxes = [_box(0.1, 0.4)]  # area = 0.04
    stats = box_stats(boxes)
    assert stats["area_p50"] == 0.04


def test_box_stats_rounds_to_4dp():
    boxes = [_box(1.0 / 3.0, 1.0 / 3.0)]
    stats = box_stats(boxes)
    assert stats["w_p50"] == round(1.0 / 3.0, 4)


# --- summarize: walk labels/<split>/*.txt, bucket by source ------------------------------------

def _write_label(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def test_summarize_buckets_by_source_with_right_counts(tmp_path):
    proc = str(tmp_path)
    train_dir = os.path.join(proc, "labels", "train")

    # cadica: 2 images, 1 box each
    _write_label(os.path.join(train_dir, "p1_v1_00005.txt"), ["0 0.5 0.5 0.10 0.10"])
    _write_label(os.path.join(train_dir, "p1_v1_00006.txt"), ["0 0.4 0.4 0.12 0.12"])

    # danilov: 1 image, 2 boxes
    _write_label(
        os.path.join(train_dir, "14_002_5_0016.txt"),
        ["0 0.5 0.5 0.20 0.20", "0 0.3 0.3 0.05 0.05"],
    )

    # arcade: 1 image, empty label (background frame -- still counts as an image)
    _write_label(os.path.join(train_dir, "train_5.txt"), [])

    summary = summarize(proc, split="train")

    assert set(summary) == {"cadica", "danilov", "arcade"}

    assert summary["cadica"]["n_images"] == 2
    assert summary["cadica"]["n_boxes"] == 2
    assert summary["cadica"]["boxes_per_img"] == 1.0

    assert summary["danilov"]["n_images"] == 1
    assert summary["danilov"]["n_boxes"] == 2
    assert summary["danilov"]["boxes_per_img"] == 2.0

    assert summary["arcade"]["n_images"] == 1
    assert summary["arcade"]["n_boxes"] == 0
    assert summary["arcade"]["boxes_per_img"] == 0.0


def test_summarize_only_counts_requested_split(tmp_path):
    proc = str(tmp_path)
    _write_label(os.path.join(proc, "labels", "train", "train_1.txt"), ["0 0.5 0.5 0.1 0.1"])
    _write_label(os.path.join(proc, "labels", "val", "val_1.txt"), ["0 0.5 0.5 0.1 0.1"])

    train_summary = summarize(proc, split="train")
    assert train_summary["arcade"]["n_images"] == 1

    val_summary = summarize(proc, split="val")
    assert val_summary["arcade"]["n_images"] == 1


def test_summarize_missing_split_dir_returns_empty_dict(tmp_path):
    proc = str(tmp_path)
    os.makedirs(proc, exist_ok=True)
    assert summarize(proc, split="train") == {}


# --- import guard: module must be importable without ultralytics/torch/cv2 --------------------

def test_module_importable_without_ultralytics():
    import src.eval.annotation_qa  # noqa: F401
