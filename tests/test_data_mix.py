"""Data-mix guards for the honest stenosis corpus:

Task 1 — Danilov per-patient frame cap: kills the ~8325-frames/64-patients redundancy that dilutes
the honest per-patient metric. `cap_frames_per_patient` keeps at most k EVENLY-SPACED frames per
patient (deterministic, no RNG); the Danilov converter only writes the retained frames.

Task 2 — CADICA (patient-diverse) -> YOLO single class 'stenosis'. Pure box math is tested in
isolation (stdlib only); `main` is exercised end-to-end on a synthetic CADICA tmp tree (no cv2
dataset needed beyond tiny PNGs) to prove file counts, PATIENT-grouped split, and normalized boxes.
"""
import glob
import os

import cv2
import numpy as np
import pytest

from src.data_prep import cadica_to_yolo as cad
from src.data_prep import danilov_to_yolo as dan
from src.data_prep import io_utils as io
from src.data_prep.io_utils import cap_frames_per_patient, group_key, split_of


# ==================================================================================================
# Task 1: cap_frames_per_patient
# ==================================================================================================
def _frame_idx(stem):
    """'14_002_5_0037' -> 37 (the trailing frame index)."""
    return int(stem.split("_")[-1])


def test_cap_keeps_k_evenly_spaced_per_patient_deterministic():
    # 100 frames of 2 patients (50 each), k=5 -> 10 kept, evenly spaced from first to last frame.
    a = [f"14_002_5_{i:04d}" for i in range(50)]
    b = [f"14_050_2_{i:04d}" for i in range(50)]
    kept = cap_frames_per_patient(a + b, 5)

    assert len(kept) == 10, "5 per patient x 2 patients"
    groups = {group_key(s) for s in kept}
    assert groups == {"14_002", "14_050"}

    for g in ("14_002", "14_050"):
        idxs = sorted(_frame_idx(s) for s in kept if group_key(s) == g)
        # linspace(0, 49, 5) rounded -> endpoints included, ~evenly spaced
        assert idxs == [0, 12, 24, 37, 49], f"{g}: not evenly spaced, got {idxs}"

    # deterministic: identical result on a reshuffled input, no RNG
    assert cap_frames_per_patient(list(reversed(a + b)), 5) == kept


def test_cap_none_keeps_all_frames():
    stems = [f"14_002_5_{i:04d}" for i in range(20)]
    assert cap_frames_per_patient(stems, None) == sorted(stems)


def test_cap_smaller_than_k_keeps_all_of_that_group():
    # patient with fewer than k frames is kept whole; the big patient is capped.
    small = [f"14_009_1_{i:04d}" for i in range(3)]
    big = [f"14_002_5_{i:04d}" for i in range(40)]
    kept = cap_frames_per_patient(small + big, 5)
    assert sum(group_key(s) == "14_009" for s in kept) == 3
    assert sum(group_key(s) == "14_002" for s in kept) == 5


def test_cap_k_one_picks_single_middle_frame():
    stems = [f"14_002_5_{i:04d}" for i in range(11)]
    kept = cap_frames_per_patient(stems, 1)
    assert kept == ["14_002_5_0005"]   # middle of 0..10


# --- Danilov converter wiring: only the capped frames are written (already-YOLO layout) -----------
def _write_danilov_yolo_tree(root, patients, frames_per_patient):
    """Synthetic Danilov 'already-YOLO' layout: <stem>.bmp + sibling <stem>.txt (one box each)."""
    os.makedirs(root, exist_ok=True)
    for site, pid, seq in patients:
        for i in range(frames_per_patient):
            stem = f"{site}_{pid}_{seq}_{i:04d}"
            cv2.imwrite(os.path.join(root, stem + ".bmp"), np.full((32, 32), 90, np.uint8))
            with open(os.path.join(root, stem + ".txt"), "w") as f:
                f.write("0 0.5 0.5 0.2 0.2\n")


def _split_stems(out_dir):
    out = {}
    for sp in ("train", "val"):
        d = os.path.join(out_dir, "images", sp)
        out[sp] = {os.path.splitext(f)[0] for f in os.listdir(d)} if os.path.isdir(d) else set()
    return out


def test_danilov_native_caps_frames_per_patient(tmp_path):
    root = str(tmp_path / "danilov")
    out = str(tmp_path / "out")
    _write_danilov_yolo_tree(root, [("14", "002", "5"), ("14", "050", "2")], frames_per_patient=50)

    n = dan._danilov_native(root, out, size=32, max_frames_per_patient=5)

    assert n == 10, "each of 2 patients capped to 5 evenly-spaced frames"
    written = _split_stems(out)
    all_stems = written["train"] | written["val"]
    assert len(all_stems) == 10
    # per patient: exactly 5, and every patient's frames land ENTIRELY on one split (no leak)
    for g in ("14_002", "14_050"):
        pf = {s for s in all_stems if group_key(s) == g}
        assert len(pf) == 5
        assert pf <= written["train"] or pf <= written["val"], f"{g} split across train/val"


def test_danilov_native_no_cap_writes_all(tmp_path):
    root = str(tmp_path / "danilov")
    out = str(tmp_path / "out")
    _write_danilov_yolo_tree(root, [("14", "002", "5")], frames_per_patient=12)
    n = dan._danilov_native(root, out, size=32, max_frames_per_patient=None)
    assert n == 12


# ==================================================================================================
# Task 2: CADICA pure box math
# ==================================================================================================
def test_cadica_boxes_to_yolo_lines_normalizes_topleft_to_center_class0():
    # abs top-left (x,y,w,h) in a 100x200 image -> center-form normalized, class 0
    lines = cad.cadica_boxes_to_yolo_lines([(10, 20, 30, 40)], W=100, H=200)
    assert lines == ["0 0.250000 0.200000 0.300000 0.200000"]


def test_cadica_boxes_multiple_all_class0_regardless_of_severity():
    boxes = [(0, 0, 50, 50), (60, 80, 20, 40)]
    lines = cad.cadica_boxes_to_yolo_lines(boxes, W=100, H=100)
    assert [ln.split()[0] for ln in lines] == ["0", "0"]     # every severity -> stenosis class 0
    assert lines[0] == "0 0.250000 0.250000 0.500000 0.500000"
    assert lines[1] == "0 0.700000 1.000000 0.200000 0.400000"


def test_parse_cadica_gt_drops_label_and_skips_malformed():
    text = "10 20 30 40 p20_50\n\n5 5 10 10 p70_90\nbroken line\n1 2 3\n"
    assert cad.parse_cadica_gt(text) == [(10.0, 20.0, 30.0, 40.0), (5.0, 5.0, 10.0, 10.0)]


def test_cadica_patient_of():
    assert cad.cadica_patient_of("/x/selectedVideos/p12/v3/input") == "p12"
    assert cad.cadica_patient_of("/x/vids/v3/input") is None


# --- CADICA main end-to-end on a synthetic tree --------------------------------------------------
def _write_cadica_tree(root, patients, frames_per_video, box="10 20 30 40 p20_50", size=100):
    """selectedVideos/pXX/vYY/{input/<f>.png, groundTruth/<f>.txt}. Returns expected out stems."""
    expected = {}
    for pid in patients:
        vdir = os.path.join(root, "selectedVideos", pid, "v1")
        indir, gtdir = os.path.join(vdir, "input"), os.path.join(vdir, "groundTruth")
        os.makedirs(indir, exist_ok=True)
        os.makedirs(gtdir, exist_ok=True)
        stems = []
        for i in range(frames_per_video):
            fstem = f"{i:05d}"
            cv2.imwrite(os.path.join(indir, fstem + ".png"), np.full((size, size), 120, np.uint8))
            with open(os.path.join(gtdir, fstem + ".txt"), "w") as f:
                f.write(box + "\n")
            stems.append(f"{pid}_v1_{fstem}")           # _out_stem for non-CADICA-named frames
        expected[pid] = stems
    return expected


def test_cadica_main_counts_patient_split_and_boxes(tmp_path, monkeypatch):
    root = tmp_path / "cadica"
    out = tmp_path / "out"
    monkeypatch.setattr(cad, "OUT", str(out))
    expected = _write_cadica_tree(str(root), ["p1", "p2"], frames_per_video=3)

    cfg = {"datasets": {"cadica": {"root": str(root)}}, "model": {"imgsz": 64}}
    n = cad.main(cfg)

    assert n == 6, "2 patients x 3 frames"
    imgs = glob.glob(os.path.join(str(out), "images", "*", "*.png"))
    lbls = glob.glob(os.path.join(str(out), "labels", "*", "*.txt"))
    assert len(imgs) == 6 and len(lbls) == 6

    written = _split_stems(str(out))
    # (a) every patient's frames land entirely on the split split_of(patient) chose -> no leakage
    for pid, stems in expected.items():
        sp = split_of(pid)
        assert set(stems) <= written[sp], f"{pid} not all in {sp}"
        other = "val" if sp == "train" else "train"
        assert not (set(stems) & written[other]), f"{pid} leaked into {other}"

    # (b) boxes normalized by ORIGINAL 100x100 image (resolution-independent of the 64px resize)
    lp = os.path.join(str(out), "labels", split_of("p1"), "p1_v1_00000.txt")
    with open(lp) as f:
        assert f.read().strip() == "0 0.250000 0.400000 0.300000 0.400000"


def test_cadica_main_robust_to_missing_root(tmp_path):
    # missing dataset dir must skip, not crash
    cfg = {"datasets": {"cadica": {"root": str(tmp_path / "nope")}}}
    assert cad.main(cfg) == 0
    assert cad.main({"datasets": {}}) == 0     # no 'cadica' key at all
