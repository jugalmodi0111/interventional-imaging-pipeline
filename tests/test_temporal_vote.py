"""Unit tests for temporal-voting post-processing (src.serve.temporal_vote).

Torch-/numpy-/model-free: the module is pure, so the recall (gap recovery) and precision (flicker
rejection) behaviour is pinned here without a detector or a GPU. Each test locks one property so a
regression -- a lost lesion frame or a resurrected flicker -- surfaces loudly.
"""
import pytest

from src.serve.temporal_vote import iou_xywhn, link_tracks, aggregate_sequence


# --- IoU math on normalized YOLO boxes ----------------------------------------------------------

def test_iou_identical_boxes_is_one():
    assert iou_xywhn((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou_xywhn((0.1, 0.1, 0.1, 0.1), (0.9, 0.9, 0.1, 0.1)) == 0.0


def test_iou_half_shifted_is_one_third():
    # Same size, shifted by exactly half a width in x: overlap is half of each -> 0.02 / 0.06.
    assert iou_xywhn((0.5, 0.5, 0.2, 0.2), (0.6, 0.5, 0.2, 0.2)) == pytest.approx(1 / 3)


def test_iou_containment_is_area_ratio():
    # Small box fully inside a 2x-per-side box: inter = small area, union = big area -> 0.04/0.16.
    assert iou_xywhn((0.5, 0.5, 0.4, 0.4), (0.5, 0.5, 0.2, 0.2)) == pytest.approx(0.25)


def test_iou_is_symmetric_and_zero_area_safe():
    a, b = (0.3, 0.3, 0.2, 0.2), (0.4, 0.35, 0.2, 0.2)
    assert iou_xywhn(a, b) == pytest.approx(iou_xywhn(b, a))
    assert iou_xywhn((0.5, 0.5, 0.0, 0.0), (0.5, 0.5, 0.2, 0.2)) == 0.0   # degenerate -> 0, no div0


# --- linking: bridge a single-frame gap into ONE track ------------------------------------------

def _lesion(cx, conf=0.8):
    return {"box": (cx, 0.5, 0.2, 0.2), "conf": conf}


def test_link_bridges_single_frame_gap():
    # Lesion detected on frames 0, 2, 4 (missed 1 & 3). max_gap=1 must link all three into ONE track.
    frames = [[_lesion(0.50)], [], [_lesion(0.52)], [], [_lesion(0.54)]]
    tracks = link_tracks(frames, iou_thr=0.3)
    assert len(tracks) == 1
    assert [fi for fi, _ in tracks[0]] == [0, 2, 4]


def test_link_keeps_distant_flicker_separate():
    # A far-away one-frame box must NOT be linked onto the lesion track (IoU 0 < thr).
    frames = [[_lesion(0.50)], [{"box": (0.1, 0.1, 0.1, 0.1), "conf": 0.4}], [_lesion(0.52)]]
    tracks = link_tracks(frames, iou_thr=0.3)
    lens = sorted(len(t) for t in tracks)
    assert lens == [1, 2]   # lesion track of 2, flicker track of 1


# --- (a) 3/5-frame lesion survives min_hits=2 and gap frames get interpolated --------------------

def _seq_with_flicker():
    return [
        [_lesion(0.50)],                                    # f0 lesion
        [{"box": (0.1, 0.1, 0.1, 0.1), "conf": 0.4}],       # f1 flicker only
        [_lesion(0.52, conf=0.7)],                          # f2 lesion
        [],                                                 # f3 nothing detected
        [_lesion(0.54, conf=0.9)],                          # f4 lesion
    ]


def test_persistent_lesion_survives_and_spans_all_frames():
    stab = aggregate_sequence(_seq_with_flicker(), iou_thr=0.3, min_hits=2)
    # Survivor spans frames 0..4 -> exactly one stabilized detection on EVERY frame (misses recovered).
    assert [len(f) for f in stab] == [1, 1, 1, 1, 1]


def test_gap_frames_are_interpolated_and_observed_are_not():
    stab = aggregate_sequence(_seq_with_flicker(), iou_thr=0.3, min_hits=2)
    interp = [stab[fi][0]["interpolated"] for fi in range(5)]
    assert interp == [False, True, False, True, False]      # frames 1 & 3 recovered by interpolation
    # Frame-1 interpolated box sits halfway between the f0 and f2 lesion boxes (cx 0.50 -> 0.52).
    assert stab[1][0]["box"][0] == pytest.approx(0.51)
    assert stab[3][0]["box"][0] == pytest.approx(0.53)      # halfway 0.52 -> 0.54


def test_aggregated_confidence_is_track_mean():
    stab = aggregate_sequence(_seq_with_flicker(), iou_thr=0.3, min_hits=2, conf_agg="mean")
    mean_conf = (0.8 + 0.7 + 0.9) / 3
    assert all(d["conf"] == pytest.approx(mean_conf) for f in stab for d in f)


def test_conf_agg_modes_max_and_min():
    seq = _seq_with_flicker()
    assert aggregate_sequence(seq, min_hits=2, conf_agg="max")[0][0]["conf"] == pytest.approx(0.9)
    assert aggregate_sequence(seq, min_hits=2, conf_agg="min")[0][0]["conf"] == pytest.approx(0.7)
    with pytest.raises(ValueError):
        aggregate_sequence(seq, min_hits=2, conf_agg="median")


# --- (b) a one-frame flicker is dropped by min_hits=2 -------------------------------------------

def test_flicker_is_dropped_and_lesion_replaces_it():
    stab = aggregate_sequence(_seq_with_flicker(), iou_thr=0.3, min_hits=2)
    # The flicker lived at (0.1, 0.1); it must be gone. Frame 1 now holds the interpolated lesion.
    assert stab[1][0]["box"][0] == pytest.approx(0.51)
    assert not any(abs(d["box"][0] - 0.1) < 1e-6 for f in stab for d in f)


def test_min_hits_one_keeps_flicker_precision_knob():
    # Lower the knob to 1: no persistence demanded -> the flicker survives (recall up, precision down).
    stab = aggregate_sequence(_seq_with_flicker(), iou_thr=0.3, min_hits=1)
    assert any(abs(d["box"][0] - 0.1) < 1e-6 for f in stab for d in f)


def test_min_hits_three_drops_short_lesion():
    # A lesion seen on only 2 frames is rejected when persistence is raised to 3 (precision up).
    frames = [[_lesion(0.50)], [_lesion(0.51)], []]
    assert aggregate_sequence(frames, min_hits=3) == [[], [], []]
    assert [len(f) for f in aggregate_sequence(frames, min_hits=2)] == [1, 1, 0]


# --- shape / purity contract --------------------------------------------------------------------

def test_output_shape_matches_input_and_empty_is_empty():
    assert aggregate_sequence([]) == []
    assert aggregate_sequence([[], [], []]) == [[], [], []]
    frames = _seq_with_flicker()
    stab = aggregate_sequence(frames, min_hits=2)
    assert len(stab) == len(frames) and all(isinstance(f, list) for f in stab)
