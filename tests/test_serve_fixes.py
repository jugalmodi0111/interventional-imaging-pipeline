"""Unit tests for the serve-layer metric fixes (track.py + infer.py).

All torch-/ultralytics-/coreml-free: they exercise the PURE helpers extracted for exactly this
reason. Each test pins one of the confirmed metric bugs so a regression re-surfaces loudly.
"""
import numpy as np
import pytest

from src.serve.track import (concurrent_tracks, fragmentation, weighted_fps,
                             suspect_flat_numbering, group_frames_by_clip)
from src.serve.infer import seg_confidence


# --- bug 1: fragmentation counts ASSIGNED track ids, not raw detections --------------

def test_concurrent_tracks_none_is_zero():
    assert concurrent_tracks(None) == 0


def test_concurrent_tracks_counts_only_non_none():
    # 2 tracked boxes + 2 untracked (spurious) detections -> concurrent tracks = 2, not 4.
    assert concurrent_tracks([1.0, None, 7.0, None]) == 2


def test_concurrent_tracks_all_assigned_and_empty():
    assert concurrent_tracks([3.0, 4.0, 5.0]) == 3
    assert concurrent_tracks([]) == 0


def test_fragmentation_basic_and_floor():
    assert fragmentation(5, 2) == 3            # 5 unique ids, 2 ever-simultaneous -> 3 re-acquires
    assert fragmentation(2, 2) == 0            # every track kept one id
    assert fragmentation(1, 3) == 0            # never negative


def test_fragmentation_false_good_is_fixed():
    # One instrument fragmented into 4 ids across a clip; only ever 1 tracked at a time, but each
    # frame also had 3 spurious untracked detections. Old metric used len(boxes)=4 -> max_tracks 4
    # -> frag=0 (false good). concurrent_tracks ignores the untracked ones -> max_concurrent=1.
    per_frame_ids = [[10.0, None, None, None],
                     [11.0, None, None, None],
                     [12.0, None, None, None],
                     [13.0, None, None, None]]
    max_concurrent = max(concurrent_tracks(ids) for ids in per_frame_ids)
    unique_ids = len({int(i) for ids in per_frame_ids for i in ids if i is not None})
    assert max_concurrent == 1
    assert fragmentation(unique_ids, max_concurrent) == 3   # the fragmentation is now visible


# --- bug 3: throughput is frame-weighted, not a mean of per-clip fps -----------------

def test_weighted_fps_is_frame_weighted():
    # A tiny fast clip and a large slow clip. Unweighted mean over-weights the tiny clip.
    rows = [{"frames": 2, "time_s": 0.1},      # 20 fps  (tiny)
            {"frames": 200, "time_s": 20.0}]   # 10 fps  (large, dominates real throughput)
    got = weighted_fps(rows)
    assert got == pytest.approx(202 / 20.1)    # ~10.05 fps, near the large clip
    unweighted_mean = (20.0 + 10.0) / 2        # = 15.0, the biased number we're avoiding
    assert got < unweighted_mean


def test_weighted_fps_falls_back_to_track_s_and_handles_zero_time():
    assert weighted_fps([{"frames": 30, "track_s": 3.0}]) == pytest.approx(10.0)
    assert weighted_fps([{"frames": 0, "time_s": 0.0}]) == 0.0
    assert weighted_fps([]) == 0.0


# --- bug 4: flat/global frame numbering must not silently collapse into one clip ------

def test_suspect_flat_numbering_flags_empty_prefix_many_frames():
    clips = {"": [f"{i:04d}.png" for i in range(300)]}   # 0000.png..0299.png, no prefix
    assert suspect_flat_numbering(clips) is True


def test_suspect_flat_numbering_ok_for_prefixed_or_small():
    assert suspect_flat_numbering({"clipA_": ["x"] * 300}) is False    # has a real prefix
    assert suspect_flat_numbering({"": ["1.png", "2.png"]}) is False   # too few frames
    assert suspect_flat_numbering({"a_": ["x"], "b_": ["y"]}) is False  # already many clips


def test_group_frames_by_clip_warns_on_flat_numbering(tmp_path, capsys):
    for i in range(60):
        (tmp_path / f"{i:04d}.png").write_bytes(b"")
    clips = group_frames_by_clip(str(tmp_path))
    out = capsys.readouterr().out
    assert list(clips.keys()) == [""] and len(clips[""]) == 60
    assert "WARNING" in out and "manifest" in out.lower()


def test_group_frames_by_clip_quiet_for_real_clip_prefixes(tmp_path, capsys):
    for clip in ("JFQ_j01_img-00000-", "JFQ_j02_img-00000-"):
        for i in range(30):
            (tmp_path / f"{clip}{i:04d}.png").write_bytes(b"")
    clips = group_frames_by_clip(str(tmp_path))
    out = capsys.readouterr().out
    assert len(clips) == 2
    assert "WARNING" not in out


# --- bug 5: two-sided segmentation confidence consistent with coverage_risk ----------

def test_seg_confidence_uniform_half_defers():
    prob = np.full((32, 32), 0.5)
    c = seg_confidence(prob)
    assert c == pytest.approx(0.5)
    assert c < 0.55            # defer_below default -> DEFERS on max uncertainty


def test_seg_confidence_confident_map_keeps():
    prob = np.where(np.arange(100) < 50, 0.98, 0.02).astype(np.float64)  # crisp fg AND bg
    c = seg_confidence(prob)
    assert c == pytest.approx(0.98)
    assert c >= 0.55           # confident -> KEEPS (does not defer)


def test_seg_confidence_is_two_sided_counts_confident_background():
    # A tiny confident foreground amid a confident background. The old one-sided score averaged
    # only the fg pixels; the two-sided score credits the confident background too.
    prob = np.full(1000, 0.01)
    prob[:5] = 0.99            # 5 confident fg pixels, 995 confident bg pixels
    c = seg_confidence(prob)
    assert c == pytest.approx(0.99, abs=1e-9)   # ~1.0: whole map is confident, so KEEP
    assert 0.5 <= c <= 1.0


def test_seg_confidence_in_unit_range_for_random_maps():
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = seg_confidence(rng.uniform(size=(16, 16)))
        assert 0.5 <= c <= 1.0
