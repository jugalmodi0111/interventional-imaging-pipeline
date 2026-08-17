"""TDD for src.data_prep.harmonize — box-size harmonization (P2.1).

The three merged sources box stenosis at very different sizes (annotation-QA: median box area
arcade 0.0108 / cadica 0.0058 / danilov 0.0029; danilov tiny_frac 0.36). Inconsistent target sizes
cap the IoU-sensitive metric (mAP50 0.209 vs mAP50-95 0.080). This clamps every box up to a common
minimum w/h floor so the model sees a consistent minimum target. Pure/torch-free/cv2-free.
"""
from src.data_prep import harmonize as H


def test_clamp_expands_small_box_keeps_center():
    cx, cy, w, h = H.clamp_box_wh(0.5, 0.5, 0.01, 0.01, 0.04)
    assert (w, h) == (0.04, 0.04)
    assert cx == 0.5 and cy == 0.5


def test_clamp_leaves_large_box_untouched():
    assert H.clamp_box_wh(0.5, 0.5, 0.2, 0.15, 0.04) == (0.5, 0.5, 0.2, 0.15)


def test_clamp_shifts_center_to_stay_in_frame():
    # a tiny box at the edge expands and its center shifts inward so the box stays within [0,1]
    cx, cy, w, h = H.clamp_box_wh(0.005, 0.005, 0.01, 0.01, 0.04)
    assert w == 0.04 and h == 0.04
    assert cx == 0.02 and cy == 0.02          # min center = w/2
    assert cx - w / 2 >= -1e-9 and cy - h / 2 >= -1e-9


def test_clamp_only_one_dim():
    cx, cy, w, h = H.clamp_box_wh(0.5, 0.5, 0.2, 0.01, 0.04)
    assert w == 0.2 and h == 0.04


def test_harmonize_label_lines_counts_changes():
    lines = ["0 0.5 0.5 0.01 0.01", "0 0.5 0.5 0.2 0.2", "0 0.4 0.4 0.03 0.5"]
    out, changed = H.harmonize_label_lines(lines, 0.04)
    assert changed == 2                        # first + third had a sub-floor dim
    assert out[1] == "0 0.5 0.5 0.2 0.2"       # untouched line preserved exactly
    assert out[0].startswith("0 0.5 0.5 0.04 0.04")


def test_harmonize_label_lines_preserves_class_and_skips_malformed():
    lines = ["1 0.5 0.5 0.01 0.01", "garbage", "", "2 0.3 0.3 0.02 0.02"]
    out, changed = H.harmonize_label_lines(lines, 0.04)
    assert changed == 2
    assert out[0].startswith("1 ") and out[3].startswith("2 ")
    assert out[1] == "garbage" and out[2] == ""   # malformed/blank passed through unchanged


def test_harmonize_labels_walks_split(tmp_path):
    d = tmp_path / "labels" / "train"
    d.mkdir(parents=True)
    (d / "a.txt").write_text("0 0.5 0.5 0.01 0.01\n0 0.5 0.5 0.3 0.3\n")
    (d / "b.txt").write_text("0 0.5 0.5 0.5 0.5\n")   # nothing to clamp
    rep = H.harmonize_labels(str(tmp_path), 0.04, splits=("train",))
    assert rep["files"] == 2
    assert rep["boxes_clamped"] == 1
    assert "0.04 0.04" in (d / "a.txt").read_text()


def test_min_wh_zero_is_noop(tmp_path):
    d = tmp_path / "labels" / "train"
    d.mkdir(parents=True)
    (d / "a.txt").write_text("0 0.5 0.5 0.001 0.001\n")
    rep = H.harmonize_labels(str(tmp_path), 0.0, splits=("train",))
    assert rep["boxes_clamped"] == 0
    assert "0.001 0.001" in (d / "a.txt").read_text()


# --- B5: train-only harmonization moves the model's box sizes but not the eval target ----------
# Clamping TRAIN only teaches the model to emit boxes >= min_box_wh while val is still scored
# against the ORIGINAL (smaller) boxes. An over-sized prediction loses IoU against a small GT box,
# so the lever can LOWER mAP50 (and mAP50-95 harder) while looking conservative. The warning has to
# fire at the moment someone switches the lever on, and must name the both-vals comparison.

def test_warning_fires_for_train_only_when_enabled():
    msg = H.harmonization_warning(0.04, ("train",))
    assert msg is not None
    low = msg.lower()
    assert "warning" in low
    assert "0.04" in msg                       # names the value that was switched on
    assert "map50-95" in low                   # names the metric that moves hardest
    assert "val" in low and "train" in low


def test_warning_recommends_scoring_both_original_and_harmonized_val():
    msg = H.harmonization_warning(0.04, ["train"])
    low = msg.lower()
    assert "both" in low                       # score the same weights on BOTH vals
    assert "harmonized" in low and "original" in low


def test_no_warning_when_lever_is_off():
    assert H.harmonization_warning(0.0, ("train",)) is None
    assert H.harmonization_warning(0.0, ("train", "val")) is None
    assert H.harmonization_warning(None, ("train",)) is None


def test_no_warning_when_both_splits_are_harmonized():
    # train and val move together -> no train/eval target mismatch -> nothing to warn about
    assert H.harmonization_warning(0.04, ("train", "val")) is None
    assert H.harmonization_warning(0.04, ["val", "train"]) is None


def test_warning_also_fires_for_val_only_which_is_the_mirror_mismatch():
    msg = H.harmonization_warning(0.04, ("val",))
    assert msg is not None and "val" in msg.lower()


def test_warning_is_pure_and_takes_no_paths(tmp_path):
    # pure function: no filesystem, no config, callable with a bare number + split list
    before = sorted(p.name for p in tmp_path.iterdir())
    H.harmonization_warning(0.04, ("train",))
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_harmonize_labels_prints_the_warning_when_train_only(tmp_path, capsys):
    d = tmp_path / "labels" / "train"
    d.mkdir(parents=True)
    (d / "a.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    H.harmonize_labels(str(tmp_path), 0.04, splits=("train",))
    out = capsys.readouterr().out
    assert "WARNING" in out and "map50-95" in out.lower()


def test_harmonize_labels_is_quiet_when_both_splits_harmonized(tmp_path, capsys):
    for sp in ("train", "val"):
        d = tmp_path / "labels" / sp
        d.mkdir(parents=True)
        (d / "a.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    H.harmonize_labels(str(tmp_path), 0.04, splits=("train", "val"))
    assert "WARNING" not in capsys.readouterr().out


def test_harmonize_labels_is_quiet_when_disabled(tmp_path, capsys):
    d = tmp_path / "labels" / "train"
    d.mkdir(parents=True)
    (d / "a.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    H.harmonize_labels(str(tmp_path), 0.0, splits=("train",))
    assert "WARNING" not in capsys.readouterr().out


def test_main_warns_when_config_enables_train_only(tmp_path, capsys):
    d = tmp_path / "labels" / "train"
    d.mkdir(parents=True)
    (d / "a.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    H.main({"harmonize": {"min_box_wh": 0.04, "splits": ["train"]}}, proc=str(tmp_path))
    assert "WARNING" in capsys.readouterr().out


def test_main_does_not_warn_on_the_shipped_off_default(tmp_path, capsys):
    H.main({"harmonize": {"min_box_wh": 0.0, "splits": ["train"]}}, proc=str(tmp_path))
    assert "WARNING" not in capsys.readouterr().out


def test_docstrings_state_the_train_eval_mismatch():
    # the module/function docs must not sell train-only as free: they have to say the predicted
    # box sizes move relative to the val target and that IoU-sensitive metrics can DROP.
    for doc in (H.__doc__, H.harmonize_labels.__doc__):
        low = (doc or "").lower()
        assert "map50-95" in low
        assert "larger" in low


def test_clamping_math_is_unchanged(tmp_path):
    # pure regression for B5: surfacing the trade-off must not move a single box coordinate.
    cases = [
        ((0.5, 0.5, 0.01, 0.01, 0.04), (0.5, 0.5, 0.04, 0.04)),
        ((0.5, 0.5, 0.2, 0.15, 0.04), (0.5, 0.5, 0.2, 0.15)),
        ((0.005, 0.005, 0.01, 0.01, 0.04), (0.02, 0.02, 0.04, 0.04)),
        ((0.995, 0.995, 0.01, 0.01, 0.04), (0.98, 0.98, 0.04, 0.04)),
        ((0.5, 0.5, 0.2, 0.01, 0.04), (0.5, 0.5, 0.2, 0.04)),
        ((0.3, 0.7, 0.01, 0.01, 1.5), (0.5, 0.5, 1.0, 1.0)),   # floor wider than the frame
    ]
    for args, want in cases:
        assert H.clamp_box_wh(*args) == want, args

    lines = ["0 0.5 0.5 0.01 0.01", "0 0.5 0.5 0.2 0.2", "1 0.4 0.4 0.03 0.5", "junk", ""]
    out, changed = H.harmonize_label_lines(lines, 0.04)
    assert changed == 2
    assert out == ["0 0.5 0.5 0.04 0.04", "0 0.5 0.5 0.2 0.2",
                   "1 0.4 0.4 0.04 0.5", "junk", ""]

    d = tmp_path / "labels" / "train"
    d.mkdir(parents=True)
    (d / "a.txt").write_text("0 0.5 0.5 0.01 0.01\n0 0.5 0.5 0.3 0.3\n")
    (d / "b.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    rep = H.harmonize_labels(str(tmp_path), 0.04, splits=("train",))
    assert rep == {"files": 2, "boxes_clamped": 1}
    assert (d / "a.txt").read_text() == "0 0.5 0.5 0.04 0.04\n0 0.5 0.5 0.3 0.3\n"
    assert (d / "b.txt").read_text() == "0 0.5 0.5 0.5 0.5\n"


def test_import_is_dependency_free():
    # Fresh interpreter so torch/cv2 loaded by an EARLIER test file (test-order pollution) can't
    # defeat the check — the property is that harmonize's OWN import pulls in neither.
    import os, subprocess, sys, textwrap
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = textwrap.dedent("""
        import sys, importlib
        importlib.import_module("src.data_prep.harmonize")
        for mod in ("cv2", "torch"):
            assert mod not in sys.modules, f"harmonize import pulled in {mod}"
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=repo_root,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
