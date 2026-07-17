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
