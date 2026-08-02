"""TDD for src.data_prep.build_router_manifest — dataset-root -> modality-label manifest builder.

The modality router (B2 train_router.py) trains on a (path,label) CSV built by scanning the
per-modality dataset roots already on disk (data/raw/<name>/ per the repo-wide convention; see
configs/coronary_seg.yaml, stenosis_yolo.yaml, catheter_track.yaml, cerebral_dsa_temporal.yaml for
actual dataset names). Labels come from substring rules on the path (dataset name -> modality), so
adding a modality = adding one rule; a path that matches no rule is dropped, never mislabeled.

`data/raw/` does not exist on this laptop (GPU-side data) -- every IO test below builds its own
temp directory tree instead of touching real data. `label_for_path` is pure and torch-free.
"""
import csv
import os

from src.data_prep.build_router_manifest import build_manifest, label_for_path

RULES = {
    "coronary_angiography": ["arcade", "danilov", "cadica", "dca1"],
    "cerebral_dsa": ["dsa", "cerebral"],
    "other_xray": ["chestxray", "mura"],
}


# ---- label_for_path (pure) --------------------------------------------------
def test_maps_coronary_source_dirs():
    assert label_for_path("/data/raw/cadica/selectedVideos/p1/v1/input/x.png", RULES) == "coronary_angiography"
    assert label_for_path("/kaggle/input/arcade/stenosis/train/img/9.png", RULES) == "coronary_angiography"


def test_maps_cerebral_and_other():
    assert label_for_path("/data/raw/cerebral_dsa/seq3/f10.png", RULES) == "cerebral_dsa"
    assert label_for_path("/data/raw/chestxray14/000001.png", RULES) == "other_xray"


def test_unmatched_path_returns_none():
    assert label_for_path("/data/raw/mystery/x.png", RULES) is None


def test_label_for_path_is_case_insensitive():
    assert label_for_path("/DATA/RAW/CADICA/x.PNG", RULES) == "coronary_angiography"


def test_label_for_path_first_matching_rule_wins_on_dict_order():
    # a path matching more than one label's substrings resolves to whichever label is FIRST in the
    # rules dict (insertion order) -- deterministic, and lets callers order rules by priority.
    rules = {"a": ["shared"], "b": ["shared"]}
    assert label_for_path("/data/raw/shared/x.png", rules) == "a"


def test_label_for_path_empty_rules_returns_none():
    assert label_for_path("/data/raw/cadica/x.png", {}) is None


# ---- build_manifest (IO, uses tmp_path — never touches real data/raw/) -----
def _make_images(root, rel_dir, names):
    d = os.path.join(root, rel_dir)
    os.makedirs(d, exist_ok=True)
    for n in names:
        open(os.path.join(d, n), "wb").write(b"\x89PNG\r\n")  # content is irrelevant; only path matters
    return d


def test_build_manifest_writes_matched_rows_only(tmp_path):
    root = str(tmp_path / "cadica")
    _make_images(root, "selectedVideos/p1/v1/input", ["a.png", "b.jpg"])
    other_root = str(tmp_path / "mystery")
    _make_images(other_root, ".", ["c.png"])  # matches no rule -> dropped

    out_csv = str(tmp_path / "out" / "manifest.csv")
    report = build_manifest([root, other_root], RULES, out_csv)

    with open(out_csv, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["path", "label"]
    assert len(rows) == 3  # header + 2 matched images
    assert all(r[1] == "coronary_angiography" for r in rows[1:])
    assert report["counts"] == {"coronary_angiography": 2}
    assert report["unmatched"] == 1
    assert report["rows"] == 2


def test_build_manifest_ignores_non_image_files(tmp_path):
    root = str(tmp_path / "cadica")
    d = _make_images(root, ".", ["a.png"])
    open(os.path.join(d, "readme.txt"), "w").write("not an image")
    open(os.path.join(d, "labels.json"), "w").write("{}")

    out_csv = str(tmp_path / "manifest.csv")
    report = build_manifest([root], RULES, out_csv)
    assert report["rows"] == 1
    assert report["counts"] == {"coronary_angiography": 1}


def test_build_manifest_respects_per_class_cap(tmp_path):
    root = str(tmp_path / "arcade")
    _make_images(root, ".", [f"{i}.png" for i in range(10)])

    out_csv = str(tmp_path / "manifest.csv")
    report = build_manifest([root], RULES, out_csv, per_class_cap=3)
    assert report["counts"] == {"coronary_angiography": 3}
    assert report["rows"] == 3


def test_build_manifest_creates_missing_output_dir(tmp_path):
    root = str(tmp_path / "cadica")
    _make_images(root, ".", ["a.png"])
    out_csv = str(tmp_path / "deep" / "nested" / "manifest.csv")
    build_manifest([root], RULES, out_csv)
    assert os.path.isfile(out_csv)


def test_build_manifest_empty_roots_writes_header_only(tmp_path):
    out_csv = str(tmp_path / "manifest.csv")
    report = build_manifest([], RULES, out_csv)
    with open(out_csv, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [["path", "label"]]
    assert report == {"counts": {}, "unmatched": 0, "rows": 0}


def test_build_manifest_multiple_labels_across_roots(tmp_path):
    cadica_root = str(tmp_path / "cadica")
    _make_images(cadica_root, ".", ["a.png"])
    cerebral_root = str(tmp_path / "cerebral_dsa")
    _make_images(cerebral_root, "seq1", ["f1.png", "f2.png"])

    out_csv = str(tmp_path / "manifest.csv")
    report = build_manifest([cadica_root, cerebral_root], RULES, out_csv)
    assert report["counts"] == {"coronary_angiography": 1, "cerebral_dsa": 2}


# ---- import-safety -----------------------------------------------------------
def test_import_is_dependency_free():
    # Fresh interpreter so torch/ultralytics/etc. loaded by an EARLIER test file (test-order
    # pollution) can't defeat the check -- the property under test is this module's OWN import.
    import subprocess
    import sys
    import textwrap

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = textwrap.dedent("""
        import sys, importlib
        importlib.import_module("src.data_prep.build_router_manifest")
        for mod in ("torch", "ultralytics", "coremltools", "transformers"):
            assert mod not in sys.modules, f"build_router_manifest import pulled in {mod}"
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=repo_root,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
