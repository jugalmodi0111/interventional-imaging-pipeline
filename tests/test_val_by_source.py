"""Per-source stenosis val: classify val-set stems by dataset of origin (ARCADE/CADICA/Danilov)
so per-source P/R/mAP can be computed. source_of/_write_lists must stay torch-free (repo
invariant: src/* imports without torch/ultralytics/cv2) -- ultralytics is only imported lazily
inside main(), which these tests never call.
"""
import os

from src.eval.val_by_source import source_of, _write_lists


# --- source_of: stem -> origin dataset --------------------------------------------------------

def test_source_of_cadica_stems():
    assert source_of("p12_v3_00045") == "cadica"
    assert source_of("p1_v1_0") == "cadica"


def test_source_of_danilov_stem():
    assert source_of("14_002_5_0016") == "danilov"


def test_source_of_arcade_stems():
    assert source_of("train_5") == "arcade"
    assert source_of("5") == "arcade"
    assert source_of("val_120") == "arcade"


# --- _write_lists: bucket val images by source, write val_<src>.txt lists ---------------------

def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def test_write_lists_buckets_by_source_and_writes_txt_files(tmp_path):
    proc = str(tmp_path)
    val_dir = os.path.join(proc, "images", "val")
    stems = {
        "cadica": ["p12_v3_00045", "p1_v1_0"],
        "danilov": ["14_002_5_0016"],
        "arcade": ["train_5", "5", "val_120"],
    }
    for src, names in stems.items():
        for name in names:
            _touch(os.path.join(val_dir, name + ".png"))

    lists = _write_lists(proc)

    assert set(lists) == {"cadica", "danilov", "arcade"}
    for src, names in stems.items():
        lp, n = lists[src]
        assert n == len(names)
        assert lp == os.path.join(proc, f"val_{src}.txt")
        written = open(lp).read().splitlines()
        assert len(written) == len(names)
        # every written path is absolute and points at one of this source's images
        for line in written:
            assert os.path.isabs(line)
            assert os.path.splitext(os.path.basename(line))[0] in names


def test_write_lists_empty_val_dir_returns_no_buckets(tmp_path):
    proc = str(tmp_path)
    os.makedirs(os.path.join(proc, "images", "val"), exist_ok=True)
    assert _write_lists(proc) == {}


# --- guard: importing the module must not require ultralytics ---------------------------------

def test_module_importable_without_ultralytics():
    import src.eval.val_by_source  # noqa: F401
