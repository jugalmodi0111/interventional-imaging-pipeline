"""Stage-0 preprocessing: CLAHE+unsharp batch walk over a directory tree."""
import os
import cv2
import numpy as np
from src.data_prep.preprocess import process_dir


def _write_gray(path, h=64, w=64):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = (np.linspace(0, 255, h * w).reshape(h, w)).astype(np.uint8)
    cv2.imwrite(path, img)


def test_process_dir_walks_tree_and_writes_grayscale_png(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_gray(str(src / "a.png"))
    _write_gray(str(src / "sub" / "b.pgm"))

    n = process_dir(str(src), str(dst))

    assert n == 2
    out_a = cv2.imread(str(dst / "a.png"), cv2.IMREAD_UNCHANGED)
    assert out_a is not None and out_a.ndim == 2          # grayscale, single channel
    # .pgm input is normalized to .png output, mirroring the sub/ structure
    assert (dst / "sub" / "b.png").exists()


def test_process_dir_resizes_when_size_given(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_gray(str(src / "a.png"), h=100, w=120)

    process_dir(str(src), str(dst), size=512)

    out = cv2.imread(str(dst / "a.png"), cv2.IMREAD_UNCHANGED)
    assert out.shape == (512, 512)


def test_process_dir_skips_non_images(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_gray(str(src / "a.png"))
    (src / "notes.txt").write_text("not an image")

    n = process_dir(str(src), str(dst))

    assert n == 1
    assert not (dst / "notes.png").exists()
    assert not (dst / "notes.txt").exists()
