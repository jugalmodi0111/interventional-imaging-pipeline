"""PngCalibrationReader unit tests — reader logic only (no quantize_static: too slow for CI).

Skip-marked when onnxruntime.quantization / cv2 are unavailable: the reader is duck-typed
(ORT accepts anything with a callable get_next via CalibrationDataReader.__subclasshook__),
but it only exists to feed quantize_static, so the tests gate on the same deps.
"""
import numpy as np
import pytest

ortq = pytest.importorskip("onnxruntime.quantization")
cv2 = pytest.importorskip("cv2")

from src.export.quantize_int8 import PngCalibrationReader


def _write_pngs(d, values, hw=(64, 96)):          # non-square: resize must actually run
    for i, v in enumerate(values):
        cv2.imwrite(str(d / f"val_{i}.png"), np.full(hw, v, np.uint8))


# --- batches: shape [1,1,size,size], float32, /255 normalization, sorted order --------------

def test_batches_shape_dtype_and_normalization(tmp_path):
    _write_pngs(tmp_path, [0, 51, 255])
    r = PngCalibrationReader(str(tmp_path), input_name="input", size=32, limit=10)
    batches = list(iter(r.get_next, None))
    assert len(batches) == 3
    for b in batches:
        assert set(b) == {"input"}
        assert b["input"].shape == (1, 1, 32, 32)
        assert b["input"].dtype == np.float32
    assert np.allclose(batches[0]["input"], 0.0)          # val_0 = 0
    assert np.allclose(batches[1]["input"], 51 / 255.0)   # val_1 = 51
    assert np.allclose(batches[2]["input"], 1.0)          # val_2 = 255


# --- cap: fewer on disk than requested -> use what exists, and say so -----------------------

def test_cap_at_what_exists_on_disk(tmp_path, capsys):
    _write_pngs(tmp_path, [10, 20])
    r = PngCalibrationReader(str(tmp_path), limit=200)
    assert len(r.paths) == 2
    assert "using 2 of 200 requested" in capsys.readouterr().out
    assert sum(1 for _ in iter(r.get_next, None)) == 2


# --- limit: more on disk than requested -> truncate, stay exhausted after -------------------

def test_limit_truncates_when_more_on_disk(tmp_path):
    _write_pngs(tmp_path, [10, 20, 30, 40])
    r = PngCalibrationReader(str(tmp_path), limit=2)
    assert sum(1 for _ in iter(r.get_next, None)) == 2
    assert r.get_next() is None


# --- rewind restarts the stream --------------------------------------------------------------

def test_rewind_restarts_stream(tmp_path):
    _write_pngs(tmp_path, [10, 20])
    r = PngCalibrationReader(str(tmp_path), limit=10)
    assert sum(1 for _ in iter(r.get_next, None)) == 2
    r.rewind()
    assert sum(1 for _ in iter(r.get_next, None)) == 2


# --- contract: ORT's CalibrationDataReader accepts the duck-typed reader --------------------

def test_satisfies_ort_calibration_reader_contract(tmp_path):
    _write_pngs(tmp_path, [10])
    assert isinstance(PngCalibrationReader(str(tmp_path)), ortq.CalibrationDataReader)


# --- empty calib dir fails loudly, not with a silent 0-image calibration --------------------

def test_empty_dir_asserts(tmp_path):
    with pytest.raises(AssertionError):
        PngCalibrationReader(str(tmp_path))
