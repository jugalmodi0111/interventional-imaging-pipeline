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


# ===========================================================================================
# Output-path collision: static and dynamic PTQ are DIFFERENT artifacts with different
# accuracy, and both used to default to "<model>.int8.onnx". A bare
# `python -m src.export.quantize_int8` therefore overwrote the dynamic student.int8.onnx —
# the build docs say "must not ship" and that no clDice gate has ever scored — with a static
# build under the same name, leaving two provenances indistinguishable on disk. Defaults are
# now method-tagged, and an explicit --out is refused when it would clobber the other method's
# artifact (--force overrides).
# ===========================================================================================
import os
from types import SimpleNamespace

from src.export import quantize_int8 as qz


@pytest.fixture
def fake_ort(monkeypatch, tmp_path):
    """Stub the actual quantizers (real ones are too slow for CI, per this file's header):
    they just write a marker file naming the method that produced it."""
    import onnxruntime as ort
    import onnxruntime.quantization as q
    import onnxruntime.quantization.shape_inference as si

    class _Sess:
        def __init__(self, *a, **k): pass
        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 1, 64, 64])]

    monkeypatch.setattr(ort, "InferenceSession", _Sess)
    monkeypatch.setattr(si, "quant_pre_process",
                        lambda src, dst, **k: open(dst, "w").write("pre"))
    monkeypatch.setattr(q, "quantize_static",
                        lambda src, out, reader, **k: open(out, "w").write("static"))
    monkeypatch.setattr(q, "quantize_dynamic",
                        lambda src, out, **k: open(out, "w").write("dynamic"))


@pytest.fixture
def model_and_calib(tmp_path):
    model = tmp_path / "student.onnx"
    model.write_text("fp32")
    calib = tmp_path / "calib"
    calib.mkdir()
    _write_pngs(calib, [10, 20])
    return str(model), str(calib)


# --- distinct default names per method ------------------------------------------------------

def test_default_out_names_are_distinct_per_method():
    assert qz.default_out("a/student.onnx", "static_ptq") == "a/student.int8.static.onnx"
    assert qz.default_out("a/student.onnx", "dynamic") == "a/student.int8.dynamic.onnx"
    assert qz.default_out("a/student.onnx", "static_ptq") != qz.default_out("a/student.onnx", "dynamic")


def test_static_default_out_is_method_tagged(fake_ort, model_and_calib, capsys):
    model, calib = model_and_calib
    out = qz.quantize(model, calib_dir=calib, config=None)
    assert out.endswith("student.int8.static.onnx")
    assert open(out).read() == "static"


def test_dynamic_default_out_is_method_tagged(fake_ort, model_and_calib):
    model, calib = model_and_calib
    out = qz.quantize(model, calib_dir=calib, config=None, dynamic=True)
    assert out.endswith("student.int8.dynamic.onnx")
    assert open(out).read() == "dynamic"


# --- the actual regression: a static build must not silently land on the dynamic artifact ----

def test_static_run_does_not_overwrite_legacy_dynamic_artifact(fake_ort, model_and_calib):
    model, calib = model_and_calib
    legacy = model.replace(".onnx", ".int8.onnx")          # the shipped-but-unquantified name
    open(legacy, "w").write("legacy-dynamic")
    out = qz.quantize(model, calib_dir=calib, config=None)
    assert out != legacy
    assert open(legacy).read() == "legacy-dynamic"         # untouched


def test_two_methods_coexist_on_disk(fake_ort, model_and_calib):
    model, calib = model_and_calib
    s = qz.quantize(model, calib_dir=calib, config=None)
    d = qz.quantize(model, calib_dir=calib, config=None, dynamic=True)
    assert os.path.exists(s) and os.path.exists(d)
    assert open(s).read() == "static" and open(d).read() == "dynamic"


def test_rerunning_same_method_overwrites_its_own_artifact(fake_ort, model_and_calib):
    model, calib = model_and_calib
    first = qz.quantize(model, calib_dir=calib, config=None)
    assert qz.quantize(model, calib_dir=calib, config=None) == first   # no refusal on re-run


# --- explicit --out that would clobber the other method's artifact --------------------------

def test_refuses_explicit_out_that_clobbers_other_method(fake_ort, model_and_calib):
    model, calib = model_and_calib
    victim = model.replace(".onnx", ".int8.dynamic.onnx")
    open(victim, "w").write("dynamic")
    with pytest.raises(FileExistsError, match="static"):
        qz.quantize(model, out=victim, calib_dir=calib, config=None)
    assert open(victim).read() == "dynamic"


def test_refuses_explicit_out_onto_untagged_existing_artifact(fake_ort, model_and_calib):
    model, calib = model_and_calib
    victim = model.replace(".onnx", ".int8.onnx")
    open(victim, "w").write("legacy-dynamic")
    with pytest.raises(FileExistsError):
        qz.quantize(model, out=victim, calib_dir=calib, config=None)
    assert open(victim).read() == "legacy-dynamic"


def test_force_allows_deliberate_overwrite(fake_ort, model_and_calib):
    model, calib = model_and_calib
    victim = model.replace(".onnx", ".int8.onnx")
    open(victim, "w").write("legacy-dynamic")
    assert qz.quantize(model, out=victim, calib_dir=calib, config=None, force=True) == victim
    assert open(victim).read() == "static"


def test_explicit_out_to_a_new_path_is_honoured(fake_ort, model_and_calib, tmp_path):
    model, calib = model_and_calib
    dest = str(tmp_path / "custom.onnx")
    assert qz.quantize(model, out=dest, calib_dir=calib, config=None) == dest
    assert open(dest).read() == "static"


def test_unknown_method_in_config_still_raises(fake_ort, model_and_calib, tmp_path):
    model, calib = model_and_calib
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("int8: {method: qat_magic}\n")
    with pytest.raises(ValueError, match="unknown int8 method"):
        qz.quantize(model, calib_dir=calib, config=str(cfg))
