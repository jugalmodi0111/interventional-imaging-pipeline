"""CoreML export smoke-test helper: `_check_coreml_output`.

Torch/ultralytics/coremltools-free by design: `_check_coreml_output` is a pure, stdlib-only
(os) sanity check on an already-exported path, so it's tested directly with tmp_path
`.mlpackage` dirs / `.mlmodel` files -- no real export() call, no heavy deps, no GPU.
`export()`/`smoketest()` themselves lazy-import ultralytics and are exercised manually on
macOS with real weights, not here.
"""
import os

import src.export.yolo_to_coreml as M


# --- import guard: module must import with no coremltools/ultralytics/torch installed -----

def test_module_imports_without_heavy_deps():
    import src.export.yolo_to_coreml  # noqa: F401 -- re-import is a no-op; just proves it loaded


# --- _check_coreml_output: missing / empty path -> False ----------------------------------

def test_missing_path_is_not_ok():
    ok, msg = M._check_coreml_output("/definitely/does/not/exist.mlpackage")
    assert ok is False


def test_empty_path_is_not_ok():
    ok, msg = M._check_coreml_output("")
    assert ok is False


def test_none_path_is_not_ok():
    ok, msg = M._check_coreml_output(None)
    assert ok is False


# --- _check_coreml_output: a real .mlpackage directory -> True + size/type in msg ----------

def test_mlpackage_dir_with_files_is_ok(tmp_path):
    pkg = tmp_path / "model.mlpackage"
    pkg.mkdir()
    (pkg / "weights.bin").write_bytes(b"x" * (2 * 1024 * 1024))   # 2 MB, well clear of 0.00 rounding
    (pkg / "manifest.json").write_bytes(b"{}" * 10)

    ok, msg = M._check_coreml_output(str(pkg))

    assert ok is True
    assert "mlpackage" in msg
    assert "2." in msg and "MB" in msg   # nonzero size reported, not just size-agnostic


# --- _check_coreml_output: a real .mlmodel file -> True ------------------------------------

def test_mlmodel_file_with_bytes_is_ok(tmp_path):
    mdl = tmp_path / "model.mlmodel"
    mdl.write_bytes(b"y" * 2048)

    ok, msg = M._check_coreml_output(str(mdl))

    assert ok is True
    assert "mlmodel" in msg


# --- _check_coreml_output: zero-byte file -> still ok, but msg warns about size-0 ----------

def test_zero_byte_mlmodel_warns_in_msg(tmp_path):
    mdl = tmp_path / "empty.mlmodel"
    mdl.write_bytes(b"")

    ok, msg = M._check_coreml_output(str(mdl))

    assert ok is True
    assert "WARNING" in msg and "0" in msg
