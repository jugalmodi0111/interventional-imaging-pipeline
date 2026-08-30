"""onnx_int8_validate HARD-GATE tests.

configs/edge_export.yaml has always declared `gate: {cldice_drop_max: 0.03}` for the ONNX
INT8 path, but nothing read the key: there was no ONNX validator at all, so the "INT8 export
gate PASSED" claim rested on numbers no script could reproduce or fail on.

These tests pin three things: the gate passes when INT8 tracks fp32, it FAILS (non-zero exit)
when INT8 shatters the centreline, and the number it enforces is the CONFIGURED one — the
same broken run flips to PASS under a config that permits a bigger drop, so 0.03 cannot be
silently hardcoded. Sessions/predictions are monkeypatched; the metric arithmetic is real.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("skimage")
yaml = pytest.importorskip("yaml")

from src.export import onnx_int8_validate as ov

REPO = str(Path(__file__).resolve().parents[1])
SIZE = 64


def _write_pair(d):
    img, msk = d / "img", d / "msk"
    img.mkdir(); msk.mkdir()
    m = np.zeros((SIZE, SIZE), np.uint8)
    m[8:56, 30:33] = 255
    cv2.imwrite(str(img / "val_0.png"), m)
    cv2.imwrite(str(msk / "val_0.png"), m)
    return str(img), str(msk)


def _write_config(d, gate=None):
    cfg = {"onnx": {"opset": 17}, "int8": {"method": "static_ptq"}}
    if gate is not None:
        cfg["gate"] = {"cldice_drop_max": gate}
    p = d / "edge_export.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def _pred_perfect(sess, x):
    return x > 0.5


def _pred_fragmented(sess, x):
    """Dice barely moves, the centreline shatters -> ~0.33 clDice drop."""
    p = x > 0.5
    p[::2, :] = False
    return p


def _argv(images, masks, config, gate=None):
    a = ["--fp32", "unused.onnx", "--int8", "unused.int8.static.onnx",
         "--images", images, "--masks", masks,
         "--size", str(SIZE), "--limit", "5", "--config", config]
    return a + (["--gate", str(gate)] if gate is not None else [])


@pytest.fixture
def stub_sessions(monkeypatch):
    monkeypatch.setattr(ov, "_session", lambda path: object())
    monkeypatch.setattr(ov, "_fp32_pred", _pred_perfect)


# --- gate passes / fails -------------------------------------------------------------------

def test_gate_passes_when_int8_tracks_fp32(tmp_path, monkeypatch, stub_sessions, capsys):
    images, masks = _write_pair(tmp_path)
    monkeypatch.setattr(ov, "_int8_pred", _pred_perfect)
    assert ov.cli(_argv(images, masks, _write_config(tmp_path, 0.03))) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "n=1" in out


def test_gate_fails_when_int8_breaks_connectivity(tmp_path, monkeypatch, stub_sessions, capsys):
    images, masks = _write_pair(tmp_path)
    monkeypatch.setattr(ov, "_int8_pred", _pred_fragmented)
    assert ov.cli(_argv(images, masks, _write_config(tmp_path, 0.03))) != 0
    assert "FAIL" in capsys.readouterr().out


# --- the CONFIGURED value is what is enforced, not a hardcoded 0.03 -------------------------

def test_configured_gate_value_is_what_is_enforced(tmp_path, monkeypatch, stub_sessions):
    """Identical broken run, two configs: strict 0.03 FAILS, permissive 0.5 PASSES."""
    images, masks = _write_pair(tmp_path)
    monkeypatch.setattr(ov, "_int8_pred", _pred_fragmented)
    strict_dir, permissive_dir = tmp_path / "strict", tmp_path / "permissive"
    strict_dir.mkdir(); permissive_dir.mkdir()
    assert ov.cli(_argv(images, masks, _write_config(strict_dir, 0.03))) != 0
    assert ov.cli(_argv(images, masks, _write_config(permissive_dir, 0.5))) == 0


def test_repo_config_gate_key_is_read(tmp_path):
    """The key configs/edge_export.yaml has always declared is now actually consumed."""
    assert ov.gate_from_config(str(Path(REPO) / "configs/edge_export.yaml")) == 0.03


def test_missing_gate_key_refuses_rather_than_passing(tmp_path):
    with pytest.raises(AssertionError, match="cldice_drop_max"):
        ov.gate_from_config(_write_config(tmp_path, gate=None))


def test_missing_config_file_refuses(tmp_path):
    with pytest.raises(AssertionError):
        ov.gate_from_config(str(tmp_path / "nope.yaml"))


def test_explicit_gate_flag_overrides_config(tmp_path, monkeypatch, stub_sessions):
    images, masks = _write_pair(tmp_path)
    monkeypatch.setattr(ov, "_int8_pred", _pred_fragmented)
    cfg = _write_config(tmp_path, 0.03)                       # config would FAIL this run
    assert ov.cli(_argv(images, masks, cfg, gate=0.9)) == 0    # explicit override PASSES


# --- mask agreement is reported (the "0.9996" in PROJECT_TRACKER had no script behind it) ---

def test_reports_mask_agreement(tmp_path, monkeypatch, stub_sessions, capsys):
    images, masks = _write_pair(tmp_path)
    monkeypatch.setattr(ov, "_int8_pred", _pred_perfect)
    ov.cli(_argv(images, masks, _write_config(tmp_path, 0.03)))
    assert "agreement 1.0000" in capsys.readouterr().out


# --- real process exit status ----------------------------------------------------------------

_DRIVER = """
import sys
sys.path.insert(0, {repo!r})
from src.export import onnx_int8_validate as ov
ov._session = lambda path: object()
ov._fp32_pred = lambda s, x: x > 0.5
def _broken(s, x):
    p = x > 0.5
    p[::2, :] = False
    return p
ov._int8_pred = (lambda s, x: x > 0.5) if {passing!r} else _broken
sys.exit(ov.cli(sys.argv[1:]))
"""


def _run_driver(tmp_path, passing, argv):
    drv = tmp_path / f"drv_{passing}.py"
    drv.write_text(_DRIVER.format(repo=REPO, passing=passing))
    return subprocess.run([sys.executable, str(drv), *argv], cwd=REPO,
                          capture_output=True, text=True)


def test_process_exit_status_is_nonzero_on_fail(tmp_path):
    images, masks = _write_pair(tmp_path)
    r = _run_driver(tmp_path, False, _argv(images, masks, _write_config(tmp_path, 0.03)))
    assert r.returncode != 0, r.stdout + r.stderr
    assert "FAIL" in r.stdout


def test_process_exit_status_is_zero_on_pass(tmp_path):
    images, masks = _write_pair(tmp_path)
    r = _run_driver(tmp_path, True, _argv(images, masks, _write_config(tmp_path, 0.03)))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout
