"""coreml_validate HARD-GATE enforcement tests.

The gate's whole job is to be able to FAIL a build. Before this file existed, `main()`
computed a correct verdict and the `__main__` block threw it away (`main(ap.parse_args())`),
so the process exited 0 whether it printed PASS or FAIL and no Makefile/CI step could catch it.

These tests pin the verdict -> process-exit-status wiring. They deliberately do NOT re-test
what is measured (dice/cldice have their own tests) — only that the measurement is enforced.
Model loading and prediction are monkeypatched so no .mlpackage / .pt artifact is needed;
the metric arithmetic in between is the real thing.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("skimage")

from src.export import coreml_validate as cv

REPO = str(Path(__file__).resolve().parents[1])
SIZE = 64


def _write_pair(d):
    """One paired image/mask: a 3px-wide vertical vessel. Image == mask, so a fake
    predictor can recover the exact GT with `x > 0.5` (see _pred_perfect)."""
    img, msk = d / "img", d / "msk"
    img.mkdir(); msk.mkdir()
    m = np.zeros((SIZE, SIZE), np.uint8)
    m[8:56, 30:33] = 255
    cv2.imwrite(str(img / "val_0.png"), m)
    cv2.imwrite(str(msk / "val_0.png"), m)
    return str(img), str(msk)


def _pred_perfect(model, x):
    return x > 0.5


def _pred_fragmented(model, x):
    """Same pixels minus every other row: Dice stays high, the centreline shatters ->
    a large clDice drop. This is exactly the failure mode the gate exists to catch."""
    p = x > 0.5
    p[::2, :] = False
    return p


def _argv(images, masks, gate="0.03"):
    return ["--coreml", "unused.mlpackage", "--weights", "unused.pt",
            "--images", images, "--masks", masks,
            "--size", str(SIZE), "--limit", "5", "--gate", gate]


@pytest.fixture
def stub_models(monkeypatch):
    monkeypatch.setattr(cv, "_load_coreml", lambda path: object())
    monkeypatch.setattr(cv, "_load_torch", lambda w, base, depth: object())
    monkeypatch.setattr(cv, "_torch_pred", _pred_perfect)


# --- verdict reaches the exit status -------------------------------------------------------

def test_cli_exits_zero_when_gate_passes(tmp_path, monkeypatch, stub_models, capsys):
    images, masks = _write_pair(tmp_path)
    monkeypatch.setattr(cv, "_coreml_pred", _pred_perfect)
    assert cv.cli(_argv(images, masks)) == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_exits_nonzero_when_gate_fails(tmp_path, monkeypatch, stub_models, capsys):
    images, masks = _write_pair(tmp_path)
    monkeypatch.setattr(cv, "_coreml_pred", _pred_fragmented)
    assert cv.cli(_argv(images, masks)) != 0
    assert "FAIL" in capsys.readouterr().out


def test_n_is_reported(tmp_path, monkeypatch, stub_models, capsys):
    images, masks = _write_pair(tmp_path)
    monkeypatch.setattr(cv, "_coreml_pred", _pred_perfect)
    cv.cli(_argv(images, masks))
    assert "n=1" in capsys.readouterr().out


# --- main() keeps returning the boolean verdict: src/train/train_seg.py:222 prints it -------

def test_main_still_returns_bool_verdict(tmp_path, monkeypatch, stub_models):
    images, masks = _write_pair(tmp_path)
    monkeypatch.setattr(cv, "_coreml_pred", _pred_fragmented)
    a = cv._parser().parse_args(_argv(images, masks))
    verdict = cv.main(a)
    assert verdict is False and isinstance(verdict, bool)


# --- the real process exit status, not just the return value --------------------------------

_DRIVER = """
import sys
sys.path.insert(0, {repo!r})
from src.export import coreml_validate as cv
cv._load_coreml = lambda path: object()
cv._load_torch = lambda w, base, depth: object()
cv._torch_pred = lambda m, x: x > 0.5
def _broken(m, x):
    p = x > 0.5
    p[::2, :] = False
    return p
cv._coreml_pred = (lambda m, x: x > 0.5) if {passing!r} else _broken
sys.exit(cv.cli(sys.argv[1:]))
"""


def _run_driver(tmp_path, passing, images, masks):
    drv = tmp_path / f"drv_{passing}.py"
    drv.write_text(_DRIVER.format(repo=REPO, passing=passing))
    return subprocess.run([sys.executable, str(drv), *_argv(images, masks)],
                          cwd=REPO, capture_output=True, text=True)


def test_process_exit_status_is_nonzero_on_fail(tmp_path):
    images, masks = _write_pair(tmp_path)
    r = _run_driver(tmp_path, False, images, masks)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "FAIL" in r.stdout


def test_process_exit_status_is_zero_on_pass(tmp_path):
    images, masks = _write_pair(tmp_path)
    r = _run_driver(tmp_path, True, images, masks)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout
