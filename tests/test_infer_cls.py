"""Hosted torch inference for Model One. Trains nothing; loads Task 4's head.pt and mirrors B3:
a calibrated probability inside the defer band NEVER becomes a confident call."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.frozen_backbone import FrozenBackboneClassifier
from src.serve.infer_cls import ClsModel


def _ckpt(tmp_path, band=(0.3, 0.6), thr=0.5):
    m = FrozenBackboneClassifier("test-tiny", imgsz=32)
    p = tmp_path / "head.pt"
    torch.save({"backbone": "test-tiny", "imgsz": 32, "head_state": m.head.state_dict(),
                "temperature": 1.0, "threshold": thr, "defer_band": list(band)}, p)
    return p


def test_returns_contract_keys_and_types(tmp_path):
    model = ClsModel(_ckpt(tmp_path))
    out = model(np.zeros((64, 64), dtype=np.uint8))
    assert set(out) >= {"prob", "confidence", "deferred", "reason", "threshold"}
    assert 0.0 <= out["prob"] <= 1.0 and isinstance(out["deferred"], bool)


def test_prob_inside_defer_band_defers(tmp_path):
    model = ClsModel(_ckpt(tmp_path, band=(0.0, 1.0)))     # band swallows everything
    out = model(np.zeros((64, 64), dtype=np.uint8))
    assert out["deferred"] is True and out["reason"] == "defer-band"


def test_prob_outside_defer_band_is_confident(tmp_path):
    model = ClsModel(_ckpt(tmp_path, band=(0.999, 1.0)))   # band swallows nothing
    out = model(np.zeros((64, 64), dtype=np.uint8))
    assert out["deferred"] is False and out["reason"] == "confident"
    assert out["confidence"] == max(out["prob"], 1.0 - out["prob"])


def test_missing_checkpoint_raises_at_construction(tmp_path):
    with pytest.raises(Exception):
        ClsModel(tmp_path / "absent.pt")


def test_module_import_does_not_pull_torch_at_module_scope():
    """Repo convention (src/serve/*): heavy deps live inside functions so the FastAPI app and the
    test collector import cheaply. Asserted structurally, in a fresh interpreter."""
    import subprocess
    import sys
    rc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import src.serve.infer_cls; "
         "assert 'torch' not in sys.modules, 'infer_cls imported torch at module scope'"],
        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
