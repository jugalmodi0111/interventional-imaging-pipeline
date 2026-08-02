"""Test the modality -> task-model registry."""
import warnings

import pytest

from src.serve.registry import load_registry, resolve, TaskEntry


def _write(tmp_path, floor_ok_line):
    """One coronary_angiography entry with the given floor_ok YAML line (or "" to omit it)."""
    y = tmp_path / "orch.yaml"
    y.write_text(
        "modalities:\n"
        "  coronary_angiography:\n"
        "    task: det\n"
        "    model_path: runs/stenosis/best.pt\n"
        "    display_name: Coronary angiography\n"
        "    finding_label: coronary_stenosis\n"
        "    finding_display: Possible coronary artery stenosis (blockage)\n"
        f"{floor_ok_line}"
    )
    return y


def test_load_and_resolve(tmp_path):
    y = tmp_path / "orch.yaml"
    y.write_text(
        "modalities:\n"
        "  coronary_angiography:\n"
        "    task: det\n"
        "    model_path: runs/stenosis/best.pt\n"
        "    display_name: Coronary angiography\n"
        "    finding_label: coronary_stenosis\n"
        "    finding_display: Possible coronary artery stenosis (blockage)\n"
        "    floor_ok: false\n"
    )
    reg = load_registry(str(y))
    e = resolve(reg, "coronary_angiography")
    assert isinstance(e, TaskEntry) and e.task == "det" and e.floor_ok is False
    assert e.finding_label == "coronary_stenosis"


def test_resolve_unknown_returns_none(tmp_path):
    y = tmp_path / "orch.yaml"
    y.write_text("modalities: {}\n")
    assert resolve(load_registry(str(y)), "cerebral_dsa") is None


def test_missing_floor_ok_defaults_false(tmp_path):
    """No floor_ok key at all -> the documented safe default, silently."""
    y = _write(tmp_path, "")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e = resolve(load_registry(str(y)), "coronary_angiography")
    assert e.floor_ok is False


@pytest.mark.parametrize("bad_value", [
    "    floor_ok: flase\n",           # typo
    "    floor_ok: TBD\n",             # arbitrary non-bool token
    "    floor_ok: \" \"\n",           # quoted whitespace string
    "    floor_ok: \"true\"\n",        # quoted -> string, NOT a real YAML bool
    "    floor_ok: 1\n",               # number (truthy under old bool())
    "    floor_ok: null\n",            # explicit null
    "    floor_ok: [true]\n",          # list
])
def test_malformed_floor_ok_fails_safe_to_false(tmp_path, bad_value):
    """A present-but-malformed floor_ok must NOT yield True (the safety-net bypass bug):
    it must fail closed to False, same as if the key were absent, and must not raise."""
    y = _write(tmp_path, bad_value)
    reg = load_registry(str(y))
    e = resolve(reg, "coronary_angiography")
    assert e.floor_ok is False


def test_malformed_floor_ok_warns_operator(tmp_path):
    y = _write(tmp_path, "    floor_ok: flase\n")
    with pytest.warns(UserWarning, match="coronary_angiography"):
        load_registry(str(y))


def test_genuine_yaml_true_yields_true(tmp_path):
    y = _write(tmp_path, "    floor_ok: true\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e = resolve(load_registry(str(y)), "coronary_angiography")
    assert e.floor_ok is True


def test_malformed_entry_does_not_take_down_other_entries(tmp_path):
    """One bad floor_ok in a multi-modality file must not prevent the rest from loading."""
    y = tmp_path / "orch.yaml"
    y.write_text(
        "modalities:\n"
        "  coronary_angiography:\n"
        "    task: det\n"
        "    model_path: runs/stenosis/best.pt\n"
        "    display_name: Coronary angiography\n"
        "    finding_label: coronary_stenosis\n"
        "    finding_display: Possible coronary artery stenosis (blockage)\n"
        "    floor_ok: garbage\n"
        "  cerebral_dsa:\n"
        "    task: seg\n"
        "    model_path: runs/dsa/best.pt\n"
        "    display_name: Cerebral DSA\n"
        "    finding_label: vessel\n"
        "    finding_display: Vessel segmentation\n"
        "    floor_ok: true\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = load_registry(str(y))
    assert len(reg) == 2
    assert resolve(reg, "coronary_angiography").floor_ok is False
    assert resolve(reg, "cerebral_dsa").floor_ok is True
