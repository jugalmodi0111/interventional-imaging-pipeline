"""Test the modality -> task-model registry."""
from src.serve.registry import load_registry, resolve, TaskEntry


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
