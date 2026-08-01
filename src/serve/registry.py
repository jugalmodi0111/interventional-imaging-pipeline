"""Modality -> task-model + finding metadata registry, loaded from YAML config.

Each modality maps to a TaskEntry that specifies the task type (detection/segmentation),
model weights path, display names for UI, finding label (data key), and `floor_ok` safety flag.

`floor_ok=False` is the safety default: the model exists but has not cleared its accuracy floor
(Phase A acceptance gate). The orchestrator surfaces its finding as a deferred screening flag,
never as a confident positive. Only flip to true when Phase A has validated F1>=0.57, recall>=0.60.
"""
from dataclasses import dataclass
import yaml


@dataclass
class TaskEntry:
    """Registry entry for a modality: task type, model weights, and finding metadata.

    Fields:
        modality: Unique modality identifier (e.g., "coronary_angiography").
        task: Task type: "det" (detection) or "seg" (segmentation).
        model_path: Path to model weights file.
        display_name: Human-readable modality name for UI.
        finding_label: Data key for the finding (e.g., "coronary_stenosis").
        finding_display: Human-readable finding description for clinician report.
        floor_ok: If False (default), the model is below accuracy floor -> defer to human.
    """
    modality: str
    task: str
    model_path: str
    display_name: str
    finding_label: str
    finding_display: str
    floor_ok: bool = False


def load_registry(path):
    """Load registry from YAML config.

    Parses a YAML file with structure:
        modalities:
          <modality_name>:
            task: det|seg
            model_path: path/to/weights
            display_name: Human name
            finding_label: data_key
            finding_display: Clinical description
            floor_ok: true|false (optional, defaults to False)

    Returns dict mapping modality name -> TaskEntry.
    """
    cfg = yaml.safe_load(open(path)) or {}
    reg = {}
    for mod, d in (cfg.get("modalities") or {}).items():
        reg[mod] = TaskEntry(
            modality=mod,
            task=d["task"],
            model_path=d["model_path"],
            display_name=d["display_name"],
            finding_label=d["finding_label"],
            finding_display=d["finding_display"],
            floor_ok=bool(d.get("floor_ok", False))
        )
    return reg


def resolve(registry, modality):
    """Look up a modality in the registry.

    Returns the TaskEntry for the modality, or None if not found.
    """
    return registry.get(modality)
