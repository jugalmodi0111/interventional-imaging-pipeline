"""Modality -> task-model + finding metadata registry, loaded from YAML config.

Each modality maps to a TaskEntry that specifies the task type (detection/segmentation),
model weights path, display names for UI, finding label (data key), and `floor_ok` safety flag.

`floor_ok=False` is the safety default: the model exists but has not cleared its accuracy floor
(Phase A acceptance gate). The orchestrator surfaces its finding as a deferred screening flag,
never as a confident positive. Only flip to true when Phase A has validated F1>=0.57, recall>=0.60.

`floor_ok` is fail-safe on parse: PyYAML's SafeLoader only resolves the literal tokens
true/false/yes/no/on/off (case-insensitively, unquoted) to real booleans -- anything else
(a typo, a quoted string, a number, null, a list) loads as a non-bool value. We only ever treat
a genuine `True` object as floor_ok=True; every other value -- including missing -- downgrades to
False (defer to human) rather than failing open.
"""
from dataclasses import dataclass
import warnings
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
        floor_ok: If False (default), the model is below accuracy floor -> defer to human. Only a
            genuine YAML boolean `true` sets this True; any malformed value fails safe to False.
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

    `floor_ok` is fail-safe: only an unquoted YAML `true` (or `yes`/`on`) -- i.e. a value PyYAML's
    SafeLoader actually resolves to the Python singleton `True` -- yields floor_ok=True. A missing
    key yields False silently (the documented default). A *present but malformed* value (a typo, a
    quoted string, a number, null, a list -- anything that isn't a real bool) also downgrades to
    False rather than raising, so one bad entry can't take the whole registry down, but it is
    reported via warnings.warn so an operator can catch the misconfiguration.

    Returns dict mapping modality name -> TaskEntry.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    reg = {}
    for mod, d in (cfg.get("modalities") or {}).items():
        raw = d.get("floor_ok", False)
        floor_ok = raw is True                      # only a genuine YAML bool `true` passes
        if not floor_ok and "floor_ok" in d and raw is not False:
            warnings.warn(
                f"registry: modality {mod!r} has non-boolean floor_ok={raw!r} "
                "(expected YAML true/false); failing safe to floor_ok=False (defer to human).",
                stacklevel=2,
            )
        reg[mod] = TaskEntry(
            modality=mod,
            task=d["task"],
            model_path=d["model_path"],
            display_name=d["display_name"],
            finding_label=d["finding_label"],
            finding_display=d["finding_display"],
            floor_ok=floor_ok
        )
    return reg


def resolve(registry, modality):
    """Look up a modality in the registry.

    Returns the TaskEntry for the modality, or None if not found.
    """
    return registry.get(modality)
