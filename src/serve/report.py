"""Output contract for the diagnostic orchestrator. A StudyReport is what a clinician sees: modality,
per-finding screening flags with calibrated confidence, and an explicit study-level defer. `to_dict`
is JSON-safe (box tuples -> lists) for the /analyze endpoint."""
from dataclasses import dataclass, field, asdict


@dataclass
class Finding:
    label: str
    display_name: str
    confidence: float
    deferred: bool
    reason: str
    severity: str | None = None
    boxes: list = field(default_factory=list)


@dataclass
class StudyReport:
    modality: str
    view: str | None
    quality_ok: bool
    findings: list
    deferred: bool
    defer_reason: str
    frames_analyzed: int
    model_versions: dict

    def to_dict(self):
        d = asdict(self)
        for f in d["findings"]:
            f["boxes"] = [list(b) for b in f["boxes"]]     # tuples -> lists for json
        return d
