"""Modality/view router decision layer + edge classifier wrapper.

The pure `decide_modality` is torch-free and unit-tested. `ModalityRouter` lazy-loads the distilled
MobileNetV3 student (edge) and delegates the keep/defer call to `decide_modality`. Safety default:
DEFER (modality 'unknown') whenever the top class is weak or the top-two margin is thin — a wrong
route sends a frame to the wrong disease model, so ambiguity must never resolve to a guess.
"""
from dataclasses import dataclass


@dataclass
class ModalityDecision:
    modality: str
    view: str | None
    quality_ok: bool
    confidence: float
    deferred: bool
    reason: str


def decide_modality(probs, *, keep_thr=0.60, margin=0.15,
                    quality_prob=None, quality_thr=0.5, view=None):
    """Softmax dict -> keep/defer decision. Defers to 'unknown' on weak top prob or thin margin."""
    quality_ok = quality_prob is None or quality_prob >= quality_thr
    if not probs:
        return ModalityDecision("unknown", view, quality_ok, 0.0, True, "router-uncertain")
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    (top_label, top_p) = ranked[0]
    runner_p = ranked[1][1] if len(ranked) > 1 else 0.0
    if not quality_ok:
        return ModalityDecision(top_label, view, False, float(top_p), True, "low-quality")
    if top_p < keep_thr or (top_p - runner_p) < margin:
        return ModalityDecision("unknown", view, quality_ok, float(top_p), True, "router-uncertain")
    return ModalityDecision(top_label, view, quality_ok, float(top_p), False, "confident")
