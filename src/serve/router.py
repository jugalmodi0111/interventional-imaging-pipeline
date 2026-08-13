"""Modality/view router decision layer + edge classifier wrapper.

The pure `decide_modality` is torch-free and unit-tested. `ModalityRouter` lazy-loads the distilled
MobileNetV3 student (edge) and delegates the keep/defer call to `decide_modality`. Safety default:
DEFER (modality 'unknown') whenever the top class is weak or the top-two margin is thin — a wrong
route sends a frame to the wrong disease model, so ambiguity must never resolve to a guess.

Fail-safe on the router's OWN failure: `classify` collapses every load/run failure — missing
weights file, timm/torch not installed, corrupt state_dict, a torch error mid-forward — into
`RouterUnavailable` (defined next to its sibling `ModelUnavailable` in orchestrator.py). The
orchestrator converts it into a deferred study (reason "router-unavailable"), so an undeployed
router never escapes as a raw ModuleNotFoundError/FileNotFoundError and is operationally
distinguishable from a genuine bug.
"""
from dataclasses import dataclass

from src.serve.orchestrator import RouterUnavailable   # torch-free import; no cycle (orchestrator
                                                       # imports this module only lazily, inside
                                                       # build_orchestrator's body)


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


class ModalityRouter:
    """Edge modality classifier. Lazy-loads the distilled MobileNetV3 student; `classify` returns a
    keep/defer ModalityDecision. Torch is imported only in `_load`/`_probs`, never at module scope,
    so this module stays import-safe on a laptop with no torch installed. All keep/defer judgment
    lives in `decide_modality`; this class only produces the probability dict it consumes."""
    def __init__(self, weights, labels, thresholds=None, size=224):
        self.weights, self.labels, self.size = weights, labels, size
        self.thresholds = thresholds or {"keep_thr": 0.60, "margin": 0.15, "quality_thr": 0.5}
        self._model = None

    def _load(self):
        import torch, timm
        m = timm.create_model("mobilenetv3_small_100", num_classes=len(self.labels), in_chans=3)
        m.load_state_dict(torch.load(self.weights, map_location="cpu")); m.eval()
        return m

    def _probs(self, frame):
        import torch, cv2
        from src.data_prep.preprocess import clahe_unsharp
        if self._model is None:
            self._model = self._load()
        x = cv2.resize(clahe_unsharp(frame), (self.size, self.size)).astype("float32") / 255.0
        t = torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1)
        with torch.no_grad():
            p = torch.softmax(self._model(t), 1).squeeze(0).tolist()
        return {l: float(pi) for l, pi in zip(self.labels, p)}

    def classify(self, frame):
        """Preprocess + run the classifier, then defer all keep/defer judgment to decide_modality.

        Any failure to LOAD or RUN the classifier itself (missing weights file, timm/torch not
        installed, a corrupt state_dict, a torch error mid-forward) raises `RouterUnavailable`
        instead of the raw exception: the orchestrator catches exactly that type and defers the
        study with reason "router-unavailable" — never a crash, never a guess."""
        try:
            probs = self._probs(frame)
        except Exception as e:
            raise RouterUnavailable(
                f"modality router unavailable (weights={self.weights!r}): {e}") from e
        return decide_modality(probs,
                               keep_thr=self.thresholds["keep_thr"],
                               margin=self.thresholds["margin"],
                               quality_thr=self.thresholds["quality_thr"])
