"""Grounded-SAM (build-side): Grounding DINO boxes -> box-prompted SAM masks for human labeling.

GPU BUILD-SIDE ONLY. Turns the auto-labeler's detection boxes into per-object masks a human
corrects, then feeds io_utils.write_pair for the segmentation-student track. torch / SAM imports
are lazy so this module imports with only numpy/cv2 (no torch, no segment-anything, no GPU).
"""
import numpy as np


class GroundedSAM:
    """Box-prompted SAM mask predictor. Loads a lightweight SAM (MobileSAM/vit_t) lazily on first use."""

    def __init__(self, sam_ckpt=None, model_type="vit_t", device=None):
        self.sam_ckpt, self.model_type, self.device = sam_ckpt, model_type, device
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        import torch
        try:                                   # MobileSAM (vit_t) -> tiny, edge-labeler friendly
            from mobile_sam import sam_model_registry, SamPredictor
        except Exception:                      # fall back to full segment-anything
            from segment_anything import sam_model_registry, SamPredictor
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_ckpt).to(self.device).eval()
        self._predictor = SamPredictor(sam)
        return self._predictor

    def mask_from_boxes(self, image, boxes_xyxy):
        """image: HxW or HxWx3 uint8. Returns list[bool mask HxW], one per input box."""
        import cv2
        arr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        predictor = self._load()
        predictor.set_image(arr)
        boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
        masks = []
        for box in boxes:
            m, _, _ = predictor.predict(box=box[None, :], multimask_output=False)
            masks.append(m[0].astype(bool))
        return masks


def mask_from_boxes(image, boxes_xyxy, sam_ckpt=None, model_type="vit_t", device=None):
    """Functional shortcut: one-shot box-prompted SAM masks (see GroundedSAM.mask_from_boxes)."""
    return GroundedSAM(sam_ckpt, model_type, device).mask_from_boxes(image, boxes_xyxy)


def to_seg_pairs(image_gray, masks, stem, out_dir, size=512):
    """Union human-corrected `masks` (list[bool HxW]) -> one binary mask, emit via io_utils.write_pair.

    Lands data/processed/<task>/{img,msk}/<stem>.png for the seg student. For per-class YOLO
    detection labels from masks instead, feed the same masks to io_utils.masks_to_yolo.
    """
    from src.data_prep import io_utils
    if not len(masks):
        return 0
    union = np.zeros(np.asarray(masks[0]).shape, np.uint8)
    for m in masks:
        union = np.maximum(union, (np.asarray(m) > 0).astype(np.uint8))
    io_utils.write_pair(image_gray, union, stem, out_dir, size)
    return 1
