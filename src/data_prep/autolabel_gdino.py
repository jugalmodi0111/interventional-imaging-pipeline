"""Build-side open-vocabulary auto-labeler: Grounding DINO (text->boxes) -> YOLO txt + COCO json.

GPU BUILD-SIDE ONLY. Grounding DINO is a teacher/labeler for scarce detection data feeding the
YOLO pseudo-label SSL loop; it is NOT a classifier and never ships to edge. All torch/transformers
imports are lazy (inside detect/autolabel_dir) so this module imports with only numpy/cv2.
"""
import glob, json, os
import numpy as np
from src.data_prep import io_utils
from src.data_prep.preprocess import clahe_unsharp, IMG_EXTS

# Grounding DINO separates candidate classes with ' . ' inside the text prompt.
DEFAULT_PROMPTS = {
    "stenosis": "stenosis",
    "catheter": "catheter . guidewire",
    "coronary": "coronary vessel . artery",
}


def dino_boxes_to_yolo_lines(boxes_xyxy, W, H, class_id=0):
    """Pixel [x1,y1,x2,y2] boxes -> center-normalized 6-dp YOLO lines (io_utils format)."""
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    lines = []
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        lines.append(f"{class_id} {(x1 + w / 2) / W:.6f} {(y1 + h / 2) / H:.6f} "
                     f"{w / W:.6f} {h / H:.6f}")
    return lines


def filter_detections(boxes, scores, labels, box_thresh=0.35):
    """Keep detections with score >= box_thresh. Returns (boxes[K,4], scores[K], labels list)."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    keep = scores >= box_thresh
    return boxes[keep], scores[keep], [l for l, k in zip(labels, keep) if k]


def dino_to_coco(detections_per_image, categories):
    """Per-image detections -> COCO dict {images, annotations, categories}, bbox = [x,y,w,h].

    categories: list[str] class names -> 1-based integer ids. Each det:
    {file_name,width,height,boxes(xyxy),scores,labels}; annotations whose label is absent from
    `categories` are skipped. This json is the labeling artifact a human reviews.
    """
    cat_id = {name: i + 1 for i, name in enumerate(categories)}
    coco = {"images": [], "annotations": [],
            "categories": [{"id": cat_id[n], "name": n} for n in categories]}
    aid = 1
    for iid, det in enumerate(detections_per_image, 1):
        coco["images"].append({"id": iid, "file_name": det["file_name"],
                               "width": int(det["width"]), "height": int(det["height"])})
        boxes = np.asarray(det.get("boxes", []), dtype=float).reshape(-1, 4)
        labels = det.get("labels", [])
        scores = det.get("scores") if det.get("scores") is not None else [None] * len(labels)
        for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
            if lab not in cat_id:
                continue
            w, h = float(x2 - x1), float(y2 - y1)
            coco["annotations"].append({"id": aid, "image_id": iid, "category_id": cat_id[lab],
                                        "bbox": [float(x1), float(y1), w, h], "area": w * h,
                                        "iscrowd": 0,
                                        "score": None if sc is None else float(sc)})
            aid += 1
    return coco


# --- heavy pieces (GPU build-side; lazy imports, not run locally) -------------

def detect(image, prompt, box_thresh=0.35, text_thresh=0.25,
           model_id="IDEA-Research/grounding-dino-tiny", device=None):
    """Grounding DINO zero-shot detection via HF transformers. Lazy-imports torch/transformers.

    image: HxW / HxWx3 uint8 array or path. prompt: ' . '-separated classes (see DEFAULT_PROMPTS).
    Returns (boxes_xyxy[K,4] float, scores[K] float, labels list[str]).
    """
    import cv2, torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
    arr = image
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    H, W = arr.shape[:2]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    text = prompt.strip().lower()
    text = text if text.endswith(".") else text + " ."   # HF wants a trailing-dot query
    inputs = processor(images=Image.fromarray(arr), text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=box_thresh, text_threshold=text_thresh,
        target_sizes=[(H, W)])[0]
    labels = res.get("text_labels", res.get("labels"))   # key renamed across transformers versions
    return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), list(labels)


def autolabel_dir(img_dir, out_dir, prompt, class_map, split=True, box_thresh=0.35,
                  text_thresh=0.25, model_id="IDEA-Research/grounding-dino-tiny",
                  size=512, device=None):
    """Loop images in img_dir, Grounding-DINO auto-label -> YOLO dataset + COCO review json.

    class_map: {label_string: yolo_class_id}. Writes images/labels/<split>/<stem>.{png,txt} through
    io_utils conventions, an autolabel_coco.json for human review, and data.yaml. Returns count.
    No-ops gracefully on an empty dir.
    """
    import cv2
    paths = [p for p in sorted(glob.glob(os.path.join(img_dir, "**", "*"), recursive=True))
             if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMG_EXTS]
    if not paths:
        print(f"[autolabel_gdino] no images under {img_dir} -> nothing to do")
        return 0
    names = [n for n, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    dets_coco, n = [], 0
    for ip in paths:
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        H, W = g.shape
        boxes, scores, labels = detect(g, prompt, box_thresh, text_thresh, model_id, device)
        boxes, scores, labels = filter_detections(boxes, scores, labels, box_thresh)
        lines = []
        for box, lab in zip(boxes, labels):
            cid = class_map.get(lab)
            if cid is not None:
                lines += dino_boxes_to_yolo_lines([box], W, H, cid)
        stem = os.path.splitext(os.path.basename(ip))[0]
        sp = io_utils.split_of(stem) if split else "train"
        io_utils.ensure(os.path.join(out_dir, "images", sp), os.path.join(out_dir, "labels", sp))
        cv2.imwrite(os.path.join(out_dir, "images", sp, stem + ".png"),
                    cv2.resize(clahe_unsharp(g), (size, size)))
        open(os.path.join(out_dir, "labels", sp, stem + ".txt"), "w").write("\n".join(lines))
        dets_coco.append({"file_name": stem + ".png", "width": W, "height": H,
                          "boxes": boxes, "scores": scores, "labels": labels})
        n += 1
    io_utils.ensure(out_dir)
    json.dump(dino_to_coco(dets_coco, names),
              open(os.path.join(out_dir, "autolabel_coco.json"), "w"), indent=2)
    io_utils.write_yolo_datayaml(out_dir, names)
    print(f"[autolabel_gdino] labeled {n} images -> {out_dir} (classes={names})")
    return n
