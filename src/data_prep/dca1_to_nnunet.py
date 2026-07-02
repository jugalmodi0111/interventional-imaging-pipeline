"""DCA1 (CIMAT, 134 imgs) -> standardized binary vessel pairs + nnU-Net raw.

DCA1 pairs an angiogram with its binary GT, commonly `<i>.pgm` + `<i>_gt.pgm`
(some mirrors use .png / _gt.png). We pair by the `_gt` suffix. Appends to the same
coronary processed/ + nnU-Net Dataset001_Coronary the ARCADE step writes.
"""
import argparse, glob, os, yaml
import cv2
from src.data_prep import io_utils as io


def _pairs(root):
    imgs = {}
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        low = p.lower()
        if not low.endswith((".pgm", ".png", ".bmp", ".tif", ".tiff")):
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.endswith("_gt") or "ground" in low:
            imgs.setdefault(stem.replace("_gt", ""), {})["gt"] = p
        else:
            imgs.setdefault(stem, {})["im"] = p
    return {k: v for k, v in imgs.items() if "im" in v and "gt" in v}


def main(cfg):
    root = cfg["datasets"]["dca1"]["root"]
    size = cfg.get("preprocess", {}).get("size", 512)
    out_dir = "data/processed/coronary"
    raw = os.path.join(os.environ.get("nnUNet_raw", "data/nnUNet_raw"), "Dataset001_Coronary")
    pairs = _pairs(root)
    if not pairs:
        raise SystemExit(f"No <img>/<img>_gt pairs found under {root!r}. Confirm the DCA1 layout.")
    for stem, pv in pairs.items():
        g = cv2.imread(pv["im"], cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(pv["gt"], cv2.IMREAD_GRAYSCALE)
        if g is None or m is None:
            continue
        io.write_pair(g, m, f"dca1_{stem}", out_dir, size)
        io.write_nnunet_case(g, m, f"dca1_{stem}", raw, size)
    n = len(glob.glob(os.path.join(raw, "imagesTr", "*_0000.png")))
    io.write_nnunet_datasetjson(raw, n)
    print(f"DCA1 -> {out_dir}/{{img,msk}} and {raw} : +{len(pairs)} cases (raw total {n})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); main(yaml.safe_load(open(a.config)))
