"""ARCADE task-1 (SYNTAX segmentation) -> standardized binary vessel pairs + nnU-Net raw.

ARCADE ships COCO-format polygon annotations (25 SYNTAX regions). For the binary vessel
student we UNION all region polygons into one 0/255 mask. Also emits nnU-Net raw for the teacher.
Confirm your ARCADE download layout; the COCO json + images are auto-discovered under the root.
"""
import argparse, os, yaml
from src.data_prep import io_utils as io


def main(cfg):
    a = cfg["datasets"]["arcade"]
    root = a["root"]
    size = cfg.get("preprocess", {}).get("size", 512)
    out_dir = "data/processed/coronary"
    raw = os.path.join(os.environ.get("nnUNet_raw", "data/nnUNet_raw"), "Dataset001_Coronary")
    n = io.coco_seg_to_pairs(root, out_dir, size=size, raw_dir=raw)
    if n == 0:
        raise SystemExit(f"No COCO images resolved under {root!r}. "
                         "Check the ARCADE path / that annotation json + images downloaded.")
    io.write_nnunet_datasetjson(raw, n)
    print(f"ARCADE -> {out_dir}/{{img,msk}} and {raw} : {n} cases")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); main(yaml.safe_load(open(a.config)))
