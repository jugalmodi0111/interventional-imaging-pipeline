"""Per-source stenosis val: split the merged val set by dataset of origin and run ultralytics
.val() on each, so we see WHERE recall fails. Run GPU-side after a training run.

Source inferred from the stem naming the converters emit:
  CADICA  pXX_vYY_NNNNN                          -> starts 'p<digits>_v<digits>'
  Danilov <site>_<pat>_<seq>_<frame> (4 all-digit groups)
  ARCADE  <split>_<n> or <n>                     -> everything else

ultralytics is imported lazily inside main() so source_of/_write_lists are unit-testable
without torch/ultralytics (repo invariant: src/* imports torch-free).
"""
import os, re, glob, argparse, yaml

_CADICA = re.compile(r"^p\d+_v\d+")
_DANILOV = re.compile(r"^\d+_\d+_\d+_\d+$")


def source_of(stem):
    """Infer the origin dataset of a YOLO stem: 'cadica' | 'danilov' | 'arcade'."""
    if _CADICA.match(stem):
        return "cadica"
    if _DANILOV.match(stem):
        return "danilov"
    return "arcade"


def _write_lists(proc):
    """Write per-source val image-path .txt lists under proc; return {src: (list_path, count)}."""
    buckets = {}
    for ip in sorted(glob.glob(os.path.join(proc, "images", "val", "*"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        buckets.setdefault(source_of(stem), []).append(os.path.abspath(ip))
    lists = {}
    for src, paths in buckets.items():
        lp = os.path.join(proc, f"val_{src}.txt")
        open(lp, "w").write("\n".join(paths))
        lists[src] = (lp, len(paths))
    return lists


def main(weights, proc, conf=0.001):
    from ultralytics import YOLO
    base = yaml.safe_load(open(os.path.join(proc, "data.yaml")))
    for src, (lp, n) in sorted(_write_lists(proc).items()):
        dy = os.path.join(proc, f"data_{src}.yaml")
        cfg = dict(base); cfg["val"] = os.path.abspath(lp)
        yaml.safe_dump(cfg, open(dy, "w"))
        b = YOLO(weights).val(data=dy, conf=conf, verbose=False).box
        print(f"{src:8s} n={n:5d}  P {b.mp:.3f}  R {b.mr:.3f}  mAP50 {b.map50:.3f}  mAP50-95 {b.map:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--proc", default="data/processed/stenosis")
    a = ap.parse_args(); main(a.weights, a.proc)
