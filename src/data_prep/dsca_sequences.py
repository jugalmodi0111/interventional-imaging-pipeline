"""Convert DSCA -> sequence tensors + MinIP. Standardize so datasets are interchangeable."""
import argparse, yaml

def main(cfg):
    # TODO: read DSCA from cfg['datasets'], apply preprocess.clahe_unsharp,
    #       write sequence tensors + MinIP into data/processed/.
    raise NotImplementedError

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); main(yaml.safe_load(open(a.config)))
