"""Train coronary segmentation: nnU-Net teacher (CLI) then distill student. Stub."""
import argparse, yaml
def main(cfg):
    # 1) nnU-Net teacher via nnUNetv2_train (subprocess) -> cache predictions
    # 2) train TinyUNet with src.models.distill.kd_loss
    # 3) eval with src.eval.metrics ; 4) export via src.export
    raise NotImplementedError
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    main(yaml.safe_load(open(ap.parse_args().config)))
