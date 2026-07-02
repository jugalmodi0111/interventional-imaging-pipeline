"""Train AVF bruit screener (small ViT / CNN-BiLSTM on mel-spectrograms). Stub."""
import argparse, yaml
def main(cfg):
    # TODO: librosa mel-spectrograms -> small ViT; labels from duplex ultrasound.
    raise NotImplementedError
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    main(yaml.safe_load(open(ap.parse_args().config)))
