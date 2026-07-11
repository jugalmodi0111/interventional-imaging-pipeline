"""CLAHE + unsharp preprocessing for grayscale XCA/DSA frames."""
import glob, os
import cv2, numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm")


def clahe_unsharp(img, clip=2.0, tile=8, unsharp=0.5):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    eq = clahe.apply(img)
    if unsharp > 0:
        blur = cv2.GaussianBlur(eq, (0, 0), 3)
        eq = cv2.addWeighted(eq, 1 + unsharp, blur, -unsharp, 0)
    return eq


def process_dir(src, dst, size=None, clip=2.0, tile=8, unsharp=0.5, exts=IMG_EXTS):
    """Walk src for images, apply clahe_unsharp (+ optional square resize), write .png to dst.

    Mirrors the relative directory structure of src under dst; every output is normalized to a
    single-channel .png (so .pgm/.tif inputs land as .png). Returns the number of frames written.
    """
    n = 0
    for path in glob.glob(os.path.join(src, "**", "*"), recursive=True):
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in exts:
            continue
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        out = clahe_unsharp(g, clip=clip, tile=tile, unsharp=unsharp)
        if size:
            out = cv2.resize(out, (size, size))
        rel = os.path.splitext(os.path.relpath(path, src))[0] + ".png"
        dst_path = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        cv2.imwrite(dst_path, out)
        n += 1
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=None, help="square resize edge, e.g. 512")
    a = ap.parse_args()
    print("wrote", process_dir(a.src, a.dst, size=a.size), "frames ->", a.dst)
