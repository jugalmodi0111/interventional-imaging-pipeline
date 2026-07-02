"""CLAHE + unsharp preprocessing for grayscale XCA/DSA frames."""
import cv2, numpy as np

def clahe_unsharp(img, clip=2.0, tile=8, unsharp=0.5):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    eq = clahe.apply(img)
    if unsharp > 0:
        blur = cv2.GaussianBlur(eq, (0, 0), 3)
        eq = cv2.addWeighted(eq, 1 + unsharp, blur, -unsharp, 0)
    return eq

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--src"); ap.add_argument("--dst")
    a = ap.parse_args()
    # TODO: walk --src, apply clahe_unsharp, write to --dst
