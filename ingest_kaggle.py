"""
Convert a Kaggle ISL image dataset into the (30, 258) .npy sequence
format expected by train.py.

Targeted at:
    https://www.kaggle.com/datasets/soumyakushwaha/indian-sign-language-dataset

...but works on any dataset shaped as `<root>/<class_name>/*.{jpg,jpeg,png}`,
auto-discovering whatever classes are present (e.g. 1-9 + A-Z, or 0-9 + A-Z).

Each image is treated as a STATIC sign: MediaPipe Holistic is run once on
the image to produce a 258-dim landmark vector, which is then replicated
across 30 timesteps to match the model's input shape. The training
augmentation pipeline (Gaussian noise + horizontal mirror) breaks the
trivial frame-to-frame symmetry at training time.

Usage
-----
    # 1. Download + unzip the Kaggle dataset somewhere, e.g.
    #    kaggle datasets download -d soumyakushwaha/indian-sign-language-dataset -p ./kaggle_isl --unzip
    #
    # 2. Run this ingest script:
    python ingest_kaggle.py --src ./kaggle_isl --out data/

Notes
-----
* Images where MediaPipe fails to detect any hand are skipped and counted
  in the summary printed at the end.
* Existing .npy files in the output directory are NOT overwritten, so you
  can mix Kaggle-derived data with your own webcam recordings from
  data_collection.py.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from feature_extraction import HolisticExtractor, FEATURE_DIM
from model import SEQUENCE_LENGTH


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_dirs(src: Path):
    """Return (class_name, dir_path) pairs for every immediate subdirectory
    that contains at least one image. Recurses one level if `src` itself
    only contains a single wrapper folder (common in Kaggle ZIPs)."""
    candidates = [p for p in src.iterdir() if p.is_dir()]

    # Unwrap a single top-level container (e.g. "Indian Sign Language Dataset/")
    if len(candidates) == 1:
        inner = [p for p in candidates[0].iterdir() if p.is_dir()]
        if inner:
            candidates = inner

    pairs = []
    for d in sorted(candidates, key=lambda p: p.name):
        has_images = any(p.suffix.lower() in IMAGE_EXTS for p in d.iterdir() if p.is_file())
        if has_images:
            pairs.append((d.name, d))
    return pairs


def image_to_sequence(img_path: Path, extractor: HolisticExtractor):
    """Run Holistic on one image and return a (30, 258) array, or None on failure."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, "unreadable"
    vec, results = extractor.extract(img)

    # Skip images where neither hand was detected — pure-zero hand vectors
    # would just teach the model that the class is "no hand visible".
    if (results.left_hand_landmarks is None
            and results.right_hand_landmarks is None):
        return None, "no_hand"

    return np.repeat(vec[None, :], SEQUENCE_LENGTH, axis=0), "ok"


def main(src, out, limit_per_class):
    src = Path(src).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    if not src.is_dir():
        sys.exit(f"--src '{src}' is not a directory")
    out.mkdir(parents=True, exist_ok=True)

    class_pairs = find_class_dirs(src)
    if not class_pairs:
        sys.exit(f"No class subdirectories with images found under {src}")

    print(f"Discovered {len(class_pairs)} classes under {src}:")
    print("  " + ", ".join(name for name, _ in class_pairs))
    print(f"Writing sequences to {out}\n")

    totals = {"ok": 0, "no_hand": 0, "unreadable": 0, "skipped_existing": 0}

    with HolisticExtractor(static_image_mode=True) as extractor:
        for cls_name, cls_dir in class_pairs:
            cls_out = out / cls_name
            cls_out.mkdir(parents=True, exist_ok=True)

            images = sorted(p for p in cls_dir.rglob("*")
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if limit_per_class:
                images = images[:limit_per_class]

            per_cls = {"ok": 0, "no_hand": 0, "unreadable": 0, "skipped_existing": 0}
            for img_path in tqdm(images, desc=f"{cls_name:>4s}", unit="img"):
                npy_path = cls_out / f"{img_path.stem}.npy"
                if npy_path.exists():
                    per_cls["skipped_existing"] += 1
                    continue
                seq, status = image_to_sequence(img_path, extractor)
                if seq is None:
                    per_cls[status] += 1
                    continue
                np.save(npy_path, seq.astype(np.float32))
                per_cls["ok"] += 1

            for k, v in per_cls.items():
                totals[k] += v
            print(f"    -> ok={per_cls['ok']}  "
                  f"no_hand={per_cls['no_hand']}  "
                  f"unreadable={per_cls['unreadable']}  "
                  f"already_present={per_cls['skipped_existing']}")

    print("\n=== Summary ===")
    for k, v in totals.items():
        print(f"  {k:>16s}: {v}")
    print(f"\nNext step:\n  python train.py --data {out} --out runs/exp1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Path to the unzipped Kaggle dataset root")
    ap.add_argument("--out", default="data",
                    help="Output directory (default: data/)")
    ap.add_argument("--limit_per_class", type=int, default=0,
                    help="Optional cap on images processed per class "
                         "(0 = no limit; useful for a quick smoke test)")
    args = ap.parse_args()
    main(args.src, args.out, args.limit_per_class)
