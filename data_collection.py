"""
Dataset collection for the CNN-LSTM ISL model.

For each sign class you want to learn, this script will:

  1. Open your webcam.
  2. Record N sequences of `SEQUENCE_LENGTH` frames each.
  3. Run MediaPipe Holistic on every frame to extract the 258-dim
     landmark vector described in feature_extraction.py.
  4. Save each sequence as  data/<class_name>/<timestamp>.npy

Usage
-----
    python data_collection.py --classes hello thankyou yes no sorry \
                              --sequences_per_class 40 \
                              --out data/

Controls during capture
-----------------------
    SPACE   skip countdown for the current sequence
    q       quit early and save everything recorded so far
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from feature_extraction import HolisticExtractor, FEATURE_DIM


SEQUENCE_LENGTH = 30          # frames per sample - matches the paper
COUNTDOWN_SECS  = 2           # pause before each recording starts


def record_one_sequence(cap, extractor, draw_window_name="ISL Capture"):
    """Record SEQUENCE_LENGTH frames and return an array of shape (T, 258)."""
    frames = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    for i in range(SEQUENCE_LENGTH):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Webcam read failed")
        frame = cv2.flip(frame, 1)             # mirror for natural UX
        vec, results = extractor.extract(frame)
        frames[i] = vec

        # Draw overlays for visual feedback
        _draw_overlays(frame, results)
        cv2.putText(frame, f"REC  frame {i+1}/{SEQUENCE_LENGTH}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(draw_window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None
    return frames


def countdown(cap, seconds, message, window_name="ISL Capture"):
    """Show a live-preview countdown with a message overlay."""
    t_end = time.time() + seconds
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.flip(frame, 1)
        remaining = int(np.ceil(t_end - time.time()))
        cv2.putText(frame, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Starts in {remaining}...", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):     return True    # skip
        if key == ord('q'):     return False   # quit
    return True


def _draw_overlays(frame, results):
    """Draw MediaPipe landmarks on the frame (for user feedback)."""
    import mediapipe as mp
    mpd   = mp.solutions.drawing_utils
    holi  = mp.solutions.holistic
    mpd.draw_landmarks(frame, results.pose_landmarks,
                       holi.POSE_CONNECTIONS)
    mpd.draw_landmarks(frame, results.left_hand_landmarks,
                       holi.HAND_CONNECTIONS)
    mpd.draw_landmarks(frame, results.right_hand_landmarks,
                       holi.HAND_CONNECTIONS)


def main(classes, sequences_per_class, out_dir, camera_index):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    with HolisticExtractor() as extractor:
        try:
            for cls in classes:
                cls_dir = out_dir / cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                existing = len(list(cls_dir.glob("*.npy")))
                print(f"\n=== Class '{cls}': already have {existing} samples ===")

                for k in range(sequences_per_class):
                    msg = f"Class '{cls}'  |  sequence {k+1}/{sequences_per_class}"
                    if not countdown(cap, COUNTDOWN_SECS, msg):
                        raise KeyboardInterrupt
                    seq = record_one_sequence(cap, extractor)
                    if seq is None:
                        raise KeyboardInterrupt

                    ts = time.strftime("%Y%m%d_%H%M%S")
                    np.save(cls_dir / f"{ts}_{k:03d}.npy", seq)

        except KeyboardInterrupt:
            print("\nStopped by user; data already written is safe.")
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", required=True,
                    help="Space-separated list of class names")
    ap.add_argument("--sequences_per_class", type=int, default=40)
    ap.add_argument("--out", default="data")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    main(args.classes, args.sequences_per_class, args.out, args.camera)
