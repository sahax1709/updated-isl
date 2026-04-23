"""
Real-time ISL recognition with a 30-frame sliding window.

Implements the continuous-recognition scheme described in the paper
(Section III.B): a sliding window of 30 frames is buffered and the
model is invoked with 50% overlap (i.e. every 15 new frames).

Usage
-----
    python realtime_inference.py --run runs/exp1
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from feature_extraction import HolisticExtractor, FEATURE_DIM
from model import SEQUENCE_LENGTH


WINDOW_STEP        = SEQUENCE_LENGTH // 2     # 50% overlap -> step = 15 frames
CONFIDENCE_THRESH  = 0.60                     # hide low-confidence predictions
SMOOTHING_HISTORY  = 5                        # majority-vote over last N preds


def main(run_dir: Path, camera_index: int):
    # ---- load model + labels ----
    model   = tf.keras.models.load_model(run_dir / "best.keras")
    classes = json.loads((run_dir / "classes.json").read_text())
    print(f"Loaded {len(classes)} classes from {run_dir}")

    # ---- webcam + MediaPipe ----
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    buffer          = deque(maxlen=SEQUENCE_LENGTH)
    pred_history    = deque(maxlen=SMOOTHING_HISTORY)
    frames_since_pred = 0
    last_label, last_conf = "", 0.0

    # FPS measurement
    t_prev, fps = time.time(), 0.0

    with HolisticExtractor() as extractor:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            vec, results = extractor.extract(frame)
            buffer.append(vec)
            frames_since_pred += 1

            # ---- fire the model once buffer is full, every WINDOW_STEP frames ----
            if len(buffer) == SEQUENCE_LENGTH and frames_since_pred >= WINDOW_STEP:
                seq  = np.expand_dims(np.stack(buffer), 0)      # (1, 30, 258)
                probs = model.predict(seq, verbose=0)[0]
                idx   = int(np.argmax(probs))
                conf  = float(probs[idx])
                if conf >= CONFIDENCE_THRESH:
                    pred_history.append(idx)
                frames_since_pred = 0

                # Majority vote over recent predictions for display stability
                if pred_history:
                    vals, counts = np.unique(pred_history, return_counts=True)
                    last_idx  = int(vals[np.argmax(counts)])
                    last_label = classes[last_idx]
                    last_conf  = conf

            # ---- draw overlays ----
            _draw_overlays(frame, results)
            _draw_hud(frame, last_label, last_conf, fps, len(buffer))

            cv2.imshow("ISL Real-Time", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # update FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

    cap.release(); cv2.destroyAllWindows()


def _draw_overlays(frame, results):
    import mediapipe as mp
    mpd   = mp.solutions.drawing_utils
    holi  = mp.solutions.holistic
    mpd.draw_landmarks(frame, results.pose_landmarks,       holi.POSE_CONNECTIONS)
    mpd.draw_landmarks(frame, results.left_hand_landmarks,  holi.HAND_CONNECTIONS)
    mpd.draw_landmarks(frame, results.right_hand_landmarks, holi.HAND_CONNECTIONS)


def _draw_hud(frame, label, conf, fps, buf_size):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    if label:
        txt = f"{label}  ({conf*100:.1f}%)"
    else:
        txt = "Collecting frames..."
    cv2.putText(frame, txt, (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS {fps:4.1f}   buffer {buf_size}/30",
                (w - 260, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="Path to a training run dir containing best.keras + classes.json")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    main(Path(args.run), args.camera)
