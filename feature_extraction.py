"""
MediaPipe Holistic landmark extraction.

Extracts a 258-dimensional feature vector per frame, exactly as described
in the paper (Section IV.B):

    63   left-hand  landmarks  (21 points x 3 coords)
    63   right-hand landmarks  (21 points x 3 coords)
    132  pose       landmarks  (33 points x 4 = x, y, z, visibility)
    ---
    258  total features per frame

Also provides the normalization scheme from the paper: hand landmarks are
re-centered on the wrist and scaled by the maximum inter-landmark distance;
pose landmarks are re-centered on the midpoint between the two hips.
"""

from __future__ import annotations

import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

FEATURE_DIM      = 258
HAND_DIM         = 63            # 21 * 3
POSE_DIM         = 132           # 33 * 4

# ----- left/right landmark-pair indices used during horizontal mirroring -----
# (Pose landmark indices: see https://google.github.io/mediapipe/solutions/pose)
POSE_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


# ---------------------------------------------------------------------------
# Holistic wrapper
# ---------------------------------------------------------------------------
class HolisticExtractor:
    """Thin wrapper around MediaPipe Holistic that returns a 258-dim vector."""

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence:  float = 0.5,
                 model_complexity:         int   = 1):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
    def close(self): self.holistic.close()

    # ---- main API ----
    def extract(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, object]:
        """
        Run Holistic on one BGR frame and return (feature_vector, results).
        `results` is the raw MediaPipe output so callers can draw overlays.
        """
        # MediaPipe expects RGB
        rgb = frame_bgr[:, :, ::-1]
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        return _results_to_vector(results), results


# ---------------------------------------------------------------------------
# Raw landmarks -> 258-dim vector
# ---------------------------------------------------------------------------
def _hand_to_vec(hand_landmarks) -> np.ndarray:
    """Return 63-dim vector for one hand. Returns zeros when not detected."""
    if hand_landmarks is None:
        return np.zeros(HAND_DIM, dtype=np.float32)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                   dtype=np.float32)        # shape (21, 3)

    # Normalize: re-center on wrist (landmark 0), scale by max inter-landmark
    # distance so features are invariant to position and hand-to-camera range.
    wrist = pts[0]
    pts   = pts - wrist
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts = pts / scale
    return pts.flatten()


def _pose_to_vec(pose_landmarks) -> np.ndarray:
    """Return 132-dim vector for pose. Returns zeros when not detected."""
    if pose_landmarks is None:
        return np.zeros(POSE_DIM, dtype=np.float32)
    pts = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                    for lm in pose_landmarks.landmark],
                   dtype=np.float32)        # shape (33, 4)

    # Normalize xyz relative to hip midpoint (landmarks 23 & 24).
    hip_mid = (pts[23, :3] + pts[24, :3]) / 2.0
    pts[:, :3] = pts[:, :3] - hip_mid
    return pts.flatten()


def _results_to_vector(results) -> np.ndarray:
    """Convert MediaPipe Holistic results to a 258-dim float32 vector."""
    left  = _hand_to_vec(results.left_hand_landmarks)
    right = _hand_to_vec(results.right_hand_landmarks)
    pose  = _pose_to_vec(results.pose_landmarks)
    vec   = np.concatenate([left, right, pose]).astype(np.float32)
    assert vec.shape[0] == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {vec.shape[0]}"
    return vec


# ---------------------------------------------------------------------------
# Horizontal-mirror augmentation (used by train.py)
# ---------------------------------------------------------------------------
def horizontal_flip_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Mirror a (T, 258) sequence across the body's vertical axis:

      * negate x coordinates,
      * swap left-hand and right-hand blocks,
      * swap left/right pose landmark pairs.
    """
    out   = seq.copy()
    T     = out.shape[0]

    # --- Hands: negate x, swap left <-> right blocks ---------------------
    left  = out[:, 0:HAND_DIM].reshape(T, 21, 3).copy()
    right = out[:, HAND_DIM:2*HAND_DIM].reshape(T, 21, 3).copy()
    left[..., 0]  *= -1
    right[..., 0] *= -1
    out[:, 0:HAND_DIM]          = right.reshape(T, HAND_DIM)
    out[:, HAND_DIM:2*HAND_DIM] = left.reshape(T, HAND_DIM)

    # --- Pose: negate x, then swap L/R landmark pairs -------------------
    pose = out[:, 2*HAND_DIM:].reshape(T, 33, 4).copy()
    pose[..., 0] *= -1
    for a, b in POSE_LR_PAIRS:
        pose[:, [a, b]] = pose[:, [b, a]]
    out[:, 2*HAND_DIM:] = pose.reshape(T, POSE_DIM)
    return out
