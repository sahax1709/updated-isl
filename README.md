# ISL CNN-LSTM 

Drop-in replacement for a simple static-image ISL classifier, implementing
the CNN-LSTM hybrid architecture from *"ISL-Web: A Deep Learning-Enabled
Web Platform for Real-Time Indian Sign Language Recognition and
Bidirectional Conversion"* (Sections III.B and IV.C).

## What actually changed from a typical "simple" ISL model

A common starter ISL model looks like:

> take a still image of a hand → CNN → softmax over 26 alphabet letters.

That design cannot recognise dynamic signs (most ISL words are motion, not
just pose). The paper's model instead does:

1. Stream video from the webcam.
2. For each frame, run **MediaPipe Holistic** to get a compact
   **258-dim landmark vector** (two hands + body pose).
3. Buffer 30 consecutive frames → tensor of shape `(30, 258)`.
4. Feed that through **Conv1D ×3 → Bi-LSTM ×2 → FC → softmax**.

Benefits: far smaller inputs (258 floats vs. a full RGB image), trajectory
is represented explicitly, the model is invariant to background /
clothing / lighting, and it naturally handles dynamic signs.

## Files

| File | Purpose |
|---|---|
| `model.py` | Keras implementation of the CNN-LSTM architecture (Table II of the paper). |
| `feature_extraction.py` | MediaPipe Holistic wrapper; produces the 258-dim vector; also implements the horizontal-mirror augmentation with correct L/R landmark swapping. |
| `data_collection.py` | Webcam recorder. Captures N sequences of 30 frames per class and saves them as `.npy`. |
| `train.py` | Training loop — Adam, cosine-anneal 1e-3 → 1e-5 over 100 epochs, batch 32, class-weighted CE, temporal-jitter / Gaussian-noise / horizontal-mirror augmentation, early stopping (patience 15). |
| `realtime_inference.py` | Live webcam inference with a 30-frame sliding window and 50% overlap, matching Section III.B. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1 — Collect data

Pick the classes you want to support. For a small demo, start with 5–10
signs so you can validate the pipeline quickly.

```bash
python data_collection.py \
    --classes hello thankyou yes no sorry please \
    --sequences_per_class 40
```

Tips:
* Vary the signer's distance from the camera, lighting, and clothing
  between sequences — the paper achieved generalisation partly through
  signer/setting diversity.
* 40 sequences per class is the bare minimum to get the model training
  sensibly; for deployment aim for 200+ per class with multiple signers.
* Data is written to `data/<class_name>/*.npy`. You can re-run the script
  to add more samples; existing files are not overwritten.

## 2 — Train

```bash
python train.py --data data/ --out runs/exp1
```

Artifacts written to `runs/exp1/`:
* `best.keras` — best checkpoint (by val accuracy)
* `final.keras` — last epoch's weights
* `classes.json` — label-index mapping
* `train_log.csv` — per-epoch metrics
* `test_metrics.json` — held-out test-set accuracy + top-5

## 3 — Run real-time

```bash
python realtime_inference.py --run runs/exp1
```

Press `q` to quit. The HUD shows the current prediction, confidence,
buffer fill, and FPS. A 0.60 confidence threshold and 5-prediction
majority vote are applied to keep the on-screen label stable; tune
`CONFIDENCE_THRESH` and `SMOOTHING_HISTORY` in the script to taste.

## Notes on faithfulness to the paper

* **Architecture** exactly follows the Table II description (Conv1D
  filters 64/128/256, kernels 3/3/5, BatchNorm + ReLU, two Bi-LSTM(256),
  FC 512 + 256, dropout 0.5 between LSTMs and 0.3 in the FC head).
* **Feature vector** is the 258-dim subset used in Section III.B
  (hands + pose, *not* the face mesh), normalised per the Section IV.B
  scheme (hands re-centered on wrist and scaled; pose re-centered on
  hip midpoint).
* **Augmentation** matches Section IV.D: temporal jitter ±3 frames,
  Gaussian noise σ=0.01 on landmarks, horizontal flipping with the
  left-hand / right-hand blocks swapped and the pose L/R landmark pairs
  swapped (this is important — a naive x-flip without the swap corrupts
  the representation).
* **Training**: Adam, cosine annealing 1e-3 → 1e-5 over 100 epochs,
  batch 32, inverse-frequency class-weighted CE, early-stop patience 15.
* **Continuous recognition** uses a 30-frame buffer with 50% overlap.

## Things deliberately left to you

* **Signer-independent splits.** The paper splits train/val/test so no
  signer appears in more than one partition — this better measures
  generalisation. `train.py` currently does a stratified random split;
  if you collect data from multiple people, replace the split with a
  `GroupShuffleSplit` on signer ID.
* **Vocabulary size.** The paper targets 500 classes (ISL-12K). You'll
  start much smaller; scale classes and sequences-per-class together.
* **Web front-end / 3D avatar / Text-to-Sign / Hindi pipeline.** The
  paper's full platform includes a React frontend with MediaPipe running
  in the browser (TF.js) and a Three.js avatar for the reverse direction.
  Those are product-engineering pieces outside this model-training scope.

## Troubleshooting

* *"No module named mediapipe"* — `pip install mediapipe`. On Apple
  Silicon macOS, use Python 3.10–3.11 (3.12 support is patchy).
* *Webcam not found* — pass `--camera 1` (or 2, 3…) to try other indices.
* *Model overfits within a few epochs* — you need more data per class.
  Increase `--sequences_per_class` when collecting.
* *"Collecting frames..." never clears in realtime inference* — this is
  normal for the first ~1 second while the buffer fills to 30 frames.
