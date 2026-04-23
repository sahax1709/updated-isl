"""
Train the CNN-LSTM ISL model.

Training recipe taken directly from Section IV.D of the paper:

    * Optimizer     Adam (initial lr = 1e-3)
    * Schedule      Cosine annealing -> 1e-5 over 100 epochs
    * Batch size    32
    * Loss          Class-weighted categorical cross-entropy
    * Augmentation  temporal jitter (+/- 3 frames),
                    Gaussian noise (sigma = 0.01),
                    random horizontal mirroring with L/R landmark swap
    * Early stop    patience = 15 on validation accuracy

Directory layout expected (produced by data_collection.py):

    data/
        hello/      0001.npy   0002.npy   ...
        thankyou/   0001.npy   ...
        yes/        ...

Usage
-----
    python train.py --data data/ --out runs/exp1
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from model import build_cnn_lstm, SEQUENCE_LENGTH, FEATURE_DIM
from feature_extraction import horizontal_flip_sequence


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(data_dir: Path):
    """Load all .npy files under data/<class>/*.npy; enforce shape (30, 258)."""
    classes = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    label_map = {c: i for i, c in enumerate(classes)}
    X, y = [], []
    for c in classes:
        for f in (data_dir / c).glob("*.npy"):
            seq = np.load(f)
            if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
                print(f"  skipped {f} (shape {seq.shape})")
                continue
            X.append(seq)
            y.append(label_map[c])
    if not X:
        raise RuntimeError(f"No usable samples found under {data_dir}")
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, classes


# ---------------------------------------------------------------------------
# tf.data augmentation pipeline (applied on training set only)
# ---------------------------------------------------------------------------
def _tf_augment(seq, label, max_jitter=3, noise_sigma=0.01, flip_prob=0.5):
    """
    Paper's augmentation applied as a tf.py_function so we can mix numpy
    (horizontal_flip_sequence with explicit L/R index swapping) with
    vectorised TF ops (jitter + noise).
    """
    def _np(s):
        s = s.copy()

        # ---- temporal jitter: pad + random crop back to SEQUENCE_LENGTH ----
        pad = max_jitter
        padded = np.concatenate([np.repeat(s[:1], pad, axis=0),
                                 s,
                                 np.repeat(s[-1:], pad, axis=0)], axis=0)
        start = np.random.randint(0, 2 * pad + 1)
        s = padded[start:start + SEQUENCE_LENGTH]

        # ---- horizontal mirror ----
        if np.random.rand() < flip_prob:
            s = horizontal_flip_sequence(s)
        return s.astype(np.float32)

    seq = tf.numpy_function(_np, [seq], tf.float32)
    seq.set_shape([SEQUENCE_LENGTH, FEATURE_DIM])

    # ---- Gaussian noise on the coordinates ----
    seq = seq + tf.random.normal(tf.shape(seq), stddev=noise_sigma)
    return seq, label


def make_tf_dataset(X, y, num_classes, batch_size, training: bool):
    ds = tf.data.Dataset.from_tensor_slices(
        (X, tf.keras.utils.to_categorical(y, num_classes=num_classes)))
    if training:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
        ds = ds.map(_tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Cosine LR schedule callback (1e-3 -> 1e-5 over `epochs` epochs)
# ---------------------------------------------------------------------------
class CosineAnnealLR(tf.keras.callbacks.Callback):
    def __init__(self, lr_start=1e-3, lr_end=1e-5, epochs=100):
        super().__init__()
        self.lr_start = lr_start
        self.lr_end   = lr_end
        self.epochs   = epochs

    def on_epoch_begin(self, epoch, logs=None):
        t  = min(epoch, self.epochs - 1) / max(self.epochs - 1, 1)
        lr = self.lr_end + 0.5 * (self.lr_start - self.lr_end) * (1 + math.cos(math.pi * t))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"  lr = {lr:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(data_dir, out_dir, epochs, batch_size, seed):
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed); tf.random.set_seed(seed)

    print(f"Loading data from {data_dir} ...")
    X, y, classes = load_dataset(data_dir)
    num_classes = len(classes)
    print(f"  samples: {len(X)}   classes: {num_classes}")

    # Stratified 70 / 15 / 15 split (signer-independent splits recommended
    # for a real deployment - substitute your own splitter if available).
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15 / 0.85,
        stratify=y_trainval, random_state=seed)
    print(f"  train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    # Class-weighted cross-entropy (paper uses inverse-frequency weights)
    cw = compute_class_weight("balanced",
                              classes=np.arange(num_classes), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    train_ds = make_tf_dataset(X_train, y_train, num_classes, batch_size, training=True)
    val_ds   = make_tf_dataset(X_val,   y_val,   num_classes, batch_size, training=False)
    test_ds  = make_tf_dataset(X_test,  y_test,  num_classes, batch_size, training=False)

    model = build_cnn_lstm(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )
    model.summary()

    callbacks = [
        CosineAnnealLR(1e-3, 1e-5, epochs),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                         patience=15, mode="max",
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.keras"),
            monitor="val_accuracy", mode="max",
            save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv")),
    ]

    model.fit(train_ds, validation_data=val_ds,
              epochs=epochs, class_weight=class_weight, callbacks=callbacks)

    # ---- final test-set evaluation ----
    results = model.evaluate(test_ds, return_dict=True)
    print("Test metrics:", results)

    # ---- persist label map + test results ----
    with open(out_dir / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    model.save(out_dir / "final.keras")
    print(f"\nArtifacts written to {out_dir.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out",  default="runs/exp1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.data, args.out, args.epochs, args.batch_size, args.seed)
