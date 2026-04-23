"""
CNN-LSTM hybrid model for Indian Sign Language recognition.

Architecture faithfully reproduces the design described in
"ISL-Web: A Deep Learning-Enabled Web Platform for Real-Time Indian Sign
Language Recognition and Bidirectional Conversion" (Section IV.C, Table II).

Input:  (batch, 30 frames, 258 features)   # MediaPipe Holistic landmarks
Output: (batch, num_classes) softmax probabilities
"""

from tensorflow.keras import layers, models, regularizers


# ----- Default hyper-parameters from the paper -----
SEQUENCE_LENGTH = 30           # 30 frames per sample
FEATURE_DIM     = 258          # 63 (L hand) + 63 (R hand) + 132 (pose x,y,z,vis)
LSTM_UNITS      = 256
FC_UNITS        = (512, 256)
DROPOUT_LSTM    = 0.5
DROPOUT_FC      = 0.3


def build_cnn_lstm(num_classes: int,
                   sequence_length: int = SEQUENCE_LENGTH,
                   feature_dim: int = FEATURE_DIM,
                   l2_reg: float = 1e-4) -> models.Model:
    """
    Build the CNN-LSTM model.

    Spatial branch: three Conv1D layers (kernels 3, 3, 5; filters 64, 128, 256)
    each followed by BatchNorm + ReLU, applied across the time axis with
    padding='same' so the sequence length is preserved. This yields a
    (30, 256) tensor that feeds directly into the temporal branch.

    Temporal branch: two stacked Bidirectional LSTM layers (256 units each)
    with dropout 0.5 between them. The second Bi-LSTM returns only the final
    hidden state, producing a 512-dim vector.

    Classifier head: FC(512) + ReLU + Dropout(0.3) -> FC(256) + ReLU +
    Dropout(0.3) -> Dense(num_classes, softmax).
    """
    reg = regularizers.l2(l2_reg) if l2_reg else None

    inp = layers.Input(shape=(sequence_length, feature_dim), name="landmark_sequence")

    # ----- Spatial feature extraction (Conv1D over time, padding='same') -----
    x = layers.Conv1D(64, kernel_size=3, padding="same",
                      kernel_regularizer=reg, name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same",
                      kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)

    x = layers.Conv1D(256, kernel_size=5, padding="same",
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)   # -> (batch, 30, 256)

    # ----- Temporal modelling (stacked Bi-LSTM) -----
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=True,
                    kernel_regularizer=reg),
        name="bilstm1")(x)
    x = layers.Dropout(DROPOUT_LSTM, name="dropout_lstm")(x)

    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=False,
                    kernel_regularizer=reg),
        name="bilstm2")(x)                           # -> (batch, 512)

    # ----- Classifier head -----
    x = layers.Dense(FC_UNITS[0], kernel_regularizer=reg, name="fc1")(x)
    x = layers.ReLU(name="relu_fc1")(x)
    x = layers.Dropout(DROPOUT_FC, name="dropout_fc1")(x)

    x = layers.Dense(FC_UNITS[1], kernel_regularizer=reg, name="fc2")(x)
    x = layers.ReLU(name="relu_fc2")(x)
    x = layers.Dropout(DROPOUT_FC, name="dropout_fc2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return models.Model(inp, out, name="ISL_CNN_LSTM")


if __name__ == "__main__":
    # Quick sanity check
    m = build_cnn_lstm(num_classes=26)   # 26-letter ISL alphabet as a demo
    m.summary()
