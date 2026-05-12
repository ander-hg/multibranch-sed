import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup: 0 → target_lr over warmup_steps, then constant."""
    def __init__(self, target_lr=1e-3, warmup_steps=400):
        super().__init__()
        self.target_lr    = float(target_lr)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.target_lr * tf.minimum(step / self.warmup_steps, 1.0)

    def get_config(self):
        return {'target_lr': self.target_lr, 'warmup_steps': self.warmup_steps}


def _positional_encoding(length, depth):
    positions = np.arange(length)[:, np.newaxis]
    div_term  = np.power(10000.0, np.arange(0, depth, 2) / depth)
    encoding  = np.zeros((length, depth), dtype=np.float32)
    encoding[:, 0::2] = np.sin(positions / div_term)
    encoding[:, 1::2] = np.cos(positions / div_term)
    return tf.cast(encoding, tf.float32)


def build_transformer_multibranch(input_shapes, num_classes,
                                   d_model=64, num_heads=4, num_layers=2,
                                   ff_dim=128, dropout=0.1):
    inputs, branch_outputs = [], []
    for input_shape in input_shapes:
        T, F = input_shape
        inp = layers.Input(shape=input_shape)
        inputs.append(inp)
        x = layers.Dense(d_model)(inp)
        x = x + _positional_encoding(T, d_model)
        for _ in range(num_layers):
            attn_out = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
            attn_out = layers.Dropout(dropout)(attn_out)
            x = layers.LayerNormalization()(x + attn_out)
            ff = layers.Dense(ff_dim, activation='relu')(x)
            ff = layers.Dropout(dropout)(ff)
            ff = layers.Dense(d_model)(ff)
            x = layers.LayerNormalization()(x + ff)
        x = layers.GlobalAveragePooling1D()(x)
        branch_outputs.append(x)
    merged = layers.Concatenate()(branch_outputs) if len(branch_outputs) > 1 else branch_outputs[0]
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LinearWarmup()),
        loss='binary_crossentropy', metrics=['AUC'])
    return model
