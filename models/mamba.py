import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from models.transformer import LinearWarmup


class MambaBlock(layers.Layer):
    """Mamba selective state-space block (Gu & Dao, 2023), tf.scan implementation."""

    def __init__(self, d_model, d_state=8, d_conv=3, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = int(expand * d_model)
        self.in_proj  = layers.Dense(self.d_inner * 2, use_bias=False)
        self.conv1d   = layers.DepthwiseConv1D(d_conv, padding='valid', use_bias=True)
        self.causal_pad = layers.ZeroPadding1D((d_conv - 1, 0))
        self.x_proj   = layers.Dense(1 + 2 * d_state, use_bias=False)
        self.dt_proj  = layers.Dense(self.d_inner, use_bias=True)
        self.out_proj = layers.Dense(d_model, use_bias=False)

    def build(self, input_shape):
        self.log_A = self.add_weight(
            'log_A', shape=(self.d_inner, self.d_state), trainable=True,
            initializer=tf.keras.initializers.Constant(
                np.log(np.tile(np.arange(1, self.d_state + 1, dtype=np.float32),
                               (self.d_inner, 1)))))
        self.D = self.add_weight('D', shape=(self.d_inner,),
                                 initializer='ones', trainable=True)
        super().build(input_shape)

    def call(self, u):
        x, z = tf.split(self.in_proj(u), 2, axis=-1)
        x = x * tf.sigmoid(x)
        x = self.causal_pad(x)
        x = self.conv1d(x)
        x = x * tf.sigmoid(x)
        x_dbl = self.x_proj(x)
        dt_raw, B, C = tf.split(x_dbl, [1, self.d_state, self.d_state], axis=-1)
        dt    = tf.nn.softplus(self.dt_proj(dt_raw))
        A     = -tf.exp(self.log_A)
        A_bar = tf.exp(tf.einsum('btd,ds->btds', dt, A))
        B_bar = tf.einsum('btd,bts->btds', dt * x, B)
        h0    = tf.zeros([tf.shape(u)[0], self.d_inner, self.d_state], dtype=u.dtype)
        hs    = tf.scan(
            lambda h, ab: ab[0] * h + ab[1],
            (tf.transpose(A_bar, [1, 0, 2, 3]),
             tf.transpose(B_bar, [1, 0, 2, 3])),
            initializer=h0)
        hs = tf.transpose(hs, [1, 0, 2, 3])
        y  = tf.einsum('btds,bts->btd', hs, C) + self.D * x
        y  = y * (z * tf.sigmoid(z))
        return self.out_proj(y)


def build_mamba_multibranch(input_shapes, num_classes,
                             d_model=64, d_state=8, num_layers=2, dropout=0.1):
    inputs, branch_outputs = [], []
    for input_shape in input_shapes:
        inp = layers.Input(shape=input_shape)
        inputs.append(inp)
        x = layers.Dense(d_model)(inp)
        for _ in range(num_layers):
            residual = x
            x = layers.LayerNormalization()(x)
            x = MambaBlock(d_model, d_state=d_state)(x)
            x = layers.Dropout(dropout)(x)
            x = x + residual
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
