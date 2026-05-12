import tensorflow as tf
from tensorflow.keras import layers, Model


def inception_block_2d(x, filters):
    b1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    b2 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    b3 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(x)
    b4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    b4 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(b4)
    return layers.Concatenate()([b1, b2, b3, b4])


def inception_block_1d(x, filters):
    b1 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
    b2 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
    b3 = layers.Conv1D(filters, 5, padding='same', activation='relu')(x)
    b4 = layers.MaxPooling1D(3, strides=1, padding='same')(x)
    b4 = layers.Conv1D(filters, 1, padding='same', activation='relu')(b4)
    return layers.Concatenate()([b1, b2, b3, b4])


def build_inception_multibranch(input_shapes, num_classes):
    inputs, branch_outputs = [], []

    for input_shape in input_shapes:
        T, F = input_shape
        inp = layers.Input(shape=input_shape)
        inputs.append(inp)

        if F >= 2:
            x = layers.Reshape((T, F, 1))(inp)
            x = inception_block_2d(x, 16)
            x = layers.MaxPooling2D((2, 2), padding='same')(x)
            x = inception_block_2d(x, 32)
            x = layers.GlobalAveragePooling2D()(x)
        else:
            x = inception_block_1d(inp, 32)
            x = layers.GlobalAveragePooling1D()(x)

        branch_outputs.append(x)

    merged = layers.Concatenate()(branch_outputs) if len(branch_outputs) > 1 else branch_outputs[0]
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model
