import tensorflow as tf
from tensorflow.keras import layers, Model


def build_cnn_multibranch(input_shapes, num_classes):
    inputs, branch_outputs = [], []

    for input_shape in input_shapes:
        T, F = input_shape
        inp = layers.Input(shape=input_shape)
        inputs.append(inp)

        if F >= 2:
            x = layers.Reshape((T, F, 1))(inp)
            x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.GlobalAveragePooling2D()(x)
        else:
            x = layers.Conv1D(32, 3, activation='relu', padding='same')(inp)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
            x = layers.GlobalAveragePooling1D()(x)

        branch_outputs.append(x)

    merged = layers.Concatenate()(branch_outputs) if len(branch_outputs) > 1 else branch_outputs[0]
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model
