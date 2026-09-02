import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, ELU, Dropout, Add, Flatten, Dense)
from tensorflow.keras.regularizers import l2


NUM_CLASSES = 27
FILTERS = 32
DROPOUT_RATE = 0.30
L2_WEIGHT = 8e-4


def conv_bn_elu_dropout(x, kernel_size, dilation_rate):
    x = Conv1D(filters=FILTERS, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="causal")(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(DROPOUT_RATE)(x)
    return x

def residual_tcn_block(x, dilation_rate):
    residual = x

    x = conv_bn_elu_dropout(x, kernel_size=3, dilation_rate=dilation_rate)
    x = conv_bn_elu_dropout(x, kernel_size=3, dilation_rate=dilation_rate)

    if residual.shape[-1] != FILTERS:
        residual = Conv1D(FILTERS, kernel_size=1, padding="same")(residual)

    return Add()([x, residual])

def dense_block(x, units, name):
    x = Dense(units, kernel_regularizer=l2(L2_WEIGHT), name=f"{name}_dense")(x)
    x = BatchNormalization(name=f"{name}_bn")(x)
    x = ELU(name=f"{name}_elu")(x)
    x = Dropout(DROPOUT_RATE, name=f"{name}_dropout")(x)
    return x

def build_eedgenet(input_shape=(32, 10), num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape, name="eeg_features")

    # Initial two-layer causal block, dilation 1.
    x = conv_bn_elu_dropout(inputs, kernel_size=10, dilation_rate=1)
    x = conv_bn_elu_dropout(x, kernel_size=10, dilation_rate=1)

    # Dilated residual blocks.
    x = residual_tcn_block(x, dilation_rate=2)
    x = residual_tcn_block(x, dilation_rate=4)

    x = Flatten(name="tcn_flatten")(x)

    # Dense transformation block.
    x = dense_block(x, 512, "dense_512")
    representation_512 = x

    x = dense_block(x, 256, "dense_256")
    x = dense_block(x, 128, "dense_128")
    x = dense_block(x, 64, "dense_64")

    outputs = Dense(num_classes, activation="softmax", name="classifier")(x)

    model = Model(inputs=inputs, outputs=outputs, name="EEdGeNet")

    encoder_512 = Model(inputs=inputs, outputs=representation_512, name="EEdGeNet_512")

    return model, encoder_512

if __name__ == "__main__":
    model, encoder_512 = build_eedgenet()
    model.summary()
