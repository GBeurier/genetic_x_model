## Tensorflow models and callbacks
import math
import numpy as np

from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    Activation,
    SpatialDropout1D,
    BatchNormalization,
    Flatten,
    Dropout,
    Input,
    GRU,
    LSTM,
    Bidirectional,
    MaxPool1D,
    AveragePooling1D,
    SeparableConv1D,
    Add,
    GlobalAveragePooling1D,
    GlobalMaxPool1D,
    Concatenate, concatenate,
    DepthwiseConv1D,
    Permute,
    MaxPooling1D,
    LayerNormalization,
    MultiHeadAttention,
    SeparableConv1D,
    ConvLSTM1D,
    LocallyConnected1D,
    Multiply,
    UpSampling1D,
    Lambda,
    Reshape
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau


def cnn_softmax(nb_features):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=11, strides=1, activation='relu', input_shape=(nb_features, 1)))
    model.add(tf.keras.layers.SpatialDropout1D(0.3))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=11, strides=3, activation='relu'))
    model.add(tf.keras.layers.SpatialDropout1D(0.3))
    model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=9, strides=5, activation='relu'))
    model.add(tf.keras.layers.SpatialDropout1D(0.3))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=13, strides=5, activation='relu'))
    model.add(tf.keras.layers.SpatialDropout1D(0.3))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=9, strides=5, activation='softmax'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', metrics=['mae', 'mse'], optimizer='adam')
    return model


def cnn_bacon(nb_features):
    model = Sequential()
    model.add(Input(shape=(nb_features, 1)))
    model.add(SpatialDropout1D(0.08))
    model.add(Conv1D(filters=8, kernel_size=15, strides=5, activation="selu"))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=21, strides=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=5, strides=3, activation="elu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(16, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='mean_squared_error', metrics=['mae', 'mse'], optimizer='adam')
    return model


def cnn_decon(nb_features):
    model = Sequential()
    model.add(Input(shape=(nb_features, 1)))
    model.add(SpatialDropout1D(0.2))
    model.add(DepthwiseConv1D(kernel_size=3, padding="same", depth_multiplier=64, activation="relu"))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
    model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=9, strides=6, padding="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(loss='mean_squared_error', metrics=['mae', 'mse'], optimizer='adam')
    return model


def depth_res(nb_features):
    inputs = Input(shape=(nb_features, 1))
    x = SpatialDropout1D(0.2)(inputs)
    x = DepthwiseConv1D(kernel_size=3, strides=3, depth_multiplier=2, activation="relu")(x)
    x = DepthwiseConv1D(kernel_size=5, strides=3, activation="relu")(x)
    x = DepthwiseConv1D(kernel_size=5, strides=3, activation="relu")(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = MultiHeadAttention(key_dim=11, num_heads=4, dropout=0.1)(x, x)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)

    outputs = x
    model = Model(inputs, outputs)
    model.compile(loss='mean_squared_error', metrics=['mae', 'mse'], optimizer='adam')
    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # inputs = tf.cast(inputs, tf.float16)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    # res = tf.cast(res, tf.float16)
    return x + res


def transformer_decoder(inputs, encoder_outputs, head_size, num_heads, ff_dim, dropout=0):
    # Masked multi-head attention
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)
    attn_output = Dropout(dropout)(attn_output)
    res = attn_output + inputs

    # Encoder-Decoder attention
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(res, encoder_outputs)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)
    attn_output = Dropout(dropout)(attn_output)
    res = attn_output + res

    # Feedforward network
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res


def transformer_model(
    nb_features,
    head_size=16,
    num_heads=2,
    ff_dim=8,
    num_transformer_blocks=1,
    mlp_units=[32, 8],
    dropout=0.05,
    mlp_dropout=0.1,
):
    inputs = Input(shape=(nb_features, 1))
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(loss='mean_squared_error', metrics=['mae', 'mse'], optimizer='adam')
    return model


def transformer_encoder2(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res


def transformer_decoder2(inputs, encoder_outputs, head_size, num_heads, ff_dim, dropout=0):
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)
    attn_output = Dropout(dropout)(attn_output)
    res = attn_output + inputs

    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(res, encoder_outputs)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)
    attn_output = Dropout(dropout)(attn_output)
    res = attn_output + res

    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res


def CNN_transformer(nb_features):
    inputs = Input(shape=(nb_features, 1))

    # First CNN layer sequence
    x = Conv1D(filters=128, kernel_size=16, strides=4, activation='relu')(inputs)
    x = Conv1D(filters=64, kernel_size=8, strides=4, activation='relu')(x)
    x = MaxPooling1D(pool_size=4, strides=2)(x)

    # Ensuring the output is reshaped appropriately for the Transformer
    feature_dim = x.shape[-1] * x.shape[-2]  # Calculate the total number of features after flattening
    x = Reshape((-1, feature_dim))(x)  # Reshape to maintain only batch size dimension dynamically

    # Transformer layers
    x = transformer_encoder2(x, head_size=128, num_heads=4, ff_dim=64, dropout=0.1)
    x = transformer_decoder2(x, x, head_size=128, num_heads=4, ff_dim=64, dropout=0.1)

    # Flatten and proceed to final dense layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Reshape((512, 1))(x)
    # x = Conv1D(filters=256, kernel_size=8, activation='relu')(x)
    x = Conv1D(filters=128, kernel_size=8, activation='relu')(x)
    x = Conv1D(filters=32, kernel_size=8, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(32, activation='sigmoid')(x)

    # Final output layer
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=output)
    optimizr = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizr, loss='mean_squared_error', metrics=['mae', 'mse'])
    return model

# Optimized callback to manage model checkpoints in memory
class Auto_Save(Callback):
    best_weights = []

    def __init__(self, model_name, verbose = 0):
        super(Auto_Save, self).__init__()
        self.model_name = model_name
        self.best = np.Inf
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if np.less(current_loss, self.best):
            if self.verbose > 0:
                print("epoch", str(epoch).zfill(5), "loss", "{:.6f}".format(current_loss), "{:.6f}".format(self.best), " " * 10)
            self.best = current_loss
            self.best_epoch = epoch
            Auto_Save.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.verbose > 1:
            print("Saved best {0:6.4f} at epoch".format(self.best), self.best_epoch)
        self.model.set_weights(Auto_Save.best_weights)
        self.model.save_weights("models/" + self.model_name + ".hdf5")
        # self.model.save(self.model_name + ".h5")
        with open("models/" + self.model_name + "_summary.txt", "w") as f:
            with redirect_stdout(f):
                self.model.summary()


def scale_fn(x):
    # return 1. ** x
    return 1 / (2.0 ** (x - 1))

# Cyclical learning rate function
def clr(epoch):
    cycle_params = {
        "MIN_LR": 0.00001,
        "MAX_LR": 0.01,
        "CYCLE_LENGTH": 128,
    }
    MIN_LR, MAX_LR, CYCLE_LENGTH = (
        cycle_params["MIN_LR"],
        cycle_params["MAX_LR"],
        cycle_params["CYCLE_LENGTH"],
    )
    initial_learning_rate = MIN_LR
    maximal_learning_rate = MAX_LR
    step_size = CYCLE_LENGTH
    step_as_dtype = float(epoch)
    cycle = math.floor(1 + step_as_dtype / (2 * step_size))
    x = abs(step_as_dtype / step_size - 2 * cycle + 1)
    mode_step = cycle  # if scale_mode == "cycle" else step
    return initial_learning_rate + (maximal_learning_rate - initial_learning_rate) * max(0, (1 - x)) * scale_fn(mode_step)

