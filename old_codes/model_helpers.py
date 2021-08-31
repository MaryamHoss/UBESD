import time

import numpy as np
from tensorflow.keras.layers import Input, Dropout, LeakyReLU, BatchNormalization, Conv1D, Multiply, Activation, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def timeStructured():
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S", named_tuple)
    return time_string


def build_autoencoder_convolutional_voltage_gated(activation_encode='relu', activation_spikes='relu',
                                                  activation_decode='relu', activation_all='tanh', learning_rate=0.001,
                                                  n_convolutions=4):
    filters = np.linspace(5, 100, n_convolutions).astype(int)
    c = filters[::-1]
    c_end = c[-1]
    c = c[0:c.shape[0] - 1]
    decay_rate = learning_rate / 150

    input_img = Input(shape=(31900, 1))
    encoded = input_img

    for n_filters in c:
        encoded = Conv1D(n_filters, 25, strides=1, activation=activation_encode, padding='causal')(encoded)
        encoded = LeakyReLU()(BatchNormalization()(encoded))
        encoded = Dropout(0.3)(encoded)
        # encoded = MaxPooling1D(2,padding='same')(encoded)
    encoded = Conv1D(c_end, 25, strides=1, activation=activation_all, padding='causal')(encoded)
    # encoded = MaxPooling1D(2, padding='same')(encoded)
    encoded = LeakyReLU()(BatchNormalization()(encoded))
    encoded = Dropout(0.3)(encoded)

    encoder_sound = Model(inputs=[input_img], outputs=encoded)
    # encoder_image.summary()
    rec_spk = Input(shape=(7975, 1))
    spikes = rec_spk
    spikes = UpSampling1D(size=4)(spikes)
    for n_filters in c:
        spikes = Conv1D(n_filters, 25, strides=1, activation=activation_spikes, padding='causal')(spikes)
        spikes = LeakyReLU()(BatchNormalization()(spikes))
        spikes = Dropout(0.3)(spikes)
        # spikes =  MaxPooling1D(2,padding='same')(spikes)
    spikes = Conv1D(c_end, 25, strides=1, activation=activation_all, padding='causal')(spikes)
    spikes = LeakyReLU()(BatchNormalization()(spikes))
    spikes = Dropout(0.3)(spikes)

    # spikes =  MaxPooling1D(2, padding='same')(spikes)
    encoder_spikes = Model(inputs=[rec_spk], outputs=spikes)
    # encoder_spikes.summary()

    gated = Multiply()([encoded, spikes])
    gated = Activation('softmax')(gated)
    gated = Multiply()([gated, encoded])

    # decoded =Average()([encoded,spikes])
    decoded = Conv1D(n_filters, 25, strides=1, activation=activation_encode, padding='causal')(gated)
    # encoder=Model(inputs=[input_img,rec_spikes], outputs =encoded)
    decoded = LeakyReLU()(decoded)
    for n_filters in filters:
        decoded = Conv1D(n_filters, 25, strides=1, activation=activation_decode, padding='causal')(decoded)
        decoded = LeakyReLU()(decoded)

        # decoded = UpSampling1D(2)(decoded)

    decoded = Conv1D(1, 25, strides=1, activation=activation_all, padding='causal')(decoded)
    decoded = Activation('tanh')(decoded)

    # define autoencoder
    autoencoder = Model(inputs=[input_img, rec_spk], outputs=decoded)
    autoencoder.summary()
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.95, epsilon=1e-03, decay=decay_rate, amsgrad=True)
    autoencoder.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])

    return autoencoder
