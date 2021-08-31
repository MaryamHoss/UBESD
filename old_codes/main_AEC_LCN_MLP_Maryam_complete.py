from __future__ import print_function

import os

GPU = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
from keras.models import load_model, Model
from keras.layers import Input, Dense, Conv2D, Dropout, Activation, concatenate
from keras.layers import Flatten
from keras.callbacks import TensorBoard
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import keras
from keras.optimizers import *
from keras.layers.advanced_activations import LeakyReLU, ELU
import scipy.io as sio
import numpy as np
import h5py
from data_preprocessing.convenience_tools import timeStructured
import traceback
from numpy.random import seed

seed(14)
from tensorflow import set_random_seed

set_random_seed(14)

# GPU configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def corr2(a, b):
    k = np.shape(a)
    H = k[0]
    W = k[1]
    c = np.zeros((H, W))
    d = np.zeros((H, W))
    e = np.zeros((H, W))

    # Calculating mean values
    AM = np.mean(a)
    BM = np.mean(b)

    # Calculating terms of the formula
    for ii in range(H):
        for jj in range(W):
            c[ii, jj] = (a[ii, jj] - AM) * (b[ii, jj] - BM)
            d[ii, jj] = (a[ii, jj] - AM) ** 2
            e[ii, jj] = (b[ii, jj] - BM) ** 2

    # Formula itself
    r = np.sum(c) / float(np.sqrt(np.sum(d) * np.sum(e)))
    return r


def corr2_mse_loss(a, b):
    a = tf.subtract(a, tf.reduce_mean(a))
    b = tf.subtract(b, tf.reduce_mean(b))
    tmp1 = tf.reduce_sum(tf.multiply(a, a))
    tmp2 = tf.reduce_sum(tf.multiply(b, b))
    tmp3 = tf.sqrt(tf.multiply(tmp1, tmp2))
    tmp4 = tf.reduce_sum(tf.multiply(a, b))
    r = -tf.divide(tmp4, tmp3)
    m = tf.reduce_mean(tf.square(tf.subtract(a, b)))
    rm = tf.add(r, m)
    return rm


class SimpleGenerator(keras.utils.Sequence):
    def __init__(self,
                 filepath_spikes,
                 filepath_stim,
                 batch_size=32):
        self.__dict__.update(filepath_spikes=filepath_spikes,
                             filepath_stim=filepath_stim,
                             batch_size=batch_size,
                             )

        self.batch_size = batch_size
        self.batch_index = 0
        self.count_lines_in_file()

        self.on_epoch_end()

    def __len__(self):
        # steps_per_epoch, if unspecified, will use the len(generator) as a number of steps.
        # hence this
        self.steps_per_epoch = int(np.floor(self.nb_lines / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.filepath_spikes, 'r')
        for key in f.keys():

            for line in range(len(f[key])):
                self.nb_lines += 1

    def __getitem__(self, index=0):
        # 'Generate one batch of data'
        # Generate data

        return self.batch_generation()

    def on_epoch_end(self):
        self.spikes_f = h5py.File(self.filepath_spikes, 'r')
        self.stim_f = h5py.File(self.filepath_stim, 'r')

    def batch_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size
        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1

        for key in self.spikes_f.keys():
            spikes_batch = self.spikes_f[key][batch_start:batch_stop, ::]  # [0:self.batch_size,::]
        for key in self.stim_f.keys():
            stim_batch = self.stim_f[key][batch_start:batch_stop, ::]  # [0:self.batch_size,::]

        return spikes_batch, stim_batch


logPath = 'data/logs/neural_trained/'
if not os.path.isdir(logPath): os.mkdir(logPath)
models_folder = 'data/trained_models/neural_trained/'
if not os.path.isdir(models_folder): os.mkdir(models_folder)
time_string = timeStructured()
logPath += time_string + '_log'

date = '2019-12-29-20-09-59_'

Decoder_name = './data/' + date + 'Decoder_val.h5'


def save_preds(encoded_preds, Decoder_name):
    print('Decoding and saving predicted features...')
    decoder = load_model(Decoder_name)
    vocoder_test_maxes = h5py.File('./data/vocoder_test_maxes.h5', 'r')
    decoded_preds = decoder.predict(encoded_preds)
    spec = decoded_preds[0].swapaxes(1, 0)
    max_spec_test = vocoder_test_maxes['max_spec_test'][:]
    for i in range(max_spec_test.shape[1]):
        spec[i, :] = spec[i, :] * [max_spec_test[:, i]]
    aper = -decoded_preds[3].swapaxes(1, 0) * vocoder_test_maxes['max_aper_test'][:]
    aperiodicity_test_r = np.load('./data/aperiodicity_test_r.npy')
    aper_all = np.concatenate((aper, aperiodicity_test_r), axis=0)
    f0 = decoded_preds[1].swapaxes(1, 0) * vocoder_test_maxes['max_f0_test'][:]
    vuv = decoded_preds[2].swapaxes(1, 0)
    sio.savemat(models_folder + 'Main_preds_test_AEC_LCN_MLP.mat',
                mdict={'spectrogram': spec, 'band_aperiodicity': aper_all, 'f0': f0, 'vuv': vuv, 'fs': 97656.25})
    print('Saving done.')


filepath_spikes_train = './data/windowed_train_concat_norm.h5'
filepath_spikes_test = './data/windowed_test_concat_norm.h5'

filepath_stim_train = './data/encoded_train.h5'
filepath_stim_test = './data/encoded_test.h5'

generator_train = SimpleGenerator(filepath_spikes=filepath_spikes_train,
                                  filepath_stim=filepath_stim_train,
                                  batch_size=32)
generator_test = SimpleGenerator(filepath_spikes=filepath_spikes_test,
                                 filepath_stim=filepath_stim_test,
                                 batch_size=32)

print('\nbuilding model')


def build_model(shp_in, shp_out):
    reg = .005  # 0.0005
    inputs = Input(shape=shp_in)
    # x = LocallyConnected2D(1, kernel_size=[5, 5], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(inputs)
    x = Conv2D(1, kernel_size=[5, 5], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg),
               name='first_conv_layer')(inputs)

    # x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = Dropout(.3)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))

    x = Conv2D(1, kernel_size=[3, 3], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg),
               name='second_conv_layer')(x)

    # x = LocallyConnected2D(1, kernel_size=[3, 3], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    # x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = Dropout(.3)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = Conv2D(2, kernel_size=[1, 1], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg),
               name='third_conv_layer')(x)
    # x = LocallyConnected2D(2, kernel_size=[1, 1], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.3)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    #    x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = Flatten()(x)

    x_MLP = Flatten()(inputs)
    # x_MLP = Dense(10,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
    x_MLP = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
    x_MLP = Dropout(.4)(ELU(alpha=1.0)(BatchNormalization()(x_MLP)))
    #    x_MLP = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x_MLP)))
    #   x_MLP = Dense(10,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
    x_MLP = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
    #    x_MLP = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x_MLP)))
    x_MLP = Dropout(.4)(ELU(alpha=1.0)(BatchNormalization()(x_MLP)))

    x = concatenate([x, x_MLP], axis=1)

    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    # x = Dense(10,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.4)(ELU(alpha=1.0)(BatchNormalization()(x)))
    #    x = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x)))

    # x = Dense(10,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.4)(ELU(alpha=1.0)(BatchNormalization()(x)))
    #    x = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x)))

    x = Dense(shp_out, kernel_initializer='he_normal')(x)
    coded_preds = Activation('tanh', name='coded_preds')(x)
    model = Model(inputs, coded_preds)
    return model


# Inits
epochs = 1000
batch_size = 32
adam = Adam(lr=0.00001)  # .0001)

shp_in = (900, 265, 1)
shp_out = (256)

print('\nbuilding model.')
model = build_model(shp_in, shp_out)

print('\ncompile model')

model.compile(loss=corr2_mse_loss, optimizer=adam)
callbacks = []
# callbacks.append(TensorBoard(logPath, histogram_freq=int(epochs / 20) + 1,
# write_graph=False, write_grads=False,
# write_images=False))
callbacks.append(TensorBoard(logPath))

print('fitting model')

try:
    history = model.fit_generator(generator_train,
                                  epochs=epochs,
                                  validation_data=generator_test,
                                  use_multiprocessing=False,
                                  shuffle=False,
                                  validation_steps=len(generator_test) / batch_size,
                                  verbose=1,
                                  workers=1,
                                  callbacks=callbacks)

    print('fitting done,saving model')
    model.save(models_folder + '/Model_Val_AEC_LCN_MLP.h5')

except Exception:
    traceback.print_exc()
