import os

import tensorflow as tf

# GPU configurations
from keras.layers import UpSampling1D

from TrialsOfNeuralVocalRecon.data_processing.convenience_tools import getRealData

GPU = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import traceback

import sys

sys.path.append('../')
from keras import Input
from keras.models import Model
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from TrialsOfNeuralVocalRecon.neural_models.neural_models_Maryam import simplified_resnet
from keras.optimizers import Adam
# repeatability
from numpy.random import seed

seed(14)
from tensorflow import set_random_seed

set_random_seed(14)

lat_dim = 256
depth = 3
epochs = 1000
sound_len = 23926
spike_len = 23926
# spikes (batch,23928) 23926*4 = 95704
# sounds (batch,95704)

filepath_spikes_train = './data/spikes_train.h5'
filepath_spikes_test = './data/spikes_test.h5'
filepath_stim_train = './data/input_train_3d.h5'
filepath_stim_test = './data/input_test_3d.h5'

generator_train_spk2snd = getRealData(lat_dim, filepath_spikes_train, filepath_stim_train,
                                      data_type='generated data_spk2snd')
generator_test_spk2snd = getRealData(lat_dim, filepath_spikes_test, filepath_stim_test,
                                     data_type='generated data_spk2snd')

generator_train_snd2snd = getRealData(lat_dim, filepath_stim_train, filepath_stim_train,
                                      data_type='generated data_snd2snd')
generator_test_snd2snd = getRealData(lat_dim, filepath_stim_test, filepath_stim_test,
                                     data_type='generated data_snd2snd')


# initialize shared weights
sound2latent_model = simplified_resnet((sound_len, 1), depth, lat_dim)
latent2sound_model = simplified_resnet((sound_len, lat_dim), depth, 1)
spike2latent_model = simplified_resnet((sound_len, 1), depth, lat_dim)


# define spike2sound
input_spike = Input((spike_len, 1))
# upsample to match the shape of the sound
upsampled = UpSampling1D(size=4)(input_spike)
latent_spike = spike2latent_model(upsampled)
output = latent2sound_model(latent_spike)
spike2sound = Model(input_spike, output)

# define sound2sound
input_sound = Input((sound_len, 1))
latent_sound = sound2latent_model(input_sound)
output = latent2sound_model(latent_sound)
sound2sound = Model(input_sound, output)

Adam = Adam(lr=0.00001)
spike2sound.compile(optimizer=Adam, loss='mse')
sound2sound.compile(optimizer=Adam, loss='mse')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='min', min_delta=0.0001,
                              cooldown=0, min_lr=0)

batch_size = 32
callbacks_snd2snd = []
logPath_snd2snd = 'data/logs/snd2snd/'
callbacks_snd2snd.append(TensorBoard(logPath_snd2snd))
callbacks_snd2snd.append(reduce_lr)

callbacks_spk2snd = []
logPath_spk2snd = 'data/logs/spk2snd/'
callbacks_spk2snd.append(TensorBoard(logPath_spk2snd))
callbacks_spk2snd.append(reduce_lr)

try:
    for _ in range(epochs):
        sound2sound.fit_generator(generator_train_snd2snd,
                                  steps_per_epoch=5,
                                  epochs=1,
                                  validation_data=generator_test_snd2snd,
                                  use_multiprocessing=False,
                                  shuffle=False,
                                  validation_steps=len(generator_test_snd2snd) / batch_size,
                                  verbose=1,
                                  workers=1,
                                  callbacks=callbacks_snd2snd)

        spike2sound.fit_generator(generator_train_spk2snd,
                                  steps_per_epoch=5,
                                  epochs=1,
                                  validation_data=generator_test_spk2snd,
                                  use_multiprocessing=False,
                                  shuffle=False,
                                  validation_steps=len(generator_test_spk2snd) / batch_size,
                                  verbose=1,
                                  workers=1,
                                  callbacks=callbacks_spk2snd)

except Exception:
    traceback.print_exc()
print('fitting done,saving model')
sound2sound.save('data/Model_resnet_snd2snd.h5')
spike2sound.save('data/Model_resnet_spk2snd.h5')
