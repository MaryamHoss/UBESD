import logging
import os
import pathlib
import sys
from time import strftime, localtime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

sys.path.append('../')

from TrialsOfNeuralVocalRecon.data_processing.convenience_tools import getData
from TrialsOfNeuralVocalRecon.neural_models.neural_models_Maryam import simplified_resnet

GPU = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


class CustomFileStorageObserver(FileStorageObserver):

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        if _id is None:
            # create your wanted log dir
            time_string = strftime('%Y_%m_%d_at_%H_%M_%S', localtime())
            timestamp = 'experiment-{}________'.format(time_string)
            options = '_'.join(meta_info['options']['UPDATE'])
            run_id = timestamp + options

            # update the basedir of the observer
            self.basedir = os.path.join(self.basedir, run_id)

            # and again create the basedir
            pathlib.Path(self.basedir).mkdir(exist_ok=True, parents=True)
        return super().started_event(ex_info, command, host_info, start_time, config, meta_info, _id)


ex = Experiment('cleaning_audio_with_brain')
# ex.observers.append(FileStorageObserver.create("experiments"))
ex.observers.append(CustomFileStorageObserver.create("experiments"))

# ex.observers.append(MongoObserver())
ex.captured_out_filter = apply_backspaces_and_linefeeds

# set up a custom logger
logger = logging.getLogger('mylogger')
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')

# attach it to the experiment
ex.logger = logger

CDIR = os.path.dirname(os.path.realpath(__file__))


@ex.config
def cfg():
    seed = 14

    # batch_size = 32
    lat_dim = 256  # 128
    depth = 8  # 5#3
    epochs = 3
    steps_per_epoch = 3900
    finetuning_epochs = 1
    finetuning_steps_per_epoch = 18000
    sound_len = 31900  # 95704 # prediction:95700 reconstruction:95704 #31900 for when we cut the data into three parts
    spike_len = 7975  # 23926 #prediction:23925 reconstruction 23926 #7975 for when it is divided by three
    data_type = 'real_prediction'  # 'real_reconstruction'  #  #'random'
    batch_size = 32  # 64
    n_filters = 25

    # spikes (batch,23925) 23925*4 = 95700
    # sounds (batch,95700)


@ex.automain
def main(seed, batch_size, data_type, epochs, steps_per_epoch, finetuning_epochs, lat_dim, sound_len, spike_len, depth, n_filters, _log):
    tf.set_random_seed(seed)
    np.random.seed(seed)

    ##############################################################
    #                    define models
    ##############################################################

    # initialize shared weights
    sound2latent_model = simplified_resnet((sound_len, 1), depth, n_filters, lat_dim)

    latent2sound_model = simplified_resnet((sound_len, lat_dim), depth, n_filters, 1)

   
    # define sound2sound
    input_sound = Input((sound_len, 1))
    latent_sound = sound2latent_model(input_sound)
    output = latent2sound_model(latent_sound)
    sound2sound = Model(input_sound, output)

    adam = Adam(lr=0.00001)
    sound2sound.compile(optimizer=adam, loss='mse')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='min', min_delta=0.0001,
                                  cooldown=0, min_lr=0)
    sound2sound.summary()
    callbacks_snd2snd = []
    logPath_snd2snd = 'data/logs/snd2snd/'
    callbacks_snd2snd.append(TensorBoard(logPath_snd2snd))
    callbacks_snd2snd.append(reduce_lr)

    ##############################################################
    #                    train
    ##############################################################
    # try:

    snd2snd_loss_hystory = []
    for _ in tqdm(range(epochs)):
        generator_train_spk2snd, generator_test_spk2snd, generator_train_snd2snd, generator_test_snd2snd = getData(
            data_type=data_type, lat_dim=lat_dim, batch_size=batch_size)

        for _ in tqdm(range(steps_per_epoch)):
            batch_snd_input_snd2snd, batch_snd_outpu_snd2snd = generator_train_snd2snd.__getitem__()

            snd2snd_loss = sound2sound.train_on_batch(batch_snd_input_snd2snd, batch_snd_outpu_snd2snd)

            snd2snd_loss_hystory.append(snd2snd_loss)

    _log.info(snd2snd_loss_hystory)

    # plot training losses
    plot_filename = 'data/train_losses.pdf'
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(snd2snd_loss_hystory, label='snd2snd')
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend()
    fig.savefig(plot_filename, bbox_inches='tight')
    ex.add_artifact(plot_filename)

    # except Exception:
    #    traceback.print_exc()
    print('fitting done,saving model')
    snd2snd_model_filepath = 'data/Model_resnet_snd2snd.h5'
    sound2sound.save(snd2snd_model_filepath)

    print('yes!')
    ex.add_artifact(snd2snd_model_filepath)

    ##############################################################
    #                    tests training
    ##############################################################

    # test sound to sound
    batch_snd_input, batch_snd_output = generator_test_snd2snd.__getitem__()

    predicted_sound = sound2sound.predict_on_batch(batch_snd_input)
    one_sound_input, one_sound_output, one_predicted_sound = batch_snd_input[0], batch_snd_output[0], predicted_sound[0]

    fig, axs = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(one_sound_input)
    axs[0].set_title('input sound')
    axs[1].plot(one_sound_output)
    axs[1].set_title('output sound')

    axs[2].plot(one_predicted_sound)
    axs[2].set_title('predicted sound')

    fig_path = 'data/snd2snd.pdf'
    fig.savefig(fig_path, bbox_inches='tight')
    ex.add_artifact(fig_path)



    ##############################################################
    #                    finetune
    ##############################################################
    snd2snd_loss_hystory_denoising = []

    for _ in tqdm(range(epochs)):
        generator_train_spk2snd_denoising, generator_test_spk2snd_denoising, generator_train_snd2snd_denoising, generator_test_snd2snd_denoising = getData(
            data_type='denoising', lat_dim=lat_dim, batch_size=batch_size)

        for _ in tqdm(range(steps_per_epoch)):

            batch_snd_input_snd2snd_denoising, batch_snd_outpu_snd2snd_denoising = generator_train_snd2snd_denoising.__getitem__()

            snd2snd_loss = sound2sound.train_on_batch(batch_snd_input_snd2snd_denoising, batch_snd_outpu_snd2snd_denoising)

            snd2snd_loss_hystory_denoising.append(snd2snd_loss)

        
    _log.info(snd2snd_loss_hystory_denoising)
    # plot training losses
    plot_filename = 'data/finetune_losses.pdf'
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(snd2snd_loss_hystory_denoising, label='snd2snd')
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend()
    fig.savefig(plot_filename, bbox_inches='tight')
    ex.add_artifact(plot_filename)


    # except Exception:
    #    traceback.print_exc()
    print('finetuning done,saving model')
    snd2snd_denoising_model_filepath = 'data/Model_resnet_snd2snd_finetuned.h5'
    sound2sound.save(snd2snd_denoising_model_filepath)

    print('yes!')
    ex.add_artifact(snd2snd_denoising_model_filepath)
    
    
    ##############################################################
    #                    tests finetuning
    ##############################################################

    # test sound to sound

    batch_input_snd_denoising, batch_output_snd_denoising = generator_test_snd2snd_denoising.__getitem__()

    predicted_sound_denoising = sound2sound.predict(batch_input_snd_denoising)
    one_sound_input_denoising, one_sound_output_denoising, one_predicted_sound_denoising = batch_input_snd_denoising[0], batch_output_snd_denoising[0], predicted_sound_denoising[0]

    fig, axs = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(one_sound_input_denoising)
    axs[0].set_title('input sound')
    axs[1].plot(one_sound_output_denoising)
    axs[1].set_title('output sound')
    axs[2].plot(one_predicted_sound_denoising)
    axs[2].set_title('predicted sound')

    fig_path = 'data/snd2snd_finetuned.pdf'
    fig.savefig(fig_path, bbox_inches='tight')
    ex.add_artifact(fig_path)
    

