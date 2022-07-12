import os, sys, shutil, json, time, copy
from datetime import timedelta

sys.path.append('../')

from tensorflow_addons.callbacks import TimeStopping
from UBESD.tools.AdaBelief import AdaBelief

import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from UBESD.tools.plot_tools import plot_history
from UBESD.tools.VeryCustomSacred import CustomExperiment, ChooseGPU
from UBESD.tools.utilities import timeStructured, setReproducible

from UBESD.neural_models import build_model
from UBESD.tools.plotting import one_plot_test
from UBESD.data_processing.data_collection import getData,getData_mes,getData_kuleuven
from tensorflow.keras.optimizers import Adam
from UBESD.tools.calculate_intelligibility import find_intel
from UBESD.tools.utils.losses import *
import tensorflow.keras.backend as K
import pickle
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
ex = CustomExperiment(random_string + '-mc-prediction', base_dir=CDIR, seed=14)


@ex.config
def cfg():
    GPU = 0
    learning_rate = 1e-05
    seed = 14
    epochs = 2
    batch_size = 8
    kernel_size=25
    

    fusion_type = '_FiLM_v1_initializer:orthogonal_unet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4'
    ## choices: 1) _concatenate 2) _FiLM_v1_orthogonal 3) _FiLM_v2 4) _FiLM_v3 5) _FiLM_v4
    # 6) _choice 7) _add 11)'' for no spikes

    exp_type = 'WithSpikes'  # choices: 1) noSpikes 2) WithSpikes

    input_type = 'denoising_eeg_'  # choices: 1) denoising_eeg_ 2) denoising_eeg_FBC_
    # 9) denoising_eeg_RAW_ # 10) kuleuven_denoising_eeg_
    data_type = input_type + exp_type + fusion_type
    prev=False
    if prev=True:
        exp_folder = '' #write the address to the previous model
        if 'WithSpikes' in exp_type:
            load_model = os.path.abspath(os.path.join(*[CDIR, 'experiments', exp_folder, 'trained_models',
                                                'model_weights_WithSpikes_predict.h5']))
        else:
            load_model = os.path.abspath(os.path.join(*[CDIR, 'experiments', exp_folder, 'trained_models',
                                                'model_weights_noSpikes_predict.h5']))

    n_channels = 64 if 'mes' in data_type else 64 if 'kuleuven' in data_type else 26 if 'reduced_ch' in data_type else 128

    optimizer = 'cwAdaBelief'  # adam #adablief
    activation = 'relu'
    downsample_sound_by = 1 if 'mes' in data_type else 1 if 'kuleuven' in data_type else 3#3  # choices: 3 and 10
    sound_len = 32000 if 'mes' in data_type else 16000 if 'kuleuven' in data_type else 87552  # 87552  # 87040 for downsample by 10 #87552 for downsample sound by=3  # 87552  # insteead of88200  #2626560#2610860
    fs = 8000 if 'mes' in data_type else 8000 if 'kuleuven' in data_type else 44100/downsample_sound_by
    spike_len = 256  # 7680 # 7679
    hours=33


def get_callbacks(hours,path_best_model, history_path, data_type, images_dir):
    checkpoint = ModelCheckpoint(path_best_model, monitor='val_loss', verbose=1, save_best_only=True)
    #save_weights_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20,min_delta=0.01)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, cooldown=1)
    csvlogger = tf.keras.callbacks.CSVLogger(history_path),

    callbacks = [
        checkpoint, earlystopping, csvlogger, reduce_lr,
        TimeStopping((hours-1)*3600, 1)  # 31 hours: 111600 s 21 hours: 75600 s #223200 62 hours #33: 115200
    ]  # , tensorboard]


    return callbacks


@ex.automain
def main(exp_type, data_type,
         learning_rate, epochs, sound_len, spike_len, batch_size, load_model,
         n_channels, downsample_sound_by, GPU, fs, testing, optimizer, activation, test_type, batch_size_test,
         sound_len_test, spike_len_test, batch_start, batch_stop, batch_step, seed,hours,kernel_size):
    exp_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    images_dir = os.path.join(*[exp_dir, 'images'])
    text_dir = os.path.join(*[exp_dir, 'text'])
    models_dir = os.path.join(*[exp_dir, 'trained_models'])
    path_best_model = os.path.join(*[models_dir, 'model_weights_{}_predict.h5'.format(exp_type)])
    path_best_optimizer = os.path.join(*[models_dir, 'optimizer_{}_predict.pkl'.format(exp_type)])
    other_dir = os.path.join(*[exp_dir, 'other_outputs'])

    starts_at, starts_at_s = timeStructured(False, True)

    ChooseGPU(GPU)
    setReproducible(seed)

    model = build_model(learning_rate=learning_rate,
                        kernel_size=kernel_size,
                        sound_shape=(None, 1),
                        spike_shape=(None, n_channels),
                        downsample_sound_by=downsample_sound_by,
                        data_type=data_type)

    # comment for now to run all the models with old structure
    # total_epochs = epochs * len(generators['train'])
    # print(total_epochs)
    # learning_rate = tf.keras.experimental.CosineDecay(learning_rate, decay_steps=int(4 * total_epochs / 5), alpha=.1)
    # learning_rate = AddWarmUpToSchedule(learning_rate, warmup_steps=total_epochs / 6)
    # optimizer = AdaBelief(learning_rate=learning_rate, clipnorm=1.0, weight_decay=.1)
    if optimizer == 'cwAdaBelief':
        opt = 'cwAdaBelief'
        optimizer = AdaBelief(learning_rate=learning_rate, weight_decay=.1, clipnorm=1.)
    elif optimizer == 'AdaBelief':
        opt = 'AdaBelief'
        optimizer = AdaBelief(learning_rate=learning_rate)
    else:
        opt = 'Adam'
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=si_sdr_loss, metrics=[si_sdr_loss, 'mse'])  # dummy_loss

    if not load_model is False:
        print('Loading weights from {}'.format(load_model))
        if "noiseinput" in data_type:
            model.load_weights(load_model)
            with open(path_best_optimizer, 'rb') as f:
                weight_values = pickle.load(f)
                model.optimizer.set_weights(weight_values)
        else:
            model = tf.keras.models.load_model(
                load_model,
                custom_objects={'si_sdr_loss': si_sdr_loss, 'AdaBelief': AdaBelief})

    model.summary()

    ##############################################################
    #                    train
    ##############################################################

    print("training model")



    history_path = other_dir + '/log.csv'
    callbacks = get_callbacks(hours,path_best_model, history_path, data_type, images_dir)

    if 'mes' in data_type:
        
        generators = getData_mes(sound_shape=(sound_len, 1),
                     spike_shape=(spike_len, n_channels),
                     sound_shape_test=(sound_len_test, 1),
                     spike_shape_test=(spike_len_test, n_channels),
                     data_type=data_type,
                     batch_size=batch_size,
                     downsample_sound_by=downsample_sound_by,
                     test_type=test_type)

        model.fit(generators['train'],
                  epochs=epochs,
                  validation_data=generators['test'],
                  callbacks=callbacks)
        
        
    elif 'kuleuven' in data_type:
    
        generators = getData_kuleuven(sound_shape=(sound_len, 1),
                     spike_shape=(spike_len, n_channels),
                     sound_shape_test=(sound_len_test, 1),
                     spike_shape_test=(spike_len_test, n_channels),
                     data_type=data_type,
                     batch_size=batch_size,
                     downsample_sound_by=downsample_sound_by,
                     test_type=test_type)
    
        model.fit(generators['train'],
                  epochs=epochs,
                  validation_data=generators['test'],
                  callbacks=callbacks)    
        
    else:
        generators = getData(sound_shape=(sound_len, 1),
                             spike_shape=(spike_len, n_channels),
                             sound_shape_test=(sound_len_test, 1),
                             spike_shape_test=(spike_len_test, n_channels),
                             data_type=data_type,
                             batch_size=batch_size,
                             downsample_sound_by=downsample_sound_by,
                             test_type=test_type)

        model.fit(generators['train'],
                  epochs=epochs,
                  validation_data=generators['val'],
                  callbacks=callbacks)

    print('training done, saving model')

    model.save(path_best_model)

    if epochs > 0:
        history_df = pd.read_csv(history_path)
        history_dict = {k: history_df[k].tolist() for k in history_df.columns.tolist()}
        history = lambda x: None
        history.history = history_dict
        plot_filename = os.path.join(*[images_dir, 'history.png'])
        plot_history(history, plot_filename, epochs)
        json_filename = os.path.join(*[other_dir, 'history.json'])
        history_jsonable = {k: np.array(v).astype(float).tolist() for k, v in history.history.items()}
        json.dump(history_jsonable, open(json_filename, "w"))

        # plot only validation curves
        history_dict = {k: history_df[k].tolist() if 'val' in k else [] for k in history_df.columns.tolist()}
        history.history = history_dict
        plot_filename = os.path.join(*[images_dir, 'history_val.png'])
        plot_history(history, plot_filename, epochs)

        del history, plot_filename

    path_model_plot = os.path.join(*[models_dir, 'model_plot.png'.format(exp_type)])

    tf.keras.utils.plot_model(model, path_model_plot)

    del callbacks
    del generators['train'], generators['test']

    shutil.copyfile(FILENAME, text_dir + '/' + os.path.split(FILENAME)[-1])
    shutil.copyfile(os.path.join(CDIR, 'neural_models', 'models_convolutional.py'),
                    text_dir + '/' + 'models_convolutional.py')
    shutil.copyfile(os.path.join(CDIR, 'neural_models', '__init__.py'),
                    text_dir + '/' + '__init__.py')


    shutil.make_archive(ex.observers[0].basedir, 'zip', exp_dir)




