import time, os, itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import wavfile
import tensorflow as tf

from TrialsOfNeuralVocalRecon.data_processing.data_generators import Prediction_Generator, Random_Generator, \
    CPC_Generator

CDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data']))
EEGNORMALIZEDDIR = os.path.abspath(os.path.join(*[DATADIR, 'Cocktail_Party', 'Normalized', 'ShortEEG']))


def timeStructured():
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S", named_tuple)
    return time_string



def get_innvestigate_analyzers():
    file = open('innvestigate_analyzers.txt', 'r')
    analyzers = []
    for line in file:
        if ':' in line:
            idx = line.index(':')
            analyzer = line[:idx].replace(' ', '').replace(',', '').replace('\"', '')
            analyzers.append(analyzer)
    return analyzers


def getDataPaths(data_type):
    filepath_spikes_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/spikes_train.h5'
    filepath_spikes_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/spikes_test.h5'
    filepath_stim_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_train.h5'
    filepath_stim_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_test.h5'

    filepath_noisy_spikes_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/spikes_noisy_train.h5'
    filepath_noisy_spikes_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/spikes_noisy_test.h5'
    filepath_stim_noisy_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_noisy_train.h5'
    filepath_stim_noisy_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_noisy_test.h5'
    filepath_stim_clean_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_clean_train.h5'
    filepath_stim_clean_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_clean_test.h5'

    filepath_spikes_train_spectrogram = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/spikes_train_normalized.h5'
    filepath_spikes_test_spectrogram = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/spikes_test_normalized.h5'
    filepath_stim_train_spectrogram = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_train_normalized.h5'
    filepath_stim_test_spectrogram = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_test_normalized.h5'

    filepath_noisy_spikes_train_spectrogram = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/spikes_noisy_train_normalized.h5'
    filepath_noisy_spikes_test_spectrogram = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/spikes_noisy_test_normalized.h5'
    filepath_stim_noisy_train_spectrogram = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_train_noisy_normalized.h5'
    filepath_stim_noisy_test_spectrogram = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_test_noisy_normalized.h5'
    filepath_stim_clean_train_spectrogram = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_train_clean_normalized.h5'
    filepath_stim_clean_test_spectrogram = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_test_clean_normalized.h5'

    data_paths = {}
    for set in ['train', 'val', 'test']:

        if 'eeg' in data_type:
            if 'denoising' in data_type:
                if 'FBC' in data_type:
                    data_paths['in1_{}'.format(set)] = os.path.join(
                        *[EEGNORMALIZEDDIR, 'neural_eeg', 'journey', 'noisy_{}.h5'.format(set)])
                    data_paths['in2_{}'.format(set)] = os.path.join(
                        *[EEGNORMALIZEDDIR, 'neural_eeg', 'journey', 'eegs_{}.h5'.format(set)])
                    data_paths['out_{}'.format(set)] = os.path.join(
                        *[EEGNORMALIZEDDIR, 'neural_eeg', 'journey', 'clean_{}.h5'.format(set)])

                else:
                    data_paths['in1_{}'.format(set)] = os.path.join(
                        *[EEGNORMALIZEDDIR, 'original_eeg', 'noisy_{}.h5'.format(set)])
                    data_paths['in2_{}'.format(set)] = os.path.join(
                        *[EEGNORMALIZEDDIR, 'original_eeg', 'eegs_{}.h5'.format(set)])
                    data_paths['out_{}'.format(set)] = os.path.join(
                        *[EEGNORMALIZEDDIR, 'original_eeg', 'clean_{}.h5'.format(set)])

            else:
                data_paths['in1_{}'.format(set)] = os.path.join(*[EEGNORMALIZEDDIR, 'clean_{}.h5'.format(set)])
                data_paths['in2_{}'.format(set)] = os.path.join(*[EEGNORMALIZEDDIR, 'eegs_{}.h5'.format(set)])
                data_paths['out_{}'.format(set)] = os.path.join(*[EEGNORMALIZEDDIR, 'clean_{}.h5'.format(set)])
        else:
            raise NotImplementedError

    return data_paths


"""
    elif not 'spectrogram' in data_type:
        if 'denoising' in data_type:
            data_paths['in1_train'] = filepath_stim_noisy_train
            data_paths['in2_train'] = filepath_noisy_spikes_train
            data_paths['out_train'] = filepath_stim_clean_train

            data_paths['in1_val'] = filepath_stim_noisy_test
            data_paths['in2_val'] = filepath_noisy_spikes_test
            data_paths['out_val'] = filepath_stim_clean_test

            data_paths['in1_test'] = filepath_stim_noisy_test
            data_paths['in2_test'] = filepath_noisy_spikes_test
            data_paths['out_test'] = filepath_stim_clean_test
        else:

            in1_train = filepath_stim_train
            in2_train = filepath_spikes_train
            out_train = filepath_stim_train

            in1_test = filepath_stim_test
            in2_test = filepath_spikes_test
            out_test = filepath_stim_test
            raise NotImplementedError
    else:
        if 'denoising' in data_type:
            in1_train = filepath_stim_noisy_train_spectrogram
            in2_train = filepath_noisy_spikes_train_spectrogram
            out_train = filepath_stim_clean_train_spectrogram

            in1_test = filepath_stim_noisy_test_spectrogram
            in2_test = filepath_noisy_spikes_test_spectrogram
            out_test = filepath_stim_clean_test_spectrogram
            raise NotImplementedError
        else:
            in1_train = filepath_stim_train_spectrogram
            in2_train = filepath_spikes_train_spectrogram
            out_train = filepath_stim_train_spectrogram

            in1_test = filepath_stim_test_spectrogram
            in2_test = filepath_spikes_test_spectrogram
            out_test = filepath_stim_test_spectrogram
            raise NotImplementedError
"""


def getData(
        sound_shape=(3, 1),
        spike_shape=(3, 1),
        data_type='real_prediction',
        batch_size=128,
        downsample_sound_by=3,
        terms=3, predict_terms=3):
    downsampled_sound_shape = (int(sound_shape[0] / downsample_sound_by), sound_shape[1])
    if not 'random' in data_type:
        data_paths = getDataPaths(data_type)

    if not any([i in data_type for i in ['cpc', 'random']]):

        generator_train = Prediction_Generator(
            filepath_input_first=data_paths['in1_train'],
            filepath_input_second=data_paths['in2_train'],
            filepath_output=data_paths['out_train'],
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            batch_size=batch_size,
            data_type=data_type,
            downsample_sound_by=downsample_sound_by)

        generator_test = Prediction_Generator(
            filepath_input_first=data_paths['in1_test'],
            filepath_input_second=data_paths['in2_test'],
            filepath_output=data_paths['out_test'],
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            batch_size=batch_size,
            data_type=data_type,
            downsample_sound_by=downsample_sound_by)

        generator_val = Prediction_Generator(
            filepath_input_first=data_paths['in1_val'],
            filepath_input_second=data_paths['in2_val'],
            filepath_output=data_paths['out_val'],
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            batch_size=batch_size,
            data_type=data_type,
            downsample_sound_by=downsample_sound_by)

    elif 'cpc' in data_type:

        generator_train = CPC_Generator(
            filepath_input_first=data_paths['in1_train'],
            filepath_input_second=data_paths['in2_train'],
            filepath_output=data_paths['out_train'],
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            batch_size=batch_size,
            terms=terms,
            predict_terms=predict_terms,
            data_type=data_type)

        generator_val = CPC_Generator(
            filepath_input_first=data_paths['in1_val'],
            filepath_input_second=data_paths['in2_val'],
            filepath_output=data_paths['out_val'],
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            batch_size=batch_size,
            terms=terms,
            predict_terms=predict_terms,
            data_type=data_type)

        if 'random' in data_type:
            generator_test = Random_Generator(
                sound_shape=sound_shape,
                spike_shape=spike_shape,
                batch_size=batch_size,
                data_type=data_type)
        else:
            generator_test = Prediction_Generator(
                filepath_input_first=data_paths['in1_test'],
                filepath_input_second=data_paths['in2_test'],
                filepath_output=data_paths['out_test'],
                sound_shape=sound_shape,
                spike_shape=spike_shape,
                batch_size=batch_size,
                data_type=data_type)

    elif 'random' in data_type and not 'cpc' in data_type and not 'spectrogram' in data_type:

        """
        generator_train, generator_val, generator_test = [Random_Generator(sound_shape=sound_shape,
                                                                           spike_shape=spike_shape,
                                                                           batch_size=batch_size,
                                                                           data_type=data_type,
                                                                           downsample_sound_by=downsample_sound_by)
                                                          for _ in range(3)]
        """

        data_type = 0 if data_type == 'noSpikes' else 1
        generator_train, generator_val, generator_test = [
            tf.data.Dataset.from_generator(Random_Generator,
                                           args=[sound_shape,
                                                 spike_shape,
                                                 batch_size,
                                                 data_type,
                                                 downsample_sound_by],
                                           output_types=((tf.float32, tf.float32), tf.float32),
                                           output_shapes=((
                                                              (batch_size, *downsampled_sound_shape),
                                                              (batch_size, *spike_shape)
                                                          ),
                                                          (batch_size, *downsampled_sound_shape))
                                           )
            for _ in range(3)]

    else:
        raise NotImplementedError

    return generator_train, generator_val, generator_test



if __name__ == '__main__':

    epochs = 2  # 15  # 75  # 3
    batch_size = 4  # for 5 seconds #16 for 2 seconds

    downsample_sound_by = 3  # choices: 3 and 10
    sound_len = 16*3 #87552  # 87040 for downsample by 10 #87552 for downsample sound by=3  # 87552  # insteead of88200  #2626560#2610860
    fs = 44100 / downsample_sound_by
    spike_len = 4 #256  # 7680 # 7679

    fusion_type = '_add'  ## choices: 1) _concatenate 2) _FiLM_v1 3) _FiLM_v2 4) _FiLM_v3
    exp_type = 'WithSpikes'  # choices: 1) noSpike 2) WithSpikes
    input_type = 'random_eeg_'  # choices: 1) denoising_eeg_ 2) denoising_eeg_FBC_ 3) real_prediction_ 4) random_eeg_
    data_type = input_type + exp_type + fusion_type
    n_channels = 128 if 'eeg' in data_type else 1

    generator_train, generator_val, generator_test = getData(sound_shape=(sound_len, 1),
                                                             spike_shape=(spike_len, n_channels),
                                                             data_type=data_type,
                                                             batch_size=batch_size,
                                                             downsample_sound_by=downsample_sound_by)

    print(generator_train.batch(batch_size))