import os, shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment, ChooseGPU
from TrialsOfNeuralVocalRecon.data_processing.convenience_tools_oldGen import getData
from TrialsOfNeuralVocalRecon.tools.naswt import IWPJS
from TrialsOfNeuralVocalRecon.neural_models import build_conv_with_fusion

CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment('nas', base_dir=CDIR, seed=14)


@ex.config
def cfg():
    GPU = 0
    learning_rate = 1e-05

    epochs = 0
    batch_size = 16  # 8 for 5 seconds #16 for 2 seconds

    downsample_sound_by = 3  # choices: 3 and 10
    sound_len = 256 * 6  # 87552  # 87040 for downsample by 10 #87552 for downsample sound by=3  # 87552  # insteead of88200  #2626560#2610860
    fs = 44100 / downsample_sound_by
    spike_len = 256  # 7680 # 7679

    fusion_type = '_FiLM_v2'  ## choices: 1) _concatenate 2) _FiLM_v1 3) _FiLM_v2 4) _FiLM_v3
    # 5) _FiLM_v4 6) _choice 7) _add 8) _transformer_classic 9) _transformer_parallel 10) _transformer_stairs 11)'' for no spikes
    # 11) _transformer_crossed_stairs
    exp_type = 'WithSpikes'  # choices: 1) noSpike 2) WithSpikes
    # fusion_type = fusion_type if not exp_type == 'noSpike' else ''
    input_type = 'denoising_eeg_'  # choices: 1) denoising_eeg_ 2) denoising_eeg_FBC_ 3) real_prediction_ 4) random_eeg_
    # 5) real_reconstruction_ 6) denoising_ 7) cpc_prediction_ 8) real_prediction_eeg_
    data_type = input_type + exp_type + fusion_type
    load_model = False  # wether we start from a previously trained model
    n_channels = 128 if 'eeg' in data_type else 1

    n_networks = 100


@ex.automain
def main(exp_type, data_type, learning_rate, sound_len, spike_len, batch_size, n_channels, downsample_sound_by, GPU, fs,
         n_networks):
    exp_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    images_dir = os.path.join(*[exp_dir, 'images'])
    models_dir = os.path.join(*[exp_dir, 'trained_models'])
    path_best_model = os.path.join(*[models_dir, 'model_weights_{}_predict.h5'.format(exp_type)])
    other_dir = os.path.join(*[exp_dir, 'other_outputs'])
    metrics_filename = os.path.join(*[other_dir, 'metrics.json'])
    save_every = 10
    ChooseGPU(GPU)

    generators = getData(
        sound_shape=(sound_len, 1),
        spike_shape=(spike_len, n_channels),
        data_type=data_type,
        batch_size=batch_size,
        downsample_sound_by=downsample_sound_by)

    batch = generators['val'].__getitem__()
    del generators

    df = pd.DataFrame(
        columns=['model_name', 'n_params', 'score', 'activation_encode', 'activation_spikes',
                 'activation_decode', 'activation_all', 'n_convolutions', 'min_filters', 'max_filters',
                 'kernel_size', 'notes'])
    activations = ['relu', 'tanh', 'linear', 'softplus']
    model_name = 'build_conv_with_fusion'
    for i in tqdm(range(n_networks)):
        try:
            # define a model
            activation_encode = np.random.choice(activations)
            activation_spikes = np.random.choice(activations)
            activation_decode = np.random.choice(activations)
            activation_all = np.random.choice(activations)
            n_convolutions = np.random.choice(range(2, 6))
            min_filters = np.random.choice(range(5, 100))
            max_filters = np.random.choice(range(5, 100))
            kernel_size = int(np.random.choice(range(5, 50)))

            model = build_conv_with_fusion(
                learning_rate=learning_rate, sound_shape=(None, 1), spike_shape=(None, n_channels),
                downsample_sound_by=downsample_sound_by, data_type=data_type,
                activation_encode=activation_encode, activation_spikes=activation_spikes,
                activation_decode=activation_decode, activation_all=activation_all,
                n_convolutions=n_convolutions, min_filters=min_filters, max_filters=max_filters,
                kernel_size=kernel_size, )

            s = IWPJS(model, batch)
            results = [dict(
                model_name=model_name, n_params=model.count_params(), score=s,
                activation_encode=activation_encode, activation_spikes=activation_spikes,
                activation_decode=activation_decode, activation_all=activation_all,
                n_convolutions=n_convolutions, min_filters=min_filters, max_filters=max_filters,
                kernel_size=kernel_size)]
            small_df = pd.DataFrame(results)

        except Exception as e:
            results = [{'model_name': model_name, 'notes': e}]
            small_df = pd.DataFrame(results)

        df = df.append(small_df)
        if i % save_every == 0:
            df.to_csv(other_dir + r'/out_{}.csv'.format(i))
            print(df)
            df = pd.DataFrame(
                columns=['model_name', 'n_params', 'score', 'activation_encode', 'activation_spikes',
                         'activation_decode', 'activation_all', 'n_convolutions', 'min_filters', 'max_filters',
                         'kernel_size', 'notes'])

        try:
            tf.keras.backend.clear_session()
            del model
        except:
            pass
        # cuda.select_device(0)
        # cuda.close()

    df.to_csv(other_dir + r'/out.csv')



    df = pd.DataFrame(columns=['model_name', 'n_params', 'score', 'layers_sequence', 'depth', 'notes'])

    ds = [d for d in os.listdir(other_dir) if '.csv' in d]
    for d in ds:
        csv_path = os.path.join(*[other_dir, d])
        small_df = pd.read_csv(csv_path)
        df = df.append(small_df)

    df = df.sort_values(by=['score'], ascending=False)
    print(df.head(20))

    df.to_csv(other_dir + r'/all_sorted.csv')

    shutil.make_archive(exp_dir, 'zip', exp_dir)

    return
