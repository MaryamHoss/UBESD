import time

import matplotlib.pyplot as plt
import numpy as np

from TrialsOfNeuralVocalRecon.data_processing.data_generators import Reconstruction_Generator, Prediction_Generator, \
    Random_Generator, CPC_Generator


def timeStructured():
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S", named_tuple)
    return time_string


def plot_predictions(sound2sound, spike2sound, generator_test_snd2snd, generator_test_spk2snd, ex):
    # test spike to sound

    batch_spk_test, batch_snd_test = generator_test_spk2snd.__getitem__()

    predicted_sound = spike2sound.predict_on_batch(batch_spk_test)
    one_spike, one_sound, one_predicted_sound = batch_spk_test[0], batch_snd_test[0], predicted_sound[0]

    fig, axs = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(one_spike)
    axs[0].set_title('spike')
    axs[1].plot(one_sound)
    axs[1].set_title('sound')
    axs[2].plot(one_predicted_sound)
    axs[2].set_title('predicted sound')

    fig_path = 'data/spk2snd.pdf'
    fig.savefig(fig_path, bbox_inches='tight')
    ex.add_artifact(fig_path)

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


def plot_losses(n2n_lh, k2n_lh, n2n_lh_cpc, k2n_lh_cpc, ex):
    # plot training losses
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(n2n_lh, label='n2n_lh')
    ax.plot(k2n_lh, label='k2n_lh')
    ax.plot(n2n_lh_cpc, label='n2n_lh_cpc')
    ax.plot(k2n_lh_cpc, label='k2n_lh_cpc')
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend()

    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    plot_filename = 'data/{}_train_losses.pdf'.format(random_string)
    fig.savefig(plot_filename, bbox_inches='tight')
    ex.add_artifact(plot_filename)


def plot_final_bar_plot(list_terms, final_scores, plot_filename):
    # compare the performance of the different approaches
    y_pos = np.arange(len(list_terms))

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.bar(y_pos, final_scores, align='center', alpha=0.5)
    # ax.xticks(y_pos, list_terms)
    ax.set_xticklabels(list_terms, fontdict=None, minor=False)

    ax.set_ylabel('MSE')
    ax.set_title('training type')

    fig.savefig(plot_filename, bbox_inches='tight')

def get_innvestigate_analyzers():
    file = open('innvestigate_analyzers.txt', 'r')
    analyzers = []
    for line in file:
        if ':' in line:
            idx = line.index(':')
            analyzer = line[:idx].replace(' ', '').replace(',', '').replace('\"', '')
            analyzers.append(analyzer)
    return analyzers

def getData(
        sound_shape=(3, 1),
        spike_shape=(3, 1),
        data_type='real_prediction',
        batch_size=128,
        terms=3, predict_terms=3):
    filepath_noisy_spikes_train = './data/spikes_noisy_train.h5'
    filepath_noisy_spikes_test = './data/spikes_noisy_test.h5'
    filepath_stim_noisy_train = './data/input_noisy_train.h5'
    filepath_stim_noisy_test = './data/input_noisy_test.h5'
    filepath_stim_clean_train = './data/input_clean_train.h5'
    filepath_stim_clean_test = './data/input_clean_test.h5'


    filepath_spikes_train = './data/spikes_train_windowed_normalized.h5'
    filepath_spikes_test = './data/spikes_test_windowed_normalized.h5'
    filepath_stim_train = './data/input_train_windowed_normalized.h5'
    filepath_stim_test = './data/input_test_windowed_normalized.h5'

    if data_type == 'real_prediction':

        generator_train_spk2snd = Prediction_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_spikes_train,
            filepath_output=filepath_stim_train,
            batch_size=batch_size)
        generator_test_spk2snd = Prediction_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_spikes_test,
            filepath_output=filepath_stim_test,
            batch_size=batch_size)

        generator_train_snd2snd = Prediction_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_stim_train,
            filepath_output=filepath_stim_train,
            batch_size=batch_size)
        generator_test_snd2snd = Prediction_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_stim_test,
            filepath_output=filepath_stim_test,
            batch_size=batch_size)

    elif data_type == 'denoising':

        generator_train_spk2snd = Prediction_Generator(filepath_input=filepath_noisy_spikes_train,
                                                       filepath_output=filepath_stim_clean_train,
                                                       batch_size=batch_size)
        generator_test_spk2snd = Prediction_Generator(filepath_input=filepath_noisy_spikes_test,
                                                      filepath_output=filepath_stim_clean_test,
                                                      batch_size=batch_size)

        generator_train_snd2snd = Prediction_Generator(filepath_input=filepath_stim_noisy_train,
                                                       filepath_output=filepath_stim_clean_train,
                                                       batch_size=batch_size)
        generator_test_snd2snd = Prediction_Generator(filepath_input=filepath_stim_noisy_test,
                                                      filepath_output=filepath_stim_clean_test,
                                                      batch_size=batch_size)


    elif data_type == 'real_reconstruction':

        generator_train_spk2snd = Reconstruction_Generator(filepath_input=filepath_spikes_train,
                                                           filepath_output=filepath_stim_train,
                                                           batch_size=batch_size)
        generator_test_spk2snd = Reconstruction_Generator(filepath_input=filepath_spikes_test,
                                                          filepath_output=filepath_stim_test,
                                                          batch_size=batch_size)

        generator_train_snd2snd = Reconstruction_Generator(filepath_input=filepath_stim_train,
                                                           filepath_output=filepath_stim_train,
                                                           batch_size=batch_size)
        generator_test_snd2snd = Reconstruction_Generator(filepath_input=filepath_stim_test,
                                                          filepath_output=filepath_stim_test,
                                                          batch_size=batch_size)

    elif data_type == 'cpc_prediction':

        generator_train_spk2snd = CPC_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_spikes_train,
            filepath_output=filepath_stim_train,
            batch_size=batch_size,
            terms=terms,
            predict_terms=predict_terms)
        generator_test_spk2snd = CPC_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_spikes_test,
            filepath_output=filepath_stim_test,
            batch_size=batch_size,
            terms=terms,
            predict_terms=predict_terms)

        generator_train_snd2snd = CPC_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_stim_train,
            filepath_output=filepath_stim_train,
            batch_size=batch_size,
            terms=terms,
            predict_terms=predict_terms)
        generator_test_snd2snd = CPC_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_stim_test,
            filepath_output=filepath_stim_test,
            batch_size=batch_size,
            terms=terms,
            predict_terms=predict_terms)

    elif data_type == 'random':

        generator_train_spk2snd = Random_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_spikes_train,
            filepath_output=filepath_stim_train,
            batch_size=batch_size)
        generator_test_spk2snd = Random_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_spikes_test,
            filepath_output=filepath_stim_test,
            batch_size=batch_size)

        generator_train_snd2snd = Random_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_stim_train,
            filepath_output=filepath_stim_train,
            batch_size=batch_size)
        generator_test_snd2snd = Random_Generator(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_stim_test,
            filepath_output=filepath_stim_test,
            batch_size=batch_size)

    else:
        raise NotImplementedError

    return generator_train_spk2snd, generator_test_spk2snd, generator_train_snd2snd, generator_test_snd2snd


if __name__ == '__main__':
    get_innvestigate_analyzers()