import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment
from GenericTools.StayOrganizedTools.utils import email_results
from TrialsOfNeuralVocalRecon.data_processing.convenience_tools import getData, plot_losses, plot_final_bar_plot, \
    plot_predictions
from TrialsOfNeuralVocalRecon.neural_models.cpc_model import network_cpc_v2

GPU = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
config = tf.compat.v1.ConfigProto()  # tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)  # tf.Session(config=config)

CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment('guinea_cleaner_CPC', base_dir=CDIR, GPU=1,seed=14)


@ex.config
def cfg():
    # sound_len and spike_len -2 have to be divisible by all elements in list_terms
    sound_len = 31892 #31900 #31901  # 95798 - 2
    spike_len = 7966 #7976  # 23930 - 2
    epochs = 100
    steps_per_epoch = 10
    batch_size = 32
    code_size = 128
    lr = 1e-3
    list_terms = [-1, 0, 7, 14]  # 0 means no CPC, and -1 means only sound2sound
    color = True
    data_type = 'cpc_prediction'


@ex.capture
def training(
        sound_shape,
        spike_shape,
        terms,
        predict_terms,
        code_size,
        lr,
        batch_size,
        epochs,
        steps_per_epoch,
        data_type,
        _log):
    ##############################################################
    #                    Load Model
    ##############################################################

    sound2sound, spike2sound, sound2sound_cpc, spike2sound_cpc = network_cpc_v2(
        sound_shape=sound_shape,
        spike_shape=spike_shape,
        terms=terms,
        predict_terms=predict_terms,
        code_size=code_size,
        learning_rate=lr)

    ##############################################################
    #                    train
    ##############################################################

    # k for spike, n for sound, lh for loss history
    k2n_lh = []
    n2n_lh = []
    k2n_lh_cpc = []
    n2n_lh_cpc = []
    for _ in tqdm(range(epochs)):

        data_type = 'cpc_prediction'
        gtrain_spk2snd_cpc, _, gtrain_snd2snd_cpc, _ = getData(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            data_type=data_type, batch_size=batch_size,
            terms=terms, predict_terms=predict_terms)

        data_type = 'real_prediction' #'random'  # 'real_prediction' #
        gtrain_spk2snd, gtest_spk2snd, gtrain_snd2snd, gtest_snd2snd = getData(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            data_type=data_type, batch_size=batch_size,
            terms=terms, predict_terms=predict_terms)

        for _ in tqdm(range(steps_per_epoch)):

            if terms > 0:
                # cpc training
                batch_spk_spk2snd, batch_snd_spk2snd = gtrain_spk2snd_cpc.__getitem__()
                batch_snd_input_snd2snd, batch_snd_output_snd2snd = gtrain_snd2snd_cpc.__getitem__()

                snd2snd_loss = sound2sound_cpc.train_on_batch(batch_snd_input_snd2snd, batch_snd_output_snd2snd)
                spk2snd_loss = spike2sound_cpc.train_on_batch(batch_spk_spk2snd, batch_snd_spk2snd)

                n2n_lh_cpc.append(snd2snd_loss)
                k2n_lh_cpc.append(spk2snd_loss)

            # non cpc training
            batch_spk_spk2snd, batch_snd_spk2snd = gtrain_spk2snd.__getitem__()
            batch_snd_input_snd2snd, batch_snd_output_snd2snd = gtrain_snd2snd.__getitem__()

            snd2snd_loss = sound2sound.train_on_batch(batch_snd_input_snd2snd, batch_snd_output_snd2snd)
            n2n_lh.append(snd2snd_loss)

            if not terms == -1:
                spk2snd_loss = spike2sound.train_on_batch(batch_spk_spk2snd, batch_snd_spk2snd)
                k2n_lh.append(spk2snd_loss)

    plot_losses(n2n_lh, k2n_lh, n2n_lh_cpc, k2n_lh_cpc, ex)
    plot_predictions(sound2sound, spike2sound, gtest_snd2snd, gtest_spk2snd, ex)

    final_score_n2n = n2n_lh[-1]
    if not terms == -1:
        final_score_k2n = k2n_lh[-1]
    else:
        final_score_k2n = -3.14
    # FIXME: implement saving Transformer
    return final_score_n2n, final_score_k2n


@ex.automain
def main(sound_len, spike_len, list_terms, _log):
    final_scores_n2n = []
    final_scores_k2n = []
    for terms in list_terms:
        _log.info('#####################################')
        _log.info('            terms {}'.format(terms))
        _log.info('#####################################')
        if terms in [-1, 0, 1]:
            spike_shape = (spike_len, 1)
            sound_shape = (sound_len, 1)
        else:
            spike_shape = (int(spike_len / terms), 1)
            sound_shape = (int(sound_len / terms), 1)
        final_score_n2n, final_score_k2n = training(sound_shape=sound_shape, spike_shape=spike_shape,
                                                    terms=terms, predict_terms=terms)
        final_scores_n2n.append(final_score_n2n)
        final_scores_k2n.append(final_score_k2n)

    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    plot_filename_n2n = 'data/{}_mse_bar_n2n.pdf'.format(random_string)
    plot_filename_k2n = 'data/{}_mse_bar_k2n.pdf'.format(random_string)
    plot_final_bar_plot(list_terms, final_scores_n2n, plot_filename_n2n)
    plot_final_bar_plot(list_terms, final_scores_k2n, plot_filename_k2n)

    ex.add_artifact(plot_filename_n2n)
    ex.add_artifact(plot_filename_k2n)

    sacred_dir = os.path.join(*[CDIR, ex.observers[0].basedir, '1'])
    email_results(
        folders_list=[sacred_dir],
        name_experiment=' guinea noise ',
        receiver_emails=['manucelotti@gmail.com', 'm.hosseinit@yahoo.com'])
