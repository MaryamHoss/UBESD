import os

from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

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

    fig_path = 'model/spk2snd.pdf'
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

    fig_path = 'model/snd2snd.pdf'
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
    plot_filename = 'model/{}_train_losses.pdf'.format(random_string)
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



def plot_test(prediction, batch_input, batch_snd_out_test, exp_type, image_title, fig_path, batch_sample):
    if 'WithSpikes' in exp_type:
        one_sound = batch_input[0][batch_sample]
    elif 'noSpike' in exp_type:
        one_sound = batch_input[batch_sample]
    else:
        raise NotImplementedError

    one_sound_out, one_predicted_sound = batch_snd_out_test[batch_sample], prediction[batch_sample]

    fig, axs = plt.subplots(3)
    fig.suptitle(image_title)
    axs[0].plot(one_sound)
    axs[0].set_title('input sound')
    axs[1].plot(one_sound_out)
    axs[1].set_title('output sound')
    axs[2].plot(one_predicted_sound)
    axs[2].set_title('predicted sound')

    fig.savefig(fig_path, bbox_inches='tight')
    plt.close('all')



def save_wav(prediction, batch_input, batch_snd_out_test, exp_type, batch_sample, fs, images_dir):
    if 'WithSpikes' in exp_type:
        one_sound_in = batch_input[0][batch_sample]
    elif 'noSpike' in exp_type:
        one_sound_in = batch_input[batch_sample]
    else:
        raise NotImplementedError

    clean_sound_path = os.path.join(*[images_dir, 'clean_{}_{}.wav'.format(exp_type, batch_sample)])
    noisy_sound_path = os.path.join(*[images_dir, 'noisy_{}_{}.wav'.format(exp_type, batch_sample)])
    predicted_sound_path = os.path.join(*[images_dir, 'prediction_{}_{}.wav'.format(exp_type, batch_sample)])

    one_sound_out, one_predicted_sound = batch_snd_out_test[batch_sample], prediction[batch_sample]
    m = np.max(np.abs(one_sound_in))
    one_sound_in32 = (one_sound_in / m).astype(np.float32)
    wavfile.write(noisy_sound_path, int(fs), one_sound_in32)

    m = np.max(np.abs(one_sound_out))
    one_sound_out32 = (one_sound_out / m).astype(np.float32)
    wavfile.write(clean_sound_path, int(fs), one_sound_out32)

    m = np.max(np.abs(one_predicted_sound))
    one_predicted_sound32 = (one_predicted_sound / m).astype(np.float32)
    wavfile.write(predicted_sound_path, int(fs), one_predicted_sound32)

