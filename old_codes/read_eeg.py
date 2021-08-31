"""
paper of interest:
probably best tutorial: https://mne.tools/dev/auto_examples/decoding/plot_receptive_field_mtrf.html#sphx-glr-auto-examples-decoding-plot-receptive-field-mtrf-py

"""

import os
import numpy as np
from mne.externals.pymatreader import read_mat
import mne
import matplotlib.pyplot as plt

from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment

CDIR = os.path.dirname(os.path.realpath(__file__))
superCDIR = os.path.join(*[CDIR, '..'])
ex = CustomExperiment('re', base_dir=superCDIR, GPU=0, seed=14)

@ex.automain
def main():
    images_dir = os.path.join(*[CDIR, ex.observers[0].basedir, 'images'])

    file_name = '../data/Subject1_Run1.mat'
    mat_data = read_mat(file_name)
    data = mat_data['eegData'].T
    fs = mat_data['fs']
    montage = mne.channels.make_standard_montage('biosemi128')
    info = mne.create_info(ch_names=montage.ch_names, sfreq=fs, ch_types='eeg').set_montage(montage)

    raw = mne.io.RawArray(data, info)
    # plot = raw.plot(n_channels=128, title='Data from arrays',
    #         show=True, block=True)

    data, times = raw[:]

    pos_3d = montage._get_ch_pos()
    pos_2d = np.array([v[:2] for _, v in pos_3d.items()])


    # plot 4 evenly spaced timesteps
    n_frames = 4
    frames = [int(i) for i in np.linspace(0, len(times)-2, num=n_frames)]

    fig, axes = plt.subplots(1, n_frames)
    for i, ax in zip(frames, axes):
        mne.viz.plot_topomap(data[:, i], pos_2d, show=False, axes=ax)
        t = np.round(times[i], 1)
        ax.set_title('{}s'.format(t))

    fig.savefig(os.path.join(*[images_dir, 'eeg.png']))

    # Power line noise
    print(raw)
    fig = raw.plot_psd(tmax=np.inf, fmax=64, average=True)
    # add some arrows at 60 Hz and its harmonics:
    for ax in fig.axes[:2]:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (60, 120, 180, 240):
            idx = np.searchsorted(freqs, freq)
            ax.arrow(x=freqs[idx], y=psds[idx] + 18, dx=0, dy=-12, color='red',
                     width=0.1, head_width=3, length_includes_head=True)