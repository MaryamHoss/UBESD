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
    config_dir = os.path.join(*[CDIR, ex.observers[0].basedir, '1'])
    images_dir = os.path.join(*[CDIR, ex.observers[0].basedir, 'images'])
    models_dir = os.path.join(*[CDIR, ex.observers[0].basedir, 'trained_models'])

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

    print('max eeg: ', np.max(np.abs(pos_2d)))

    mne.viz.plot_topomap(data[:, 2], pos_2d)
    plt.savefig(os.path.join(*[images_dir, 'eeg.png']))


    # for referencing, add 2 channels this way
    # https://mne.tools/dev/auto_examples/preprocessing/plot_find_ref_artifacts.html#sphx-glr-auto-examples-preprocessing-plot-find-ref-artifacts-py

    ref_info = mne.create_info(['LPA', 'RPA'], raw.info['sfreq'], 'stim')
    ref_raw = mne.io.RawArray(mat_data['mastoids'].T, ref_info)
    raw.add_channels([ref_raw], force_update_info=True)

    # and then this way
    # https://mne.tools/stable/auto_tutorials/preprocessing/plot_55_setting_eeg_reference.html

    raw.set_eeg_reference(ref_channels=['LPA', 'RPA'])

    data, times = raw[:]
    pos_2d = np.concatenate([pos_2d, np.array([[-79.32e-3, 0], [79.32e-3, 0]])])
    mne.viz.plot_topomap(data[:, 2], pos_2d)

    print(pos_2d)