"""
paper of interest:
probably best tutorial: https://mne.tools/dev/auto_examples/decoding/plot_receptive_field_mtrf.html#sphx-glr-auto-examples-decoding-plot-receptive-field-mtrf-py

"""

from mne.externals.pymatreader import read_mat
import os
import mne
from mne.preprocessing import ICA
import numpy as np
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
    data = mat_data['eegData'].T[:, :22000] * 1e-6
    #n_timesteps = data.shape[1]
    print(data)

    # remove DC
    mean_data = np.mean(data, axis=1)[:, np.newaxis]
    data = data - mean_data

    fs = mat_data['fs']
    montage = mne.channels.make_standard_montage('biosemi128')
    info = mne.create_info(ch_names=montage.ch_names, sfreq=fs, ch_types='eeg').set_montage(montage)

    raw = mne.io.RawArray(data, info)
    raw.info['bads'].append('A22')

    # cropping the raw object to just three seconds for easier plotting
    #raw.crop(tmin=0, tmax=3).load_data()
    raw.plot()

    # Preprocessing following mne tutorial
    # https://mne.tools/dev/auto_tutorials/preprocessing/plot_40_artifact_correction_ica.html#tut-artifact-ica

    # Filtering to remove slow drifts
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=1., h_freq=None)

    # Fitting and plotting the ICA solution
    ica = ICA(n_components=15, random_state=0)
    ica.fit(filt_raw)

    raw.load_data()
    fig = ica.plot_sources(raw)
    fig.show()
    ica.plot_components()

    # blinks
    exclusion_list = [0, 1, 2]
    ica.plot_overlay(raw, exclude=exclusion_list, picks='eeg')
    ica.plot_properties(raw, picks=exclusion_list)

    # Selecting ICA components manually

    ica.exclude = exclusion_list

    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)
    ica.apply(filt_raw)
    eeg_data_interp = filt_raw.copy().interpolate_bads(reset_bads=False)

    reconst_raw.plot()
    filt_raw.plot()
    raw.plot()
    eeg_data_interp.plot()
    ica.plot_components()
