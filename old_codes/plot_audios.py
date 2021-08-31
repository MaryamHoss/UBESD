"""
Experiment Information
Subjects 1-17 were instructed to attend to 'Twenty Thousand Leagues Under the Sea' (20000), played in the left ear
Subjects 18-33 were instructed to attend to 'Journey to the Centre of the Earth' (Journey), played in the right ear
"""

import os
import scipy.io
import numpy as np
from tqdm import tqdm
import h5py

CDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data']))
NORMALIZEDDIR = os.path.abspath(os.path.join(*[DATADIR, 'Cocktail Party', 'Normalized']))
EXAMPLE_EEG = os.path.join(*[DATADIR, 'Cocktail Party', 'EEG', 'Subject1', 'Subject1_Run1.mat'])
EXAMPLE_SOUND_1 = os.path.join(*[DATADIR, 'Cocktail Party', 'Stimuli', 'Envelopes', '20000', '20000_1_env.mat'])
EXAMPLE_SOUND_2 = os.path.join(*[DATADIR, 'Cocktail Party', 'Stimuli', 'Envelopes', 'Journey', 'Journey_1_env.mat'])

# create normalized data for Subject 2
n_runs = 30
n_subjects = 33
min_len = 1e10  # some trials are longer, pick the shortest length
eegs = []
noisy = []
clean = []

# FIXME:
# Subject 6 of 33 20 % |██ | 6 / 30[00:00 < 00:00, 29.92    it / s][Errno 2]
# No such file or directory: 'C:\\Users\\PlasticDiscobolus\\work\\TrialsOfNeuralVocalRecon\\data\\Cocktail Party\\EEG\\Subject6\\Subject6_Run8.mat'
# operands could not be broadcast together with shapes(128, 7681)(2, 1)

subject = 2
i = 5
print('Subject {} of {}'.format(subject, n_subjects))
eeg_i_path = os.path.join(*[DATADIR, 'Cocktail Party', 'EEG',
                            'Subject{}'.format(subject), 'Subject{}_Run{}.mat'.format(subject, i)])
sound_1_i_path = os.path.join(
    *[DATADIR, 'Cocktail Party', 'Stimuli', 'Envelopes', '20000', '20000_{}_env.mat'.format(i)])
sound_2_i_path = os.path.join(
    *[DATADIR, 'Cocktail Party', 'Stimuli', 'Envelopes', 'Journey', 'Journey_{}_env.mat'.format(i)])

mat = scipy.io.loadmat(eeg_i_path)
raw_eeg = mat['eegData']
mastoids = mat['mastoids']
mean_mastoids = np.mean(mastoids, axis=1, keepdims=True)
referenced_eeg = (raw_eeg - mean_mastoids)

mat = scipy.io.loadmat(sound_1_i_path)
sound_1 = mat['envelope']
mat = scipy.io.loadmat(sound_2_i_path)
sound_2 = mat['envelope']
sound_mix = sound_1 + sound_2

import matplotlib.pyplot as plt

plt.plot(sound_1)
plt.plot(sound_2)
plt.show()