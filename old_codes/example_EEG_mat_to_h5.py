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

if not os.path.isfile(NORMALIZEDDIR + '/eegs.h5'):
    if not os.path.isfile(EXAMPLE_EEG):
        print('Data is not located where expected. '
              'These paths should exist:\n\t{}\n\t{}'.format(EXAMPLE_EEG, EXAMPLE_SOUND_1))
        exit()

    if not os.path.isdir(NORMALIZEDDIR):
        os.mkdir(NORMALIZEDDIR)

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
    
    for subject in range(1, n_subjects + 1):
        print('Subject {} of {}'.format(subject, n_subjects))
        for i in tqdm(range(1, 31)):
            try:
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

                time_steps = [referenced_eeg.shape[0], sound_1.shape[0], sound_2.shape[0], min_len]
                min_len = int(np.min(time_steps))

                eegs.append(referenced_eeg[np.newaxis])
                noisy.append(np.squeeze(sound_mix)[np.newaxis])
                clean_sound = sound_1 if subject <= 17 else sound_2
                clean.append(np.squeeze(clean_sound)[np.newaxis])

            except Exception as e:
                print(e)

    data = dict()
    data['eegs'] = np.concatenate([m[:, :min_len] for m in eegs], axis=0)
    data['noisy'] = np.concatenate([m[:, :min_len, np.newaxis] for m in noisy], axis=0)
    data['clean'] = np.concatenate([m[:, :min_len, np.newaxis] for m in clean], axis=0)

    # shuffled indices
    n_samples = data['eegs'].shape[0]
    p = np.random.permutation(n_samples)

    print()
    for k in data.keys():
        # normalize
        max_val = np.max(np.abs(data[k]))
        data[k] /= max_val

        # shuffle
        data[k] = data[k][p]

        print('{}.shape:  '.format(k), data[k].shape)

        splits = np.split(data[k], [int(.6 * n_samples), int(.8 * n_samples)])
        split_names = ['train', 'validate', 'test']
        for name, split in zip(split_names, splits):
            print('{}.shape:  '.format(name), split.shape)
            f = h5py.File(NORMALIZEDDIR + '/{}_{}.h5'.format(k, name), 'w')
            f.create_dataset('{}_{}'.format(k, name), data=split)
            f.close()
