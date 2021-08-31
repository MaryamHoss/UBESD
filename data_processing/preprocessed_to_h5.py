import mne, os, h5py, random
import numpy as np
from scipy.io.wavfile import read as read_wav
# from tqdm import tqdm
import tensorflow as tf
import os, sys

sys.path.append('../')
from TrialsOfNeuralVocalRecon.tools.nice_tools import rms_normalize

seed = 14
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# FIXME: agree on preferred namings with Maryam
# choose what type of data you want, on the right the possibilities, commented
n_splits = 30  # 1 30
preprocessing = 'fbc'  # , 'eeg', 'fbc', raw_eeg
seconds = int(60 / n_splits)

if seconds == 60:
    time_folder = '60s'
elif seconds == 2:
    time_folder = '2s'
else:
    NotImplementedError

#CDIR = os.path.dirname(os.path.realpath(__file__))
CDIR = 'C:/Users\hoss3301\work\TrialsOfNeuralVocalRecon\data_processing'
DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data', 'Cocktail_Party']))
TIMEDIR = os.path.join(*[DATADIR, 'Normalized', time_folder])
h5_DIR = os.path.join(*[TIMEDIR, preprocessing])

EEGDIR = os.path.join(*[DATADIR, 'EEG'])
FULLAUDIODIR = os.path.join(*[DATADIR, 'Stimuli', 'Full_Audio'])

if preprocessing == 'raw_eeg':
    PEEGDIR = os.path.join(*[DATADIR, 'preprocessed_EEG', 'RAW_EEG'])
elif preprocessing == 'eeg':
    PEEGDIR = os.path.join(*[DATADIR, 'preprocessed_EEG', 'EEG'])
elif preprocessing == 'fbc':
    PEEGDIR = os.path.join(*[DATADIR, 'preprocessed_EEG', 'FBC'])
else:
    raise NotImplementedError

for path in [TIMEDIR, h5_DIR, FULLAUDIODIR, PEEGDIR]:
    if not os.path.isdir(path):
        os.mkdir(path)

# detect which runs are missing for each subject

subjects = [s for s in os.listdir(EEGDIR) if not 'txt' in s]
subject_trials = {}
for s in subjects:
    subject_folder = os.path.join(EEGDIR, s)
    runs = [r.split('.')[0].split('_')[1].replace('Run', '') for r in os.listdir(subject_folder)]
    runs = sorted([int(r) for r in runs])
    subject_trials.update({s: runs})

# go through each subject and each trial
min_t_activity = 1e8
min_t_audio = 1e8
eegs = []
noisy = []
clean = []
unattended = []
subject_list = []
#subject_indices = [1, 2, 5, 7, 8, 13, 15, 17, 19, 20, 21,     #all subjects
                   #22, 23, 24, 25, 26, 28, 29, 30, 31]
#subject_indices = [19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31]  #journey
'''subject_indices=[i for i in range(1,6)]
subject_indices_2=[i for i in range(7,32)]
for x in subject_indices_2:
    print(x)
    subject_indices.append(x)
'''



for subject_i in subject_indices:
    print('Subject {}'.format(subject_i))
    set_filepath = PEEGDIR + r'/subject{}.set'.format(subject_i)  # for EEG


    epochs = mne.io.read_epochs_eeglab(set_filepath)
    raw = epochs._data
    events = epochs.events
    events_times = events[:, 0]
    data = raw[:]  # (29,128,7680)

    t_1 = 0
    for n, t in zip(subject_trials['Subject{}'.format(subject_i)],
                    range(1, np.shape(events_times)[0] + 1)):

        trial = np.transpose(data[t_1:t, :, :], (0, 2, 1))
        trial_reshape = np.concatenate(np.split(trial, n_splits, axis=1), axis=0)
        eegs.append(trial_reshape)

        t_1 = t
        audio_path_20000 = os.path.join(FULLAUDIODIR, r'20000/20000_{}.wav'.format(n))
        audio_path_Journey = os.path.join(FULLAUDIODIR, r'Journey/Journey_{}.wav'.format(n))
        _, twenty = read_wav(audio_path_20000)
        _, journey = read_wav(audio_path_Journey)
        #twenty = rms_normalize(twenty[0:2646000])
        twenty = (twenty[0:2646000])
        #journey = rms_normalize(journey[0:2646000])
        journey = (journey[0:2646000])
        clean_sound = twenty if subject_i <= 17 else journey
        unattended_sound = twenty if subject_i > 17 else journey

        clean_sound_reshape = np.concatenate(np.split(clean_sound[None], n_splits, axis=1), axis=0)
        unattended_sound = np.concatenate(np.split(unattended_sound[None], n_splits, axis=1), axis=0)

        clean.append(clean_sound_reshape)
        unattended.append(unattended_sound)
        subject_list.append(np.array([subject_i] * n_splits))

        noisysnd = (twenty + journey)[None]
        noisysnd_reshape = np.concatenate(np.split(noisysnd, n_splits, axis=1), axis=0)

        noisy.append(noisysnd_reshape)

        min_t_activity = min(min_t_activity, trial_reshape.shape[1])
        min_t_audio = min(min_t_audio, clean_sound_reshape.shape[1])

data = dict()
data['eegs'] = np.concatenate([m[:, :min_t_activity] for m in eegs], axis=0)
data['noisy'] = np.concatenate([m[:, :min_t_audio, None] for m in noisy], axis=0)
data['clean'] = np.concatenate([m[:, :min_t_audio, None] for m in clean], axis=0)
data['unattended'] = np.concatenate([m[:, :min_t_audio, None] for m in unattended], axis=0)
data['subjects'] = np.concatenate(subject_list, axis=0)

# shuffled indices
n_samples = data['eegs'].shape[0]
p = np.random.permutation(n_samples)

for k in data.keys():
    print()
    # shuffle
    data[k] = data[k][p]

    print('{}.shape:  '.format(k), data[k].shape)

    # the array is split in 2 locations: 0.6*n_samples and 0.8*n_samples
    splits = np.split(data[k], [int(.6 * n_samples), int(.8 * n_samples)])
    split_names = ['train', 'val', 'test']

    max_value = None
    for name, split in zip(split_names, splits):
        if name == 'train':
            max_value = np.max(np.abs(split))

        # normalize
        if not 'subjects' in k:
            split = split / max_value

        if name=='test':
            print('{}.shape:  '.format(name), split.shape)
            f = h5py.File(h5_DIR + '/{}_{}.h5'.format(k, name), 'w')
            f.create_dataset('{}_{}'.format(k, name), data=split)
            f.close()
            
        print('{}.shape:  '.format(name), split.shape)
        f = h5py.File(h5_DIR + '/{}_{}.h5'.format(k, name), 'w')
        f.create_dataset('{}_{}'.format(k, name), data=split)
        f.close()
