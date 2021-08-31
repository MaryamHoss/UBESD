import os, h5py, argparse
import numpy as np
from scipy.io.wavfile import read as read_wav
from TrialsOfNeuralVocalRecon.tools.nice_tools import rms_normalize
import scipy.io as sio
from GenericTools.StayOrganizedTools.download_utils import download_and_unzip



parser = argparse.ArgumentParser(description='KU leuven dataset')
# type = normalize or download, since in Compute Canada where you can normalize you can't download
# and viceversa
parser.add_argument('--type', default='normalize', type=str, help='main behavior')
parser.add_argument('--target_seconds', default=60, type=int, help='time_steps in the batch')
args = parser.parse_args()
# data_links = ['https://zenodo.org/record/3618205/files/ds-eeg-snhl.tar']

# https://zenodo.org/record/3618205#.X-D_RdhKjb2  # only envelopes for now
# https://zenodo.org/record/3997352#.X-D719hKjb0
# https://zenodo.org/record/1199011#.X-D8JNhKjb1

# dataset from the article: Selective auditory attention in normal-hearing and hearing-impaired listeners
# by Søren A. Fuglsang; Jonatan Märcher-Rørsted;  Torsten Dau;  Jens Hjortkjær

data_links = [
    'https://zenodo.org/record/3997352/files/preprocess_data.m',
    'https://zenodo.org/record/3997352/files/readme.txt',
    'https://zenodo.org/record/3997352/files/stimuli.zip',
]
data_links += ['https://zenodo.org/record/3997352/files/S{}.mat'.format(i) for i in range(1, 17)]

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'kuleuven'))
if not os.path.isdir(DATAPATH): os.mkdir(DATAPATH)

if args.type == 'download':
    download_and_unzip(data_links, DATAPATH)
SOUNDDIR = os.path.join(DATAPATH, 'stimuli')

# min eeg time is 6.483333333333333 mins = 388.9998 s
min_eeg_duration = 388.9998
target_seconds = args.target_seconds
n_splits = int(min_eeg_duration / target_seconds)  # 1 30
min_eeg_duration = n_splits*target_seconds
preprocessing = 'eeg'  # , 'eeg', 'fbc', raw_eeg
time_folder = '{}s'.format(target_seconds)

NORMALDIR = os.path.join(DATAPATH, 'Normalized')
TIMEDIR = os.path.join(NORMALDIR, time_folder)
h5_DIR = os.path.join(*[TIMEDIR, preprocessing])
for folder in [NORMALDIR, TIMEDIR, h5_DIR]:
    if not os.path.isdir(folder): os.mkdir(folder)


subject_indices = [1, 5]  # range(1, 17)
# subject_indices = range(1, 17)
test_subjects = [1, 2]  # np.arange(28, 34).tolist()
test_trials = [1, 4]
val_subjects = [3, 4]
val_trials = [2, 3]

# go through each subject and each trial
if args.type == 'normalize':
    data = {}
    for k_1 in ['eegs_', 'noisy_', 'clean_', 'unattended_', 'subjects_']:
        for k_2 in ['train', 'val', 'test']:
            data[k_1 + k_2] = []

    for subject_i in subject_indices:
        print('Subject {}'.format(subject_i))

        mat_s = os.path.join(DATAPATH, 'S{}.mat'.format(subject_i))
        matstruct_contents = sio.loadmat(mat_s)['trials']
        keys = matstruct_contents[0, 0].dtype.fields.keys()

        for t in range(20):
            val = matstruct_contents[0, t]
            if val['repetition'][0][0][0][0] == 0:
                part = val['part'][0][0][0][0]
                attended_track = val['attended_track'][0][0][0][0]
                tracks = [1, 2]
                tracks.remove(attended_track)
                unattended_track = tracks[0]

                attended_sound_name = 'part{}_track{}_dry.wav'.format(part, attended_track)
                unattended_sound_name = 'part{}_track{}_dry.wav'.format(part, unattended_track)
                afs, attended_sound = read_wav(os.path.join(SOUNDDIR, attended_sound_name))
                attended_sound = rms_normalize(np.copy(attended_sound).astype(float))
                ufs, unattended_sound = read_wav(os.path.join(SOUNDDIR, unattended_sound_name))
                unattended_sound = rms_normalize(np.copy(unattended_sound).astype(float))

                eeg = val['RawData'][0][0][0][0][1]

                attended_sound = attended_sound[:int(min_eeg_duration*afs)]
                unattended_sound = unattended_sound[:int(min_eeg_duration*ufs)]
                eeg = eeg[:int(min_eeg_duration*128)]

                eeg_reshape = np.concatenate(np.split(eeg[None], n_splits, axis=1), axis=0)
                attended_sound_reshape = np.concatenate(np.split(attended_sound[None], n_splits, axis=1), axis=0)
                unattended_sound_reshape = np.concatenate(np.split(unattended_sound[None], n_splits, axis=1), axis=0)
                noisysnd_reshape = np.concatenate(np.split((unattended_sound + attended_sound)[None], n_splits, axis=1), axis=0)
                subject_list = (np.array([subject_i] * n_splits))

                if subject_i in test_subjects or t in test_trials:
                    if subject_i in test_subjects:
                        print('this subject is in test subjects: {}'.format(subject_i))
                    elif t in test_trials:
                        print('this trial is in test trials: {}'.format(t))
                    else:
                        raise NotImplementedError

                    data['eegs_test'].append(eeg_reshape)
                    data['clean_test'].append(attended_sound_reshape[..., None])
                    data['noisy_test'].append(noisysnd_reshape[..., None])
                    data['unattended_test'].append(unattended_sound_reshape[..., None])
                    data['subjects_test'].append(subject_list)

                elif subject_i in val_subjects or t in val_trials:
                    if subject_i in val_subjects:
                        print('this subject is in val subjects: {}'.format(subject_i))
                    elif t in val_trials:
                        print('this trial is in val trials: {}'.format(t))
                    else:
                        raise NotImplementedError

                    data['eegs_val'].append(eeg_reshape)
                    data['clean_val'].append(attended_sound_reshape[..., None])
                    data['noisy_val'].append(noisysnd_reshape[..., None])
                    data['unattended_val'].append(unattended_sound_reshape[..., None])
                    data['subjects_val'].append(subject_list)

                else:
                    data['eegs_train'].append(eeg_reshape)
                    data['clean_train'].append(attended_sound_reshape[..., None])
                    data['noisy_train'].append(noisysnd_reshape[..., None])
                    data['unattended_train'].append(unattended_sound_reshape[..., None])
                    data['subjects_train'].append(subject_list)

    data_copy = data
    del data

    # get the train max for normalization
    train_maxes = {}
    for k in data_copy.keys():
        set = [s for s in ['train', 'val', 'test'] if s in k][0]
        train_string = k.replace(set, 'train')
        train_maxes[k] = np.max(np.abs(data_copy[train_string]))

    permutations = {}
    for k in data_copy.keys():
        set = [s for s in ['train', 'val', 'test'] if s in k][0]

        if not 'subject' in k:
            # normalize
            data_copy[k] = data_copy[k] / train_maxes[k]
            # concatenate
            data_copy[k] = np.concatenate(data_copy[k], axis=0)
        else:
            data_copy[k] = np.concatenate(data_copy[k], axis=0)

        print('{}.shape:  {}'.format(k, data_copy[k].shape))

        # shuffle
        if not set in permutations.keys():
            n_samples = data_copy[k].shape[0]
            permutations[set] = np.random.permutation(n_samples)
        data_copy[k] = data_copy[k][permutations[set]]

    for k in data_copy.keys():
        print('{}.shape:  '.format(k), data_copy[k].shape)
        f = h5py.File(h5_DIR + '/{}.h5'.format(k), 'w')
        f.create_dataset('{}='.format(k), data=data_copy[k])
        f.close()
