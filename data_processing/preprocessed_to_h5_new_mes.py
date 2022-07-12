
import mne, os, h5py, random
import numpy as np
from scipy.io.wavfile import read as read_wav
import tensorflow as tf
import sys
import scipy.io as spio

sys.path.append('../')


seed = 14
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

n_splits = 12#9 # 1 30
preprocessing = 'eeg'  # , 'eeg', 'fbc', raw_eeg
seconds = int(48 / n_splits)

if seconds == 4:
    time_folder = '4s'

else:
    NotImplementedError


    
def rms_normalize(audio):
    rms = np.sqrt(np.mean((audio ** 2)))
    audio= audio / rms
    
    return audio




def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


CDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data', 'Mesgarani']))
TIMEDIR = os.path.join(*[DATADIR, 'Normalized', time_folder])
h5_DIR = os.path.join(*[TIMEDIR, preprocessing,'new'])

EEGAUDIODIR = os.path.join(*[DATADIR, 'EEGAUDIO'])

if preprocessing == 'raw_eeg':
    PEEGDIR = os.path.join(*[DATADIR, 'preprocessed_EEG', 'RAW_EEG'])
elif preprocessing == 'eeg':
    PEEGDIR = os.path.join(*[DATADIR, 'EEGAUDIO'])
elif preprocessing == 'fbc':
    PEEGDIR = os.path.join(*[DATADIR, 'preprocessed_EEG', 'FBC'])
else:
    raise NotImplementedError

for path in [TIMEDIR, h5_DIR, EEGAUDIODIR, PEEGDIR]:
    if not os.path.isdir(path):
        os.mkdir(path)

# detect which runs are missing for each subject


subjects_all = [s for s in os.listdir(EEGAUDIODIR) if not 'txt' in s]
subject_trials = {}
for s in subjects_all:
    subjects=s.split('_')[0].split('-')[1].replace('0', '') 

# go through each subject and each trial
min_t_activity = 1e8
min_t_audio = 1e8
# subject_indices = np.arange(1, 33).tolist()
subject_indices = np.arange(23,45)
#subject_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] #speaker specific


trial=np.arange(1,33).tolist()
random.Random(14).shuffle(trial)
test_trials = trial[0:3]


data = {}
for k_1 in ['eegs_', 'noisy_', 'clean_', 'unattended_', 'subjects_']:
    for k_2 in ['train', 'test']:
        data[k_1 + k_2] = []

max_val = None
for subject_i in subject_indices:
    print('Subject {}'.format(subject_i))
    set_filepath = PEEGDIR + r'/sub-0{}_EEG_AUDIO.mat'.format(subject_i)
    
    all_data=loadmat(set_filepath)
    audio_data=all_data['srdat']['aud_feature']
    eeg_data=all_data['srdat']['eeg_feature']
    eeg_all = eeg_data['eeg_tt']
    clean_sound_all = audio_data['target_tt']
    unattended_sound_all = audio_data['masker_tt']
    
    
    
    t_1 = 0
    for n, t in enumerate(range(1, np.shape(audio_data['target_tt'])[1] + 1)):
        
        
        
        eeg = np.transpose(eeg_all[:, :,t_1:t], (2, 0, 1))

        
        clean_sound=clean_sound_all[:,t_1]
        unattended_sound=unattended_sound_all[:,t_1]
        t_1 = t

        
        clean_sound = rms_normalize(clean_sound)
        unattended_sound = rms_normalize(unattended_sound)
        
        #eeg_reshape = np.concatenate(np.split(eeg[:,0:2304,:], n_splits, axis=1), axis=0)
        eeg_reshape = np.concatenate(np.split(eeg[:,0:3072,:], n_splits, axis=1), axis=0)
        clean_sound_reshape = np.concatenate(np.split(clean_sound[:384000,None], n_splits, axis=0), axis=1)
        clean_sound_reshape=np.transpose(clean_sound_reshape,(1,0))
        unattended_sound_reshape = np.concatenate(np.split(unattended_sound[:384000,None], n_splits, axis=0), axis=1)
        unattended_sound_reshape=np.transpose(unattended_sound_reshape,(1,0))
        noisysnd_reshape = clean_sound_reshape + unattended_sound_reshape
        subject_list = (np.array([subject_i] * n_splits))
        
        # trial_i = n[t - 1]
        trial_i = n
        if trial_i in test_trials:
            
            if trial_i in test_trials:
                print('this trial is in test trials: {}'.format(trial_i))
            else:
                raise NotImplementedError
            
            data['eegs_test'].append(eeg_reshape)
            data['clean_test'].append(clean_sound_reshape[..., None])
            data['noisy_test'].append(noisysnd_reshape[..., None])
            data['unattended_test'].append(unattended_sound_reshape[..., None])
            data['subjects_test'].append(subject_list)
        
 
        else:
            data['eegs_train'].append(eeg_reshape)
            data['clean_train'].append(clean_sound_reshape[..., None])
            data['noisy_train'].append(noisysnd_reshape[..., None])
            data['unattended_train'].append(unattended_sound_reshape[..., None])
            data['subjects_train'].append(subject_list)
        
        min_t_activity = min(min_t_activity, eeg_reshape.shape[1])
        min_t_audio = min(min_t_audio, clean_sound_reshape.shape[1])

min_t = None
data_copy = data

# get the train max for normalization
train_maxes = {}
for k in data.keys():
    set = [s for s in ['train', 'test'] if s in k][0]
    train_string = k.replace(set, 'train')
    train_maxes[k] = np.max(np.abs(data_copy[train_string]))

permutations = {}
for k in data.keys():
    set = [s for s in ['train', 'test'] if s in k][0]
    min_t = min_t_activity if 'eeg' in k else min_t_audio
    
    if not 'subject' in k:
        # normalize
        data_copy[k] = data_copy[k] / train_maxes[k]
        # concatenate
        data_copy[k] = np.concatenate([m for m in data_copy[k]], axis=0)
    else:
        data_copy[k] = np.concatenate(data_copy[k], axis=0)
    
    print('{}.shape:  {}'.format(k, data_copy[k].shape))

    # shuffle
    if not set in permutations.keys():
        n_samples = data_copy[k].shape[0]
        permutations[set] = np.random.permutation(n_samples)
    data_copy[k] = data_copy[k][permutations[set]]
print(h5_DIR)
for k in data_copy.keys():
    print('{}.shape:  '.format(k), data_copy[k].shape)
    f = h5py.File(h5_DIR + '/{}.h5'.format(k), 'w')
    f.create_dataset('{}='.format(k), data=data_copy[k])
    f.close()

