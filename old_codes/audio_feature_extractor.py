import sys
sys.path.append('../')

sys.path.append('C:/Users/hoss3301/work/models/research/audioset/yamnet')

import numpy as np
import resampy
import tensorflow as tf
import h5py as hp
import params as yamnet_params
import yamnet as yamnet_model

path_yamnet_weights='C:/Users/hoss3301/work/models/research/audioset/yamnet/yamnet.h5'
params = yamnet_params.Params()
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights(path_yamnet_weights)
path_data='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/clean_train.h5'
path_data='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/clean_val.h5'
path_data='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/clean_test.h5'

path_data='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/noisy_train.h5'
path_data='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/noisy_val.h5'
path_data='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/noisy_test.h5'

sr=44100
file=hp.File(path_data,'r')
wav_data_all=file['clean_train'][:]
wav_data_all=file['clean_val'][:]
wav_data_all=file['clean_test'][:]

wav_data_all=file['noisy_train'][:]
wav_data_all=file['noisy_val'][:]
wav_data_all=file['noisy_test'][:]


wav_data_all = wav_data_all.astype('float32')
file.close()
features=[]

for i in range(0,np.shape(wav_data_all)[0]):
    print(i)
    
    
    waveform=wav_data_all[i,:,0]
    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
      waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
      waveform = resampy.resample(waveform, sr, params.sample_rate)
    # Predict YAMNet classes.
    scores, embeddings, spectrogram = yamnet(waveform)
    embeddings=np.array(embeddings)
    embeddings=embeddings[np.newaxis]
    features.append(embeddings)

f=np.concatenate(features,axis=0)


#for clean
wav_data_all = wav_data_all.astype('float32')
file.close()
features=[]

for i in range(0,np.shape(wav_data_all)[0]):
    print(i)
    
    
    waveform=wav_data_all[i,:,0]
    # Convert to mono and the sample rate expected by YAMNet.
    
    if sr != params.sample_rate:
      waveform = resampy.resample(waveform, sr, 8000)
      waveform=waveform[:,np.newaxis]
      waveform=waveform[np.newaxis]
    features.append(waveform)

f=np.concatenate(features,axis=0)


file=hp.File('C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/clean_train_8k.h5','w')
file=hp.File('C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/clean_val_8k.h5','w')
file=hp.File('C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/clean_test_8k.h5','w')

file=hp.File('C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/noisy_train_features.h5','w')
file=hp.File('C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/noisy_val_features.h5','w')
file=hp.File('C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc/noisy_test_features.h5','w')

file.create_dataset('clean_train',data=f)
file.create_dataset('clean_val',data=f)
file.create_dataset('clean_test',data=f)

file.create_dataset('noisy_train',data=f)
file.create_dataset('noisy_val',data=f)
file.create_dataset('noisy_test',data=f)

file.close()

