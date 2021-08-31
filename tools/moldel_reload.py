# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:51:32 2020

@author: hoss3301
"""

import os, sys
import h5py as hp

sys.path.append('../')
from TrialsOfNeuralVocalRecon.neural_models import build_model

import tensorflow as tf
from TrialsOfNeuralVocalRecon.tools.utils.losses import si_sdr_loss,estoi_loss
sound_len=218880
#with spikes
exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/5-speaker specific/'
#exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/6/'
With_spike=exp_folder+'film3/trained_models/model_weights_WithSpikes_predict.h5'                #model_weights_WithSpikes_predict.h5'

test_path= 'C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/'
file=hp.File(test_path+'/noisy_test.h5','r')
snd=file['noisy_test'][:]
file.close()
file=hp.File(test_path+'/eegs_test.h5','r')
eeg=file['eegs_test'][:]
file.close()
snd=snd[:,0:sound_len,:]
snd=snd[:,::3,:]
model_old=tf.keras.models.load_model(With_spike, custom_objects={'si_sdr_loss': si_sdr_loss})
prediction_old_fimv3=model_old.predict([snd,eeg])
prediction_old_noSpike=model_old.predict(snd)

file=hp.File(test_path+'/clean_test.h5','r')
clean=file['clean_test'][:]
file.close()
clean=clean[:,0:sound_len,:]
clean=clean[:,::3,:]

import numpy as np

prediction=np.load(exp_folder+'film1/prediction.npy')
###
W_init = model_old.get_weights()
learning_rate = 1e-5
data_type='denoising_eeg_FBC_WithSpikes_FiLM_v1'
new_model=build_model(learning_rate=learning_rate,
                        sound_shape=(2626560, 1),
                        spike_shape=(7680, 128),
                        downsample_sound_by=3,
                        data_type=data_type)
file=hp.File(test_path+'/long/noisy_test.h5','r')
snd_long=file['noisy_test'][:]
file.close()
snd_long=snd_long[:,0:2626560,:]
file=hp.File(test_path+'/long/eegs_test.h5','r')
eeg_long=file['eegs_test'][:]
file.close()   
new_model.set_weights(W_init)
prediction = new_model.predict([snd_long[:10,::3,:],eeg_long[:10,:,:]])







exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/5-speaker specific/'
No_spike=exp_folder+'no_spike/trained_models/model_weights_noSpikes_predict.h5'
test_path= 'C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/'
file=hp.File(test_path+'/noisy_test.h5','r')
snd=file['noisy_test'][:]
file.close()

snd=snd[:,0:sound_len,:]
model_old_noSpike=tf.keras.models.load_model(No_spike, custom_objects={'si_sdr_loss': si_sdr_loss})
prediction_old_noSpike=model_old_noSpike.predict(snd[:,::3,:])

file=hp.File(test_path+'/clean_test.h5','r')
clean=file['clean_test'][:]
file.close()
clean=clean[:,0:sound_len,:]
clean=clean[:,::3,:]


prediction=np.load(exp_folder+'no_spike/prediction.npy')



























