# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:35:13 2021

@author: hoss3301
"""
import os,sys
sys.path.append('../')

import h5py as hp
import matplotlib.pyplot as plt
from TrialsOfNeuralVocalRecon.tools.calculate_intelligibility import find_intel
from TrialsOfNeuralVocalRecon.tools.utils.losses import *
import tensorflow as tf
import numpy as np


n_splits = 1  # 1 30
preprocessing = 'eeg'  # , 'eeg', 'fbc', raw_eeg
seconds = int(60 / n_splits)

if seconds == 60:
    time_folder = '60s'
    sound_len = 2626560 

elif seconds == 2:
    time_folder = '2s'
    sound_len = 87552  

else:
    NotImplementedError
    
name='2021-01-19--15-17-20--6488-mcp_'
optimizer = 'adam'
activation = 'relu'
CDIR = os.path.dirname(os.path.realpath(__file__))
#CDIR = 'C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/tools'
DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data', 'Cocktail_Party']))
TIMEDIR = os.path.join(*[DATADIR, 'Normalized', time_folder])
h5_DIR = os.path.join(*[TIMEDIR, preprocessing])



MODELDIR=os.path.abspath(os.path.join(*[CDIR, '..', 'experiments', name,'trained_models','model_weights_WithSpikes_predict.h5']))

file=hp.File(h5_DIR+'/noisy_test.h5','r')
snd=file['noisy_test'][:]
file.close()

snd=snd[:,0:sound_len,:]
snd=snd[:,::3,:]

file=hp.File(h5_DIR+'/eegs_test.h5','r')
eeg=file['eegs_test'][:]
file.close()
eeg=np.repeat(eeg,114,1)

#load clean sound
file=hp.File(h5_DIR+'/clean_test.h5','r')
clean=file['clean_test'][:]
file.close()
clean=clean[:,0:sound_len,:]
clean=clean[:,::3,:]
file.close()


if optimizer == 'AdaBelief':
    model = tf.keras.models.load_model(MODELDIR,
                                       custom_objects={'si_sdr_loss': si_sdr_loss, 'AdaBelief': AdaBelief})
elif activation == 'snake':
    model = tf.keras.models.load_model(MODELDIR, custom_objects={'si_sdr_loss': si_sdr_loss, 'snake': snake})
else:
    model = tf.keras.models.load_model(MODELDIR, custom_objects={'si_sdr_loss': si_sdr_loss})
    
np.shape(eeg)[0]    

prediction_all=[]

intel_matrix=[]    
for i in range(20):
    print(i)
    prediction=model.predict([snd[i:i+1,:,:],eeg[i:i+1,:,:]])
    prediction_all.append(prediction)
    prediction_concat=np.concatenate(prediction_all,axis=0)
    intel_etoi=find_intel(clean[i:i+1,:,:],prediction,metric='estoi')
    intel_matrix.append(intel_etoi)
    intel_matrix_concat=np.concatenate(intel_matrix,axis=0)

np.save(MODELDIR+'prediction',prediction_concat)
np.save(MODELDIR+'stoi_matrix',intel_matrix_concat)    
    
    
    
    
    
    

    
