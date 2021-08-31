# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:27:43 2019

@author: hoss3301
"""
from work import DNNBrainSpeechRecon
from DNNBrainSpeechRecon.world import main
import numpy as np
import h5py, os
import scipy.io


h5_folder = '../data/h5_sounds/'
if not os.path.isdir(h5_folder): os.mkdir(h5_folder)   

dataPath = '../data/original/input.mat'
mat1 = scipy.io.loadmat(dataPath)
fs=97656.25

for k in ['train','test']:
    data = mat1['input_' + k]
    print(k, data.shape)
    
    
    #for i in range(data.shape[1]):
    #    data[:,i]=data[:,i]/(np.max(data[:,i]))
    
    x = np.concatenate([sound for sound in np.transpose(data)])

    vocoder = main.World()

    dat= vocoder.encode(fs, x, f0_method='dio', target_fs=32552,frame_period=20, allowed_range=0.2, is_requiem=True)
    
    print('')
    for key, _ in dat.items():
        if 'aperio' in key:
            where_are_NaNs = np.isnan(dat[key])
            dat[key][where_are_NaNs] = 0
        max_value = np.max(np.abs(dat[key]))
        dat[key] /= max_value 
        print(key, max_value)

    sp=dat['spectrogram']
    sp=sp.swapaxes(1,0)

    vuv=dat['vuv']
    vuv = np.expand_dims(vuv, axis=1)

    ap=dat['aperiodicity']
    #ap=ap[0,:]
    #ap=np.expand_dims(ap,axis=1)

    f0=dat['f0']
    f0 = np.expand_dims(f0, axis=1)

    temporal_positions=dat['temporal_positions']
    temporal_positions = np.expand_dims(temporal_positions, axis=1)    
    
    print(f0.shape, vuv.shape, ap.shape, sp.shape, temporal_positions.shape)
    
    hf = h5py.File(h5_folder + 'clean_guinea_sounds_' + k + '.h5', 'w')
    hf.create_dataset('f0', data=f0)
    hf.create_dataset('vuv', data=vuv)
    hf.create_dataset('aperiodicity', data=ap)
    hf.create_dataset('spectrogram', data=sp)
    hf.create_dataset('temporal_positions', data=temporal_positions)
    hf.close()
    
    
