# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:45:56 2020

@author: hoss3301
"""
import scipy.io as spio
import numpy as np
mat=spio.loadmat('C:/Users/hoss3301/work/deep_guinea_ears/data/data/data_rawData/in samples/Matlab Data/input.mat')
input_train=mat['input_train']  #shape: (95703,8)
input_test=mat['input_test']   #shape: (95703,3)

input_test=input_test[:,:,np.newaxis,np.newaxis] #shape: 95703,3,1,1,
list_test=[input_test]*20
input_test=np.concatenate(list_test,axis=2) #shape: 95703,3,20,1,

list_test=[input_test]*265
input_test=np.concatenate(list_test,axis=3)  ##shape: 95703,3,20,265




input_test=np.transpose(input_test,(2,3,1,0))  #shape: 20,265,3,95703

input_test_reshaped=np.zeros(shape=(20*265*3,95703))
input_test_reshaped=np.reshape(input_test,(20*265*3,95703))  #shape: 15900,95703

c=input_test_reshaped[0,:]

stride_ms=1
window_ms=25
sample_rate=97656.25

stride_size = int(0.001 * sample_rate * stride_ms)
window_size = int(0.001 * sample_rate * window_ms)

f,t, Sxx= signal.spectrogram(c,fs=sample_rate,window=signal.get_window('hann',window_size),noverlap=stride_size,nfft=1024)
