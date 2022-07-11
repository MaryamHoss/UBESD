# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:09:47 2022

@author: hoss3301
"""

import numpy as np
import matplotlib.pyplot as plot
from scipy.io import wavfile
import os
import h5py as hp
import requests

samplingFrequency=14700


path='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments'

prediction_UBESD_EEG=np.load(os.path.join(*[path,'prediction_UBESD_EEG.npy']))
prediction_UBESD_EEG=prediction_UBESD_EEG[0,:,0]
prediction_BESD_EEG=np.load(os.path.join(*[path,'prediction_BESD_EEG.npy']))
prediction_BESD_EEG=prediction_BESD_EEG[0,:,0]
prediction_BESD_FBC=np.load(os.path.join(*[path,'prediction_BESD_FBC.npy']))
prediction_BESD_FBC=prediction_BESD_FBC[0,:,0]
prediction_UBESD_FBC=np.load(os.path.join(*[path,'prediction_UBESD_FBC.npy']))
prediction_UBESD_FBC=prediction_UBESD_FBC[0,:,0]

file=hp.File(os.path.join(*[path,'clean.h5']),'r')
clean=file['clean_test='][:]
clean=clean[:,::3,:]
clean=clean[0,0:291840,0]
file.close()
file=hp.File(os.path.join(*[path,'noisy.h5']),'r')
noisy=file['noisy_test='][:]
noisy=noisy[:,::3,:]
noisy=noisy[0,0:291840,0]



clean_part=clean[0:5*14700]
noisy_part=noisy[0:5*14700]
prediction_UBESD_FBC_part=prediction_UBESD_FBC[0:5*14700]
prediction_BESD_FBC_part=prediction_BESD_FBC[0:5*14700]
prediction_BESD_EEG_part=prediction_BESD_EEG[0:5*14700]
prediction_UBESD_EEG_part=prediction_UBESD_EEG[0:5*14700]


spectrum_prediction_BESD_EEG=plot.specgram(prediction_BESD_EEG_part,Fs=samplingFrequency,NFFT=512)
spectrum_prediction_UBESD_EEG=plot.specgram(prediction_UBESD_EEG_part,Fs=samplingFrequency)
spectrum_prediction_BESD_FBC=plot.specgram(prediction_BESD_FBC_part,Fs=samplingFrequency)
spectrum_prediction_UBESD_FBC=plot.specgram(prediction_UBESD_FBC_part,Fs=samplingFrequency)


spectrum_noise=plot.specgram(noisy_part-clean_part,Fs=samplingFrequency)
spectrum_clean=plot.specgram(clean_part,Fs=samplingFrequency)
spectrum_noisy=plot.specgram(noisy_part,Fs=samplingFrequency)


spectrum_clean_mag=10*(np.log10(spectrum_clean[0]))
spectrum_noise_mag=10*(np.log10(spectrum_noise[0]))
spectrum_noisy_mag=10*(np.log10(spectrum_noisy[0]))
spectrum_prediction_BESD_EEG_mag=10*(np.log10(spectrum_prediction_BESD_EEG[0]))
spectrum_prediction_UBESD_EEG_mag=10*(np.log10(spectrum_prediction_UBESD_EEG[0]))
spectrum_prediction_BESD_FBC_mag=10*(np.log10(spectrum_prediction_BESD_FBC[0]))
spectrum_prediction_UBESD_FBC_mag=10*(np.log10(spectrum_prediction_UBESD_FBC[0]))

fig=plot.figure(figsize = (10,10))
im1=plot.imshow((spectrum_prediction_BESD_EEG_mag), cmap = 'seismic',origin='lower')#,alpha=0.9)
plot.title('BESD EEG')
fig.colorbar(im1,shrink=0.2)

fig=plot.figure(figsize = (10,10))
plot.imshow(spectrum_prediction_UBESD_EEG_mag, cmap = 'seismic',origin='lower')#,alpha=0.9)
plot.title('UBESD EEG')
fig.colorbar(im1,shrink=0.2)


fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(10, 10))
plot.imshow(spectrum_prediction_BESD_FBC_mag, cmap = 'seismic',origin='lower')#,alpha=0.9)
plot.title('BESD FBC')
fig.colorbar(im1,shrink=0.2,ax=ax)
ax.set_yticks(np.array([1,1.5,2,2.5]))

fig=plot.figure(figsize = (10,10))
plot.imshow(spectrum_prediction_UBESD_FBC_mag, cmap = 'seismic',origin='lower')#,alpha=0.9)
plot.title('UBESD FBC')
fig.colorbar(im1,shrink=0.2)

fig=plot.figure(figsize = (10,10))
plot.imshow(spectrum_clean_mag, cmap = 'seismic',origin='lower')#,alpha=0.9)
plot.title('clean')
fig.colorbar(im1,shrink=0.2)

fig=plot.figure(figsize = (10,10))
plot.imshow(spectrum_noise_mag, cmap = 'seismic',origin='lower')#,alpha=0.9)
plot.title('noise')
fig.colorbar(im1,shrink=0.2)

fig=plot.figure(figsize = (10,10))
plot.imshow(spectrum_noisy_mag, cmap = 'seismic',origin='lower')#,alpha=0.9)
plot.title('noisy')
fig.colorbar(im1,shrink=0.2)


plot.rcParams["font.family"] = "serif"

fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(10, 5))
spectrum_prediction_BESD_EEG=ax.specgram(prediction_BESD_EEG_part,Fs=samplingFrequency,NFFT=512,cmap = 'seismic')
ax.set_yticks(np.array([0,1000,3000,5000,7000]))
ax.set_xticks(np.array([0,1,2,3,4,5]))
ax.tick_params('y',labelsize=15)
ax.tick_params('x',labelsize=15)
plot.xlabel('time (s)')
ax.xaxis.label.set_size(15)
plot.ylabel('frequeny (Hz)')
ax.yaxis.label.set_size(15)
fig.colorbar(spectrum_prediction_BESD_EEG[3])

fig.savefig(os.path.join(path,'prediction_BESD_EEG.eps'),
            bbox_inches='tight',format='eps')


fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(10, 5))
spectrum_prediction_UBESD_EEG=ax.specgram(prediction_UBESD_EEG_part,Fs=samplingFrequency,NFFT=512,cmap = 'seismic')
ax.set_yticks(np.array([0,1000,3000,5000,7000]))
ax.set_xticks(np.array([0,1,2,3,4,5]))
ax.tick_params('y',labelsize=15)
ax.tick_params('x',labelsize=15)
plot.xlabel('time (s)')
ax.xaxis.label.set_size(15)
plot.ylabel('frequeny (Hz)')
ax.yaxis.label.set_size(15)
fig.colorbar(spectrum_prediction_UBESD_EEG[3])

fig.savefig(
            os.path.join(path,'prediction_UBESD_EEG.eps'),
            bbox_inches='tight',format='eps'
        )
            
fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(10, 5))
spectrum_prediction_BESD_FBC=plot.specgram(prediction_BESD_FBC_part,Fs=samplingFrequency,NFFT=512,cmap = 'seismic')
ax.set_yticks(np.array([0,1000,3000,5000,7000]))
ax.set_xticks(np.array([0,1,2,3,4,5]))
ax.tick_params('y',labelsize=15)
ax.tick_params('x',labelsize=15)
plot.xlabel('time (s)')
ax.xaxis.label.set_size(15)
plot.ylabel('frequeny (Hz)')
ax.yaxis.label.set_size(15)
fig.colorbar(spectrum_prediction_BESD_FBC[3])
fig.savefig(
            os.path.join(path,'prediction_BESD_FBC.eps'),
            bbox_inches='tight',format='eps'
        )
            

fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(10, 5))
spectrum_prediction_UBESD_FBC=plot.specgram(prediction_UBESD_FBC_part,Fs=samplingFrequency,NFFT=512,cmap = 'seismic')
ax.set_yticks(np.array([0,1000,3000,5000,7000]))
ax.set_xticks(np.array([0,1,2,3,4,5]))
ax.tick_params('y',labelsize=15)
ax.tick_params('x',labelsize=15)
plot.xlabel('time (s)')
ax.xaxis.label.set_size(15)
plot.ylabel('frequeny (Hz)')
ax.yaxis.label.set_size(15)
fig.colorbar(spectrum_prediction_UBESD_FBC[3])
fig.savefig(
            os.path.join(path,'prediction_UBESD_FBC.eps'),
            bbox_inches='tight',format='eps'
        )


fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(10, 5))
spectrum_noise=plot.specgram(noisy_part-clean_part,Fs=samplingFrequency,NFFT=512,cmap = 'seismic')
ax.set_yticks(np.array([0,1000,3000,5000,7000]))
ax.set_xticks(np.array([0,1,2,3,4,5]))
ax.tick_params('y',labelsize=15)
ax.tick_params('x',labelsize=15)
plot.xlabel('time (s)')
ax.xaxis.label.set_size(15)
plot.ylabel('frequeny (Hz)')
ax.yaxis.label.set_size(15)
fig.colorbar(spectrum_noise[3])
fig.savefig(
            os.path.join(path,'noise.eps'),
            bbox_inches='tight',format='eps'
        )


fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(10, 5))
spectrum_clean=plot.specgram(clean_part,Fs=samplingFrequency,NFFT=512,cmap = 'seismic')
ax.set_yticks(np.array([0,1000,3000,5000,7000]))
ax.set_xticks(np.array([0,1,2,3,4,5]))
ax.tick_params('y',labelsize=15)
ax.tick_params('x',labelsize=15)
plot.xlabel('time (s)')
ax.xaxis.label.set_size(15)
plot.ylabel('frequeny (Hz)')
ax.yaxis.label.set_size(15)
fig.colorbar(spectrum_clean[3])
fig.savefig(
            os.path.join(path,'clean.eps'),
            bbox_inches='tight',format='eps'
        )


fig, ax = plot.subplots(nrows=1, ncols=1, figsize=(10, 5))
spectrum_noisy=plot.specgram(noisy_part,Fs=samplingFrequency,NFFT=512,cmap = 'seismic')
ax.set_yticks(np.array([0,1000,3000,5000,7000]))
ax.set_xticks(np.array([0,1,2,3,4,5]))
ax.tick_params('y',labelsize=15)
ax.tick_params('x',labelsize=15)
plot.xlabel('time (s)')
ax.xaxis.label.set_size(15)
plot.ylabel('frequeny (Hz)')
ax.yaxis.label.set_size(15)
fig.colorbar(spectrum_noisy[3])
fig.savefig(
            os.path.join(path,'noisy.eps'),
            bbox_inches='tight',format='eps'
        )







path_s='C:/Users/hoss3301/work/snhl/data/stimuli/'
path_dl='https://resources.drcmr.dk/uheal/data_without_audio/stimuli/'
def exists(path,auth=('paper','paper')):
    r = requests.head(path,auth=auth)
    return r.status_code == requests.codes.ok



for s in ['masker']:
    for subjects in range(33,34):
        print('Downloading data from subject {}'.format(subjects))
        for i in range(1,49):
            print('Audio number: {}'.format(i))
            if len(str(i))==1:
                sub=str(subjects)
                song=str(i)
                path_download=path_dl+'sub0'+sub+'/'+s+'/m00'+song+'.wav'
                path_save=path_s+'sub0'+sub+'/'+s+'/m00'+song+'.wav'
                save_folder=path_s+'sub0'+sub+'/'+s
            else:
                sub=str(subjects)
                song=str(i)
                path_download=path_dl+'sub0'+sub+'/'+s+'/m0'+song+'.wav'
                path_save=path_s+'sub0'+sub+'/'+s+'/m0'+song+'.wav'
                save_folder=path_s+'sub0'+sub+'/'+s

            try:
                if exists(path_download,auth=('paper', 'paper')):
                    doc = requests.get(path_download,auth=('paper', 'paper'))

                    if not os.path.isdir(save_folder):
                        os.mkdir(save_folder)

                    f = open(path_save,"wb")
                    f.write(doc.content)
                    f.close()
                else:
                    print('This file does not exist: audio {} of the {}'.format(i, subjects))

            except:
                print('This file does not exist: audio {} of the {}'.format(i, subjects))