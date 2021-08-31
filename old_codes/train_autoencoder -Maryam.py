from __future__ import print_function
import os, sys
import numpy as np
from models.audio_autoencoder import build_autoencoder, corr2_mse_loss
import h5py
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from world import main
from tensorflow.python.keras.models import load_model
from data_preprocessing.convenience_tools import timeStructured
import scipy.io
import scipy.io as sio
from numpy.random import seed
seed(14)
from tensorflow import set_random_seed
set_random_seed(14)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.set_printoptions(threshold=sys.maxsize, precision=1)

import tensorflow as tf
from keras.callbacks import TensorBoard

# GPU configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

epochs = 5000 # 5000
batch_size = 75 #100
batch_size_val = 64

logPath = 'data/logs/'
if not os.path.isdir(logPath): os.mkdir(logPath)   
models_folder = 'data/trained_models/'
if not os.path.isdir(models_folder): os.mkdir(models_folder)   
train = True
if train:
    time_string = timeStructured()
    models_folder += time_string + '_' 
    logPath += time_string + '_log' 
    
print('Loading data...')
####h5py




dataPath = './data/original/vocoder_sep.mat'
mat1 = scipy.io.loadmat(dataPath)

f0_train = mat1['f0_train'].swapaxes(1,0)
max_f0_train=np.max(np.abs(f0_train))
f0_train=f0_train/[max_f0_train]

f0_test = mat1['f0_test'].swapaxes(1,0)
max_f0_test=np.max(np.abs(f0_test))
f0_test=f0_test/[max_f0_test]


vuv_train = mat1['vuv_train'].swapaxes(1,0).astype(np.float64)
vuv_test = mat1['vuv_test'].swapaxes(1,0).astype(np.float64)

aperiodicity_train = mat1['aperiodicity_train'][0:1,:].swapaxes(1,0)
max_aper_train=np.max(np.abs(aperiodicity_train))
aperiodicity_train=aperiodicity_train/[max_aper_train]


aperiodicity_test = mat1['aperiodicity_test'][0:1,:].swapaxes(1,0)
max_aper_test=np.max(np.abs(aperiodicity_test))
aperiodicity_test=aperiodicity_test/[max_aper_test]


#c=np.argwhere(aperiodicity_train!=-1)
#aperiodicity_train[c]=0

spectrogram_train = mat1['spec_train'].swapaxes(1,0)
max_spec_train=np.zeros((1,spectrogram_train.shape[1]))
for i in range(spectrogram_train.shape[1]):
    max_spec_train[:,i]=np.max(np.abs(spectrogram_train[:,i]))
    spectrogram_train[:,i]=spectrogram_train[:,i]/[max_spec_train[:,i]]
    
spectrogram_test = mat1['spec_test'].swapaxes(1,0)
max_spec_test=np.zeros((1,spectrogram_test.shape[1]))
for i in range(spectrogram_test.shape[1]):
    max_spec_test[:,i]=np.max(np.abs(spectrogram_test[:,i]))
    spectrogram_test[:,i]=spectrogram_test[:,i]/[max_spec_test[:,i]]

    
    
temporal_positions_train = mat1['temp_pos_train'].swapaxes(1,0)
temporal_positions_test = mat1['temp_pos_test'].swapaxes(1,0)


"""h5f = h5py.File('data/h5_sounds/clean_guinea_sounds_test.h5', 'r')
f0_val = h5f['f0'][0:294,:]
max_f0_val=np.max(np.abs(f0_val))
f0_val=f0_val/[max_f0_val]
vuv_val = h5f['vuv'][0:294,:]
aperiodicity_val = h5f['aperiodicity'][0:294,:]
max_aper_val=np.max(np.abs(aperiodicity_val))

aperiodicity_val=aperiodicity_val/[max_aper_val]
#c_val=np.argwhere(aperiodicity_val!=-1)
#aperiodicity_val[c_val]=0

spectrogram_val = h5f['spectrogram'][0:294,:]
max_spec_val=np.zeros((1,spectrogram_val.shape[1]))

for i in range(spectrogram_val.shape[1]):
    max_spec_val[:,i]=np.max(np.abs(spectrogram_val[:,i]))
    spectrogram_val[:,i]=spectrogram_val[:,i]/[max_spec_val[:,i]]
temporal_positions_val = h5f['temporal_positions'][:]
h5f.close()
"""
spec_shp = spectrogram_train.shape[1]

#features_train = np.concatenate((spectrogram_train, aperiodicity_train, f0_train, vuv_train), axis=1)
features_train = np.concatenate((spectrogram_train, f0_train,  vuv_train, np.abs(aperiodicity_train)), axis=1)
train_data_input = features_train
train_data_output = [features_train[:, 0:spec_shp], features_train[:, spec_shp:spec_shp + 1], features_train[:, spec_shp + 1:spec_shp + 2], features_train[:, spec_shp + 2:spec_shp + 3]]
features_val = np.concatenate((spectrogram_test, f0_test,  vuv_test, np.abs(aperiodicity_test)), axis=1)
val_data = [features_val[:batch_size_val, :], [features_val[:batch_size_val, 0:spec_shp], features_val[:batch_size_val, spec_shp:spec_shp+1], features_val[:batch_size_val, spec_shp + 1:spec_shp+2], features_val[:batch_size_val, spec_shp + 2:spec_shp+3]]]

print('Loading done.')

if train:
    aec = build_autoencoder(
        feats_shp=features_train.shape[1], 
        spec_shp=spectrogram_train.shape[1],
        lat_dim=256
        )
    print('              Training:')
    callbacks = []
    callbacks.append(TensorBoard(logPath, histogram_freq=int(epochs / 20) + 1,
                                 write_graph=True, write_grads=False,
                                 write_images=False, batch_size=batch_size))
    history = aec.autoencoder.fit(train_data_input,
                                  train_data_output,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=val_data)
    
    
    
    aec.autoencoder.save(models_folder + 'Autoencoder_val.h5')
    aec.encoder.save(models_folder + 'Encoder_val.h5')
    aec.decoder.save(models_folder + 'Decoder_val.h5')
    
    prediction = aec.autoencoder.predict(features_train)
    prediction_test = aec.autoencoder.predict(features_val)


else:
    # autoencoder = load_model(models_folder + 'Autoencoder_val.h5', custom_objects={'corr2_mse_loss': corr2_mse_loss})
    autoencoder = load_model(models_folder + 'Autoencoder_val.h5')
    prediction = autoencoder.predict(features_train)
    prediction_test = aec.autoencoder.predict(features_val)





####### test for the train set
spectrogram_prediction = prediction[0][0:784,:]
for i in range(spectrogram_prediction.shape[1]):
    spectrogram_prediction[:,i]=spectrogram_prediction[:,i]*[max_spec_train[:,i]]
for i in range(spectrogram_prediction.shape[0]):
    print('i='+ str(i)) 
    for j in range(spectrogram_prediction.shape[1]):
        print('j='+ str(j))
        if spectrogram_prediction[i,j]==0:
            spectrogram_prediction[i,j]=0.000001
aperiodicity_predict=prediction[3]
aperiodicity_predict=-aperiodicity_predict*max_aper_train

f0_predict=prediction[1]*max_f0_train


h5f = h5py.File('data/h5_sounds/clean_guinea_sounds_train.h5', 'r') 
#prediction
dat_p = {}
dat_p['is_requiem'] = True
dat_p['fs'] = 97656.25
dat_p['f0'] = f0_predict[:,0]
dat_p['vuv'] = prediction[2][0:784,0]
aperiodicity_predict=aperiodicity_predict[np.newaxis, :,0]
aperiodicity_r_predict=h5f['aperiodicity_r'][:]
#aperiodicity_r=np.pad(aperiodicity_r,((0,0),(0,16)),'constant', constant_values=((0,0),(0, 0)))
dat_p['aperiodicity'] = np.concatenate((aperiodicity_predict,aperiodicity_r_predict),axis=0)
dat_p['spectrogram'] = spectrogram_prediction.swapaxes(1,0)
temporal_positions=h5f['temporal_positions'][:]
#temporal_positions=np.pad(temporal_positions,(0,16),'constant', constant_values=(0, 0))
dat_p['temporal_positions'] = temporal_positions
ps_spectrogram=h5f['ps_spectrogram'][:]
#ps_spectrogram=np.pad(ps_spectrogram,(0,16),'constant',constant_values=(0,0))
dat_p['ps spectrogram']=ps_spectrogram

#reality    
dat = {}
dat['is_requiem'] = True
dat['fs'] = 97656.25
dat['f0'] = h5f['f0'][0:784,0]
dat['vuv'] = h5f['vuv'][0:784,0]
aperiodicity_train_plot=h5f['aperiodicity'][:].swapaxes(1,0)
aperiodicity_r_train_plot=h5f['aperiodicity_r'][:]
#aperiodicity_r=np.pad(aperiodicity_r,((0,0),(0,16)),'constant', constant_values=((0,0),(0, 0)))
dat['aperiodicity'] = np.concatenate((aperiodicity_train_plot[:,0:784],aperiodicity_r_train_plot),axis=0)
dat['spectrogram'] = h5f['spectrogram'][0:784,:].swapaxes(1,0)
#temporal_positions=np.pad(temporal_positions,(0,16),'constant', constant_values=(0, 0))
dat['temporal_positions'] = h5f['temporal_positions'][:]
#ps_spectrogram=np.pad(ps_spectrogram,(0,16),'constant',constant_values=(0,0))
dat['ps spectrogram']=h5f['ps_spectrogram'][:]


print(aperiodicity_train[:10])


h5f.close()




############### test for test set
#prediction
spectrogram_p_test = prediction_test[0][:]
for i in range(spectrogram_p_test.shape[1]):
    spectrogram_p_test[:,i]=spectrogram_p_test[:,i]*[max_spec_test[:,i]]
for i in range(spectrogram_p_test.shape[0]):
    print('i='+ str(i)) 
    for j in range(spectrogram_test.shape[1]):
        print('j='+ str(j))
        if spectrogram_p_test[i,j]==0:
            spectrogram_p_test[i,j]=0.000001
spectrogram_p_test=spectrogram_p_test.swapaxes(1,0)

f0_p_test=prediction_test[1].swapaxes(1,0)*max_f0_test

vuv_p_test=prediction_test[2].swapaxes(1,0)

aperiodicity_p_test_r = mat1['aperiodicity_test'][1:8,:]
aperiodicity_p_test=prediction_test[3].swapaxes(1,0)
aperiodicity_p_test=-aperiodicity_p_test*max_aper_test
aperiodicity_p_test=np.concatenate((aperiodicity_p_test,aperiodicity_p_test_r),axis=0)
temporal_positions_p_test = mat1['temp_pos_test']



spectrogram_test = mat1['spec_test']
aperiodicity_test = mat1['aperiodicity_test']
f0_test = mat1['f0_test']
vuv_test = mat1['vuv_test'].astype(np.float64)
temporal_positions_test = mat1['temp_pos_test']

fs=97656.25
sio.savemat(models_folder +'preds_test_autoencoder.mat', mdict={'spectrogram':spectrogram_p_test, 'band_aperiodicity':aperiodicity_p_test, 'f0':f0_p_test, 'vuv':vuv_p_test, 'temporal_positions':temporal_positions_p_test, 'fs':fs})
sio.savemat(models_folder +'test_autoencoder.mat', mdict={'spectrogram':spectrogram_test, 'band_aperiodicity':aperiodicity_test, 'f0':f0_test, 'vuv':vuv_test, 'temporal_positions':temporal_positions_test, 'fs':fs})


   

"""h5f = h5py.File('data/h5_sounds/clean_guinea_sounds_test.h5', 'r') 

dat_p_test = {}
dat_p_test['is_requiem'] = True
dat_p_test['fs'] = 97656.25
dat_p_test['f0'] = f0_test[:,0]
dat_p_test['vuv'] = prediction_test[2][:,0]
aperiodicity_test=aperiodicity_test[np.newaxis, :,0]
aperiodicity_r_test=h5f['aperiodicity_r'][:]
#aperiodicity_r=np.pad(aperiodicity_r,((0,0),(0,16)),'constant', constant_values=((0,0),(0, 0)))
dat_p_test['aperiodicity'] = np.concatenate((aperiodicity_test,aperiodicity_r_test),axis=0)
dat_p_test['spectrogram'] = spectrogram_test.swapaxes(1,0)
dat_p_test['temporal_positions'] = h5f['temporal_positions'][:]
dat_p_test['ps spectrogram']=h5f['ps_spectrogram'][:]


#reality

dat_test = {}
dat_test['is_requiem'] = True
dat_test['fs'] = 97656.25
dat_test['f0'] = h5f['f0'][0:294,0]
dat_test['vuv'] = h5f['vuv'][0:294,0]
aperiodicity_test_plot=h5f['aperiodicity'][:].swapaxes(1,0)
aperiodicity_r_test_plot=h5f['aperiodicity_r'][:]
#aperiodicity_r=np.pad(aperiodicity_r,((0,0),(0,16)),'constant', constant_values=((0,0),(0, 0)))
dat_test['aperiodicity'] = np.concatenate((aperiodicity_test_plot[:,0:294],aperiodicity_r_test_plot),axis=0)
dat_test['spectrogram'] = h5f['spectrogram'][0:294,:].swapaxes(1,0)
#temporal_positions=np.pad(temporal_positions,(0,16),'constant', constant_values=(0, 0))
dat_test['temporal_positions'] = h5f['temporal_positions'][:]
#ps_spectrogram=np.pad(ps_spectrogram,(0,16),'constant',constant_values=(0,0))
dat_test['ps spectrogram']=h5f['ps_spectrogram'][:]


#plot the decoded results for the train set
vocoder = main.World()

dat_decoded = vocoder.decode(dat)
signal = dat['out']

dat_p_decoded = vocoder.decode(dat_p)
signal_p = dat_p['out']
print(signal_p)

fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True, sharey=True)
fig.suptitle('Audio Autoencoding')
max_value = np.max(np.max(features_train[:, 0:spec_shp]))


axs[0].set_title('original')
axs[0].plot(signal)
axs[1].set_title('prediction')
axs[1].plot(signal_p)
fig.savefig('data/AE_reconstruction.png')



#plot the decoded results for the test set
vocoder = main.World()

dat_test_decoded = vocoder.decode(dat_test)
signal_test = dat_test_decoded['out']

dat_p_test_decoded = vocoder.decode(dat_p_test)
signal_p_test = dat_p_test['out']

fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True, sharey=True)
fig.suptitle('Audio Autoencoding test set')
max_value = np.max(np.max(features_val[:, 0:spec_shp]))


axs[0].set_title('original')
axs[0].plot(signal_test)
axs[1].set_title('prediction')
axs[1].plot(signal_p_test)
fig.savefig('data/AE_reconstruction_test.png')


"""