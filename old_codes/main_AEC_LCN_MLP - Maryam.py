from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, Activation, concatenate, LocallyConnected2D
from keras.layers import MaxPooling1D, MaxPooling2D, Flatten, LSTM, noise, Reshape, Add, Lambda
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.regularizers import l2
from keras.utils import np_utils, plot_model
from keras.layers.normalization import BatchNormalization
import keras
from keras import backend as BK
from keras.optimizers import *
from keras.layers.advanced_activations import LeakyReLU, ELU
import scipy.io as sio
import numpy as np
import h5py
import scipy
import random
from data_preprocessing.convenience_tools import timeStructured

import sys
sys.path.append('../')
from world import main
from numpy.random import seed
seed(14)
from tensorflow import set_random_seed
set_random_seed(14)

#GPU configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def corr2(a,b):
    k = np.shape(a)
    H=k[0]
    W=k[1]
    c = np.zeros((H,W))
    d = np.zeros((H,W))
    e = np.zeros((H,W))

    #Calculating mean values
    AM=np.mean(a)
    BM=np.mean(b)  

    #Calculating terms of the formula
    for ii in range(H):
      for jj in range(W):
        c[ii,jj]=(a[ii,jj]-AM)*(b[ii,jj]-BM)
        d[ii,jj]=(a[ii,jj]-AM)**2
        e[ii,jj]=(b[ii,jj]-BM)**2

    #Formula itself
    r = np.sum(c)/float(np.sqrt(np.sum(d)*np.sum(e)))
    return r

def corr2_mse_loss(a,b):
    a = tf.subtract(a, tf.reduce_mean(a))
    b = tf.subtract(b, tf.reduce_mean(b))
    tmp1 = tf.reduce_sum(tf.multiply(a,a))
    tmp2 = tf.reduce_sum(tf.multiply(b,b))
    tmp3 = tf.sqrt(tf.multiply(tmp1,tmp2))
    tmp4 = tf.reduce_sum(tf.multiply(a,b))
    r = -tf.divide(tmp4,tmp3)
    m=tf.reduce_mean(tf.square(tf.subtract(a, b)))
    rm=tf.add(r,m)
    return rm



logPath = 'data/logs/neural_trained/'
if not os.path.isdir(logPath): os.mkdir(logPath)   
models_folder = 'data/trained_models/neural_trained/'
if not os.path.isdir(models_folder): os.mkdir(models_folder)   
time_string = timeStructured()
#models_folder += time_string + '_' 
logPath += time_string + '_log' 


####two subjects
print('Loading data and models...')
print('Loading data...')

"""h5f = h5py.File('./data/h5_sounds/clean_guinea_sounds_train.h5','r')
f0_train = h5f['f0'][0:784,:]
max_f0_train=np.max(np.abs(f0_train))
f0_train=f0_train/[max_f0_train]
vuv_train = h5f['vuv'][0:784,:]
aperiodicity_train = h5f['aperiodicity'][0:784,:]
max_aper_train=np.max(np.abs(aperiodicity_train))
aperiodicity_train=aperiodicity_train/[max_aper_train]
spectrogram_train = h5f['spectrogram'][0:784,:]
max_spec_train=np.zeros((1,spectrogram_train.shape[1]))
for i in range(spectrogram_train.shape[1]):
    max_spec_train[:,i]=np.max(np.abs(spectrogram_train[:,i]))
    spectrogram_train[:,i]=spectrogram_train[:,i]/[max_spec_train[:,i]]
h5f.close()
"""
"""dataPath = './data/original/vocoder_sep.mat'
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

ap=mat1['aperiodicity_test']
aperiodicity_test = ap[0:1,:].swapaxes(1,0)
aperiodicity_test_r=ap[1:8,:]
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
"""
"""
neu_train = np.load('./data/neural_data/rawData/windowed_train_concat.npy')
neu_train=neu_train[0:784,:,:,:]
max_neural_train=np.max(np.abs(neu_train))
neu_train=neu_train/max_neural_train

neu_val = np.load('./data/neural_data/rawData/windowed_test_concat.npy')
neu_val=neu_val[0:294,:,:,:]
max_neural_val=np.max(np.abs(neu_val))
neu_val=neu_val/max_neural_val"""

"""h5f = h5py.File('./data/h5_sounds/clean_guinea_sounds_test.h5','r')
f0_val = h5f['f0'][0:294,]
max_f0_val=np.max(np.abs(f0_val))
f0_val=f0_val/[max_f0_val]

vuv_val = h5f['vuv'][0:294,:]
aperiodicity_val = h5f['aperiodicity'][0:294,:]
max_aper_val=np.max(np.abs(aperiodicity_val))
aperiodicity_val=aperiodicity_val/[max_aper_val]

spectrogram_val = h5f['spectrogram'][0:294,:]
max_spec_val=np.zeros((1,spectrogram_val.shape[1]))

for i in range(spectrogram_val.shape[1]):
    max_spec_val[:,i]=np.max(np.abs(spectrogram_val[:,i]))
    spectrogram_val[:,i]=spectrogram_val[:,i]/[max_spec_val[:,i]]
h5f.close()

"""


#features_train = np.concatenate((spectrogram_train, f0_train,  vuv_train, np.abs(aperiodicity_train)), axis=1)

#features_train = np.concatenate((spectrogram_train[0:100,:], aperiodicity_train[0:100,:], f0_train[0:100,:], vuv_train[0:100,:]), axis=1)
#features_val = np.concatenate((spectrogram_val[0:10,:], aperiodicity_val[0:10,:], f0_val[0:10,:], vuv_val[0:10,:]), axis=1)
#features_val = np.concatenate((spectrogram_test, f0_test,  vuv_test, np.abs(aperiodicity_test)), axis=1)

print('Loading and concatenation done.')

#Bottleneck='B256'
date= '2019-12-29-20-09-59_'#'2019-12-13-13-00-54_'#'2019-12-09-17-34-02_'
#print('Coding features...')

#Encoder_name='./data/trained_models/'+date+'Encoder_val.h5'

#encoder = load_model(Encoder_name)#custom_objects={'corr2_mse_loss': corr2_mse_loss})
#encoded_train = encoder.predict(features_train)
#encoded_val = encoder.predict(features_val)
#print('Coding done.')


def save_preds(encoded_preds,date):
    print('Decoding and saving predicted features...')
    #decoder = load_model(D_name, custom_objects={'corr2_mse_loss': corr2_mse_loss})
    Decoder_name='./data/trained_models/'+date+'Decoder_val.h5'   
    decoder = load_model(Decoder_name)

    decoded_preds = decoder.predict(encoded_preds)
    #spec=np.power(decoded_preds[0],10)
    spec=decoded_preds[0].swapaxes(1,0)
    for i in range(spectrogram_test.shape[1]):
        spec[i,:]=spec[i,:]*[max_spec_test[:,i]]
    #aper=-np.power(10,decoded_preds[3])+1
    aper=-decoded_preds[3].swapaxes(1,0)*max_aper_test
    aper_all=np.concatenate((aper,aperiodicity_test_r),axis=0)
    #f0=np.power(10,decoded_preds[1])-1
    f0=decoded_preds[1].swapaxes(1,0)*max_f0_test
    vuv=decoded_preds[2].swapaxes(1,0)
    #vuv=np.round(decoded_preds[2])
    sio.savemat(models_folder +'Main_preds_test_AEC_LCN_MLP.mat', mdict={'spectrogram':spec, 'band_aperiodicity':aper_all, 'f0':f0, 'vuv':vuv,'temporal_positions':temporal_positions_test, 'fs':97656.25})
    print('Saving done.')
    
    
"""def find_time_signal(spec,aper,f0,vuv,test_path):
    h5f = h5py.File('./data/h5_sounds/clean_guinea_sounds_test.h5','r')
    dat_val = {}
    dat_val['is_requiem'] = True
    dat_val['fs'] = 97656.25
    dat_val['f0'] = h5f['f0'][0:294,0]
    dat_val['vuv'] =  h5f['vuv'][0:294,0]
    aperiodicity = h5f['aperiodicity'][0:294,:]
    aperiodicity=aperiodicity[np.newaxis, :,0]    
    dat_val['spectrogram'] = h5f['spectrogram'][0:294,:].swapaxes(1,0)
    aperiodicity_r=h5f['aperiodicity_r'][:]  
    dat_val['aperiodicity'] = np.concatenate((aperiodicity,aperiodicity_r),axis=0)
    dat_val['temporal_positions'] = h5f['temporal_positions'][:]
    dat_val['ps spectrogram']=h5f['ps_spectrogram'][:]
    vocoder = main.World()
    dat_val_decoded = vocoder.decode(dat_val)
    signal_val_decoded = dat_val_decoded['out']
    
    dat_decoded = {}
    dat_decoded['is_requiem'] = True
    dat_decoded['fs'] = 97656.25
    dat_decoded['f0'] = f0[:,0]
    dat_decoded['vuv'] =  vuv[:,0]
    dat_decoded['spectrogram'] = spec.swapaxes(1,0)
    aperiodicity_r=h5f['aperiodicity_r'][:]  
    dat_decoded['aperiodicity'] = np.concatenate((aper.swapaxes(1,0),aperiodicity_r),axis=0)
    dat_decoded['temporal_positions'] = h5f['temporal_positions'][:]
    dat_decoded['ps spectrogram']=h5f['ps_spectrogram'][:]
    vocoder = main.World()
    dat_out_decoded = vocoder.decode(dat_decoded)
    signal_decoded = dat_out_decoded['out']    
    
    return signal_val_decoded,signal_decoded
"""
#main network
#adam=Adam(lr=.0001)
shp_in=(900,265,1)
shp_out=(4100)
def build_model(shp_in,shp_out):
    reg=.0005
    inputs = Input(shape=shp_in)
#    x = LocallyConnected2D(1, kernel_size=[5, 5], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(inputs)
    x = Conv2D(1, kernel_size=[5, 5], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(inputs)
    
    x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = Conv2D(1, kernel_size=[3, 3], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    #x = LocallyConnected2D(1, kernel_size=[3, 3], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = Conv2D(2, kernel_size=[1, 1], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    #x = LocallyConnected2D(2, kernel_size=[1, 1], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = Flatten()(x)

    x_MLP = Flatten()(inputs)
    x_MLP = Dense(10,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
 #   x_MLP = Dense(256,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
    x_MLP = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x_MLP)))
    x_MLP = Dense(10,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
#    x_MLP = Dense(256,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
    x_MLP = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x_MLP)))

    x = concatenate([x,x_MLP], axis=1)

    #x = Dense(256,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dense(10,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x)))
    x = Dense(10,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
#    x = Dense(128,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x)))
    x = Dense(shp_out,kernel_initializer='he_normal')(x)
    coded_preds = Activation('tanh', name='coded_preds')(x)
    model = Model(inputs, coded_preds)
    return model

#Inits
epochs=1000
batch_size=32
adam=Adam(lr=0.00001)#.0001)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
num_iter=1
shp_in=neu_val.shape[1:]
shp_out=encoded_val.shape[1]
#loss_history=np.empty((num_iter,2), dtype='float32')
#cnt_lr=0

model=build_model(shp_in,shp_out)
model.compile(loss=corr2_mse_loss, optimizer=adam)
callbacks = []
#callbacks.append(TensorBoard(logPath, histogram_freq=int(epochs / 20) + 1,
 #                             batch_size=batch_size))
callbacks.append(TensorBoard(logPath, histogram_freq=int(epochs / 20) + 1,
                                 write_graph=True, write_grads=False,
                                 write_images=False, batch_size=batch_size))
#filepath='./data/Main_models/'neural_trained/Model_Best_Val_AEC_LCN_MLP.h5'
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]
    
    #history = model.fit(neu_train[0:100,:,:,:], encoded_train, epochs=1, batch_size=256, verbose=1, callbacks=callbacks_list,  validation_data=(neu_val[0:10,:,:,:], encoded_val))
history = model.fit(neu_train, encoded_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks,  validation_data=(neu_val, encoded_val))

    #loss_history[j,0]=history.history['loss'][0]
    #loss_history[j,1]=history.history['val_loss'][0]
    #if i>4 and cnt_lr<2:
    #    if loss_history[i,j-5,1]<loss_history[i,j,1] and loss_history[i,j-5,1]<loss_history[i,j-1,1] and loss_history[i,j-5,1]<loss_history[i,j-2,1] and loss_history[i,j-5,1]<loss_history[i,j-3,1] and loss_history[i,j-5,1]<loss_history[i,j-4,1]:
    #    print("########### Validation loss didn't improve after 5 epochs, lr is divided by 2 ############")
    #    BK.set_value(model.optimizer.lr, .5*BK.get_value(model.optimizer.lr))
    #    cnt_lr+=1
model.save(models_folder +'/Model_Val_AEC_LCN_MLP.h5')
#model.load_weights(filepath)
encoded_preds = model.predict(neu_val)
h5f = h5py.File(models_folder +'/Encoded_test_AEC_LCN_MLP','w')
h5f.create_dataset('encoded_preds', data=encoded_preds)
h5f.close()
save_preds(encoded_preds,Decoder_name)
