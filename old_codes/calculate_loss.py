import sys

sys.path.append('../')
# from tools.utils.losses import *
import numpy as np
import scipy
import tensorflow.keras.backend as K
import tensorflow as tf
import functools
from TrialsOfNeuralVocalRecon.tools.utils.OBM import OBM

window_fn = functools.partial(tf.contrib.signal.hann_window, periodic=True)

"""    print("######## ESTOI LOSS ########")
print("y_true shape:      ", K.int_shape(y_true))   
print("y_pred shape:      ", K.int_shape(y_pred)) 

y_true = K.squeeze(y_true,axis=-1)
y_pred = K.squeeze(y_pred,axis=-1)
y_pred_shape = K.shape(y_pred)

stft_true = tf.contrib.signal.stft(y_true,256,128,512,window_fn,pad_end=False)  
stft_pred = tf.contrib.signal.stft(y_pred,256,128,512,window_fn,pad_end=False)  
print("stft_true shape:   ", K.int_shape(stft_true))
print("stft_pred shape:   ", K.int_shape(stft_pred))

OBM1 = tf.convert_to_tensor(OBM)
OBM1 = K.tile(OBM1,[y_pred_shape[0],1,])
OBM1 = K.reshape(OBM1,[y_pred_shape[0],15,257,])
print("OBM1 shape:        ", K.int_shape(OBM1))

OCT_pred = K.sqrt(tf.matmul(OBM1,K.square(K.abs(tf.transpose(stft_pred)))))    
OCT_true = K.sqrt(tf.matmul(OBM1,K.square(K.abs(tf.transpose(stft_true)))))  
OCT_pred_shape = K.shape(OCT_pred)


#print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[0]))
#print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[1]))  
#print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[2]))
N = 30          # length of temporal envelope vectors
J = 15          # Number of one-third octave bands (cannot be varied) 
M = int(nbf-(N-1)) # number of temporal envelope vectors
smallVal = 0.0000000001 # To avoid divide by zero



d = K.variable(0.0,'float')
for i in range(0, I):
    print(i)
    for m in range(0, M):
        x = K.squeeze(tf.slice(OCT_true,    [i,0,m], [1,J,N]),axis=0)
        y = K.squeeze(tf.slice(OCT_pred,    [i,0,m], [1,J,N]),axis=0)                   
        #print("x shape:   ", K.int_shape(x))
        #print("y shape:   ", K.int_shape(y))                                
        #print("x shape:   ", K.eval(x))
        #print("y shape:   ", K.eval(y))
                   
        xn = x-K.mean(x,axis=-1,keepdims=True)
        #print("xn shape:   ", K.eval(xn))
        yn = y-K.mean(y,axis=-1,keepdims=True)
        #print("yn shape:   ", K.eval(yn))            
        
        xn = xn / (K.sqrt(K.sum(xn*xn,axis=-1,keepdims=True)) + smallVal )
        #print("xn shape:   ", K.eval(xn))
        
        yn = yn / (K.sqrt(K.sum(yn*yn,axis=-1,keepdims=True)) + smallVal )
        #print("yn shape:   ", K.eval(yn))
        
        xn = xn - K.tile(K.mean(xn,axis=-2,keepdims=True),[J,1,])
        #print("xn shape:   ", K.eval(xn))

        yn = yn - K.tile(K.mean(yn,axis=-2,keepdims=True),[J,1,])
        #print("yn shape:   ", K.eval(yn))
        
        xn = xn / (K.sqrt(K.sum(xn*xn,axis=-2,keepdims=True)) + smallVal )
        #print("xn shape:   ", K.eval(xn))
        
        yn = yn / (K.sqrt(K.sum(yn*yn,axis=-2,keepdims=True)) + smallVal )
        #print("yn shape:   ", K.eval(yn))
                    
        di = K.sum( xn*yn ,axis=-1,keepdims=True)
        #print("di shape:   ", K.eval(di))
        di = 1/N*K.sum( di ,axis=0,keepdims=False)
        #print("di shape:   ", K.eval(di))
        d  = d + di
        #print("d shape:   ", K.eval(d))

print("Compiling ESTOI LOSS Done!")
return 1-(d/K.cast(I*M,dtype='float'))
"""


def estoi_loss(I, nbf, y_true, y_pred):
    print("######## ESTOI LOSS ########")
    print("y_true shape:      ", np.shape(y_true))
    print("y_pred shape:      ", np.shape(y_pred))

    y_true = np.squeeze(y_true, axis=-1)
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred_shape = np.shape(y_pred)

    stft_true = scipy.signal.stft(y_true, fs=97656.25, window=scipy.signal.windows.hann(256, sym=False), nperseg=256,
                                  noverlap=128, nfft=256, detrend=False, return_onesided=False,
                                  boundary='zeros', padded=False, axis=-1)

    stft_pred = scipy.signal.stft(y_pred, fs=97656.25, window=scipy.signal.windows.hann(256, sym=False), nperseg=256,
                                  noverlap=128, nfft=256, detrend=False, return_onesided=False,
                                  boundary='zeros', padded=False, axis=-1)
    # stft_true = tf.contrib.signal.stft(y_true,256,128,512,window_fn,pad_end=False)
    # stft_pred = tf.contrib.signal.stft(y_pred,256,128,512,window_fn,pad_end=False)
    print("stft_true shape:   ", np.shape(stft_true[2]))
    print("stft_pred shape:   ", np.shape(stft_pred[2]))

    # OBM1 = tf.convert_to_tensor(OBM)
    OBM1 = np.tile(OBM, [y_pred_shape[0], 1, ])
    # OBM1 = K.tile(OBM1,[y_pred_shape[0],1,])
    OBM1 = np.reshape(OBM1, [y_pred_shape[0], 15, 257, ])
    print("OBM1 shape:        ", np.shape(OBM1))
    # ,perm=[0,2,1]
    # OCT_pred = np.sqrt(np.matmul(OBM1,np.square(np.abs(np.transpose(stft_pred[2],(0,2,1))))))
    OCT_pred = np.sqrt(np.matmul(OBM1[:, :, :-1], np.square(np.abs(stft_pred[2]))))

    # OCT_true = K.sqrt(tf.matmul(OBM1,K.square(K.abs(tf.transpose(stft_true)))))
    OCT_true = np.sqrt(np.matmul(OBM1[:, :, :-1], np.square(np.abs(stft_true[2]))))

    OCT_pred_shape = np.shape(OCT_pred)

    # print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[0]))
    # print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[1]))
    # print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[2]))
    N = 30  # length of temporal envelope vectors
    J = 15  # Number of one-third octave bands (cannot be varied)
    M = int(nbf - (N - 1))  # number of temporal envelope vectors
    smallVal = 0.0000000001  # To avoid divide by zero

    d = 0  # K.variable(0.0,'float')
    for i in range(0, I):
        print(i)
        for m in range(0, M):
            x = np.squeeze(OCT_true[i:i + 1, 0:0 + J, m:m + N], axis=0)
            y = np.squeeze(OCT_pred[i:i + 1, 0:0 + J, m:m + N], axis=0)
            # print("x shape:   ", K.int_shape(x))
            # print("y shape:   ", K.int_shape(y))
            # print("x shape:   ", K.eval(x))
            # print("y shape:   ", K.eval(y))

            xn = x - np.mean(x, axis=-1, keepdims=True)
            # print("xn shape:   ", K.eval(xn))
            yn = y - np.mean(y, axis=-1, keepdims=True)
            # print("yn shape:   ", K.eval(yn))

            xn = xn / (np.sqrt(np.sum(xn * xn, axis=-1, keepdims=True)) + smallVal)
            # print("xn shape:   ", K.eval(xn))

            yn = yn / (np.sqrt(np.sum(yn * yn, axis=-1, keepdims=True)) + smallVal)
            # print("yn shape:   ", K.eval(yn))

            xn = xn - np.tile(np.mean(xn, axis=-2, keepdims=True), [J, 1, ])
            # print("xn shape:   ", K.eval(xn))

            yn = yn - np.tile(np.mean(yn, axis=-2, keepdims=True), [J, 1, ])
            # print("yn shape:   ", K.eval(yn))

            xn = xn / (np.sqrt(np.sum(xn * xn, axis=-2, keepdims=True)) + smallVal)
            # print("xn shape:   ", K.eval(xn))

            yn = yn / (np.sqrt(np.sum(yn * yn, axis=-2, keepdims=True)) + smallVal)
            # print("yn shape:   ", K.eval(yn))

            di = np.sum(xn * yn, axis=-1, keepdims=True)
            # print("di shape:   ", K.eval(di))
            di = 1 / N * np.sum(di, axis=0, keepdims=False)
            # print("di shape:   ", K.eval(di))
            d = d + di
            print(d)
            # print("d shape:   ", K.eval(d))

    print("Compiling ESTOI LOSS Done!")
    loss = 1 - (d / (I * M))
    return loss


def calculate_loss(batch_size, nb_stft_frames, loss_type, y_true, y_pred):
    if loss_type == "estoi":
        loss = estoi_loss(batch_size, nb_stft_frames, y_true, y_pred)
    return loss


# b_stft_frames = (nb_samples/128)-1
batch_size = 32
"""
path='D:/data/trained_models/5_matin/'
file_name='prediction_finetuned.npy'

prediction_finetuned_spikes=np.load(path+file_name)

import h5py
path='D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/old_fromPuget/'
file_name='input_clean_test.h5'
output=h5py.File(path+file_name,'r')
output.keys
output=output['input_test'][:]


path='D:/data/trained_models/6_matin/'
file_name='prediction_finetune_NoSpike.npy'

prediction_finetuned_Nospikes=np.load(path+file_name)
"""
nbf = (np.shape(prediction_finetuned_Nospikes)[1] / 128) - 1  # (should be 31900/128-1)

y_true = output[0:1, 1:].astype(np.float32)
y_true = y_true.transpose(1, 0)

y_pred = prediction_finetuned_spikes[0:1, :].astype(np.float32)
y_pred = y_pred.transpose(1, 0)

y_pred = prediction_finetuned_Nospikes[0:1, :].astype(np.float32)
y_pred = y_pred.transpose(1, 0)

I = batch_size
l = estoi_loss(batch_size, nbf, y_true, y_pred)
loss_spikes = loss
loss_noSpikes = loss
