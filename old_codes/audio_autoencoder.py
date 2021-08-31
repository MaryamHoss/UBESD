from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, Activation, concatenate
from keras.layers import MaxPooling1D, MaxPooling2D, Flatten, LSTM, noise, Reshape, Add, Lambda
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.utils import np_utils, plot_model
from keras.layers.normalization import BatchNormalization
import keras
from keras import backend as BK
from keras.optimizers import *
from keras.layers.advanced_activations import LeakyReLU, ELU
import scipy.io as sio
import numpy as np
import scipy
import random
from keras import losses


class build_autoencoder():
    #initialization
    def __init__(self, feats_shp, spec_shp, lat_dim):
        self.adam=Adam(lr=.0000001)
        self.reg=.001
        self.feats_shp = feats_shp #features_train.shape[1]
        self.spec_shp = spec_shp #spectrogram_train.shape[1]
        self.lat_dim = lat_dim
        
        #building models
        self.encoder = self.build_encoder()
        #self.encoder.compile(loss=self.corr2_mse_loss, optimizer=self.adam)
        self.encoder.compile(loss='mean_squared_error', optimizer=self.adam)
        
        self.decoder = self.build_decoder()
        #self.decoder.compile(loss=self.corr2_mse_loss, optimizer=self.adam)
        self.decoder.compile(loss='mean_squared_error', optimizer=self.adam)
        inputs = Input(shape=(self.feats_shp,))
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        #decoded = [self.renamer(decoded[0],'spec'), self.renamer(decoded[1],'aper'), self.renamer(decoded[2],'f0'), self.renamer(decoded[3],'vuv')]
        decoded = [self.renamer(decoded[0],'spec'), self.renamer(decoded[1],'f0'), self.renamer(decoded[2],'vuv'), self.renamer(decoded[3],'aper')]
        
        self.autoencoder = Model(inputs,decoded)
        #self.autoencoder.compile(loss=self.corr2_mse_loss, optimizer=self.adam)
        self.autoencoder.compile(loss='mean_squared_error', optimizer=self.adam)
        #self.autoencoder.summary()
    
    def renamer(self,x,name):
        renamer_lambda = Lambda(lambda x:x, name=name)
        return renamer_lambda(x)
    
    #custom loss function
    def corr2_mse_loss(self,a,b):
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
    
    #encoder part of auto-encoder
    def build_encoder(self):
        in_encoder = Input(shape=(self.feats_shp,))
        x = Dense(512)(in_encoder)
        x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(400)(x)
        x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(300)(x)
        x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(200)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(100)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(50)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(self.lat_dim)(x)
        x = Activation('tanh')(BatchNormalization()(x))
        out_encoder = noise.GaussianNoise(.2)(x)
        self.out_encoder_shape = int(out_encoder.shape[1])
        
        encoder = Model(in_encoder,out_encoder)
        return encoder
    
    #decoder part of auto-encoder
    def build_decoder(self):
        in_decoder = Input(shape=(self.out_encoder_shape,))
        x = Dense(300)(in_decoder)
        x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(100)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(200)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(300)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(400)(x)
        x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(512)(x)
        x = LeakyReLU()(BatchNormalization()(x))

        #spec branch
        x_spec = Dense(512)(x)
        x_spec = LeakyReLU()(x_spec)
        x_spec = Dense(self.spec_shp)(x_spec)
        x_spec = Activation('relu', name='spec')(x_spec)

        #aperiodicity branch
        x_aper = Dense(32)(x)
        x_aper = LeakyReLU()(BatchNormalization()(x_aper))
        x_aper = Dense(16)(x_aper)
        x_aper = LeakyReLU()(BatchNormalization()(x_aper))
        x_aper = Dense(8)(x_aper)
        x_aper = LeakyReLU()(BatchNormalization()(x_aper))
        x_aper = Dense(4)(x_aper)
        x_aper = LeakyReLU()(x_aper)
        #x_aper = Activation('relu', name='aper')(Dense(1)(x_aper))
        x_aper = Activation('relu', name='aper')(Dense(1)(x_aper))

        #f0 branch
        x_f0 = Dense(32)(x)
        x_f0 = LeakyReLU()(BatchNormalization()(x_f0))
        x_f0 = Dense(8)(x_f0)
        x_f0 = LeakyReLU()(x_f0)
        x_f0 = Activation('relu', name='f0')(Dense(1)(x_f0))

        #vuv branch
        x_vuv = Dense(32)(x)
        x_vuv = LeakyReLU()(BatchNormalization()(x_vuv))
        x_vuv = Dense(8)(x_vuv)
        x_vuv = LeakyReLU()(x_vuv)
        x_vuv = Activation('relu', name='vuv')(Dense(1)(x_vuv))

        #decoder = Model(in_decoder,[x_spec, x_aper, x_f0, x_vuv])
        decoder = Model(in_decoder,[x_spec,x_f0,x_vuv ,x_aper ])
        return decoder
    
    
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







"""from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, Activation, concatenate
from keras.layers import MaxPooling1D, MaxPooling2D, Flatten, LSTM, noise, Reshape, Add, Lambda
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.utils import np_utils, plot_model
from keras.layers.normalization import BatchNormalization
import keras
from keras import backend as BK
from keras.optimizers import *
from keras.layers.advanced_activations import LeakyReLU, ELU
import scipy.io as sio
import numpy as np
import scipy
import random


class build_autoencoder():
    #initialization
    def __init__(self, feats_shp, spec_shp, lat_dim):
        self.adam=Adam(lr=.00001)
        self.reg=.001
        self.feats_shp = feats_shp #features_train.shape[1]
        self.spec_shp = spec_shp #spectrogram_train.shape[1]
        self.lat_dim = lat_dim
        
        #building models
        self.encoder = self.build_encoder()
        self.encoder.compile(loss=self.corr2_mse_loss, optimizer=self.adam)
        
        self.decoder = self.build_decoder()
        self.decoder.compile(loss=self.corr2_mse_loss, optimizer=self.adam)
        
        inputs = Input(shape=(self.feats_shp,))
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        #decoded = [self.renamer(decoded[0],'spec'), self.renamer(decoded[1],'aper'), self.renamer(decoded[2],'f0'), self.renamer(decoded[3],'vuv')]
        decoded = [self.renamer(decoded[0],'spec'), self.renamer(decoded[1],'f0'), self.renamer(decoded[2],'vuv'), self.renamer(decoded[3],'aper')]
        
        self.autoencoder = Model(inputs,decoded)
        self.autoencoder.compile(loss=self.corr2_mse_loss, optimizer=self.adam)
        #self.autoencoder.summary()
    
    def renamer(self,x,name):
        renamer_lambda = Lambda(lambda x:x, name=name)
        return renamer_lambda(x)
    
    #custom loss function
    def corr2_mse_loss(self,a,b):
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
    
    #encoder part of auto-encoder
    def build_encoder(self):
        in_encoder = Input(shape=(self.feats_shp,))
        x = Dense(512)(in_encoder)
        x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(400)(x)
        x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(300)(x)
        x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(200)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(100)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(50)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(self.lat_dim)(x)
        x = Activation('tanh')(BatchNormalization()(x))
        out_encoder = noise.GaussianNoise(.2)(x)
        self.out_encoder_shape = int(out_encoder.shape[1])
        
        encoder = Model(in_encoder,out_encoder)
        return encoder
    
    #decoder part of auto-encoder
    def build_decoder(self):
        in_decoder = Input(shape=(self.out_encoder_shape,))
        x = Dense(300)(in_decoder)
        x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(100)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(200)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        #x = Dense(300)(x)
        #x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(400)(x)
        x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(512)(x)
        x = LeakyReLU()(BatchNormalization()(x))

        #spec branch
        x_spec = Dense(512)(x)
        x_spec = LeakyReLU()(x_spec)
        x_spec = Dense(self.spec_shp)(x_spec)
        x_spec = Activation('relu', name='spec')(x_spec)

        #aperiodicity branch
        x_aper = Dense(32)(x)
        x_aper = LeakyReLU()(BatchNormalization()(x_aper))
        x_aper = Dense(16)(x_aper)
        x_aper = LeakyReLU()(BatchNormalization()(x_aper))
        x_aper = Dense(8)(x_aper)
        x_aper = LeakyReLU()(BatchNormalization()(x_aper))
        x_aper = Dense(4)(x_aper)
        x_aper = LeakyReLU()(x_aper)
        x_aper = Activation('relu', name='aper')(Dense(1)(x_aper))

        #f0 branch
        x_f0 = Dense(32)(x)
        x_f0 = LeakyReLU()(BatchNormalization()(x_f0))
        x_f0 = Dense(8)(x_f0)
        x_f0 = LeakyReLU()(x_f0)
        x_f0 = Activation('relu', name='f0')(Dense(1)(x_f0))

        #vuv branch
        x_vuv = Dense(32)(x)
        x_vuv = LeakyReLU()(BatchNormalization()(x_vuv))
        x_vuv = Dense(8)(x_vuv)
        x_vuv = LeakyReLU()(x_vuv)
        x_vuv = Activation('relu', name='vuv')(Dense(1)(x_vuv))

        decoder = Model(in_decoder,[x_spec, x_aper, x_f0, x_vuv])
        return decoder
    
    
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
"""