# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:39:05 2019

@author: hoss3301
"""

import numpy as np
import scipy.io as spio
import h5py


def LagGen(x, b, a):
    l, m, p = x.shape
    length = np.ceil(l / b).astype(int)
    x0 = np.pad(x, ((0, a), (0, 0), (0, 0)), 'edge')
    o = np.zeros((length, a, m, p))
    for i in range(length):
        for j in range(m):
            for c in range(p):
                o[i, :, j, c] = x0[i * b:i * b + a, j, c]
    return o


## to shape the data like mesgarani

arrays = {}
f = h5py.File('C:/Users/hoss3301/work/deep_guinea_ears/data/data/data_rawData/in samples/Matlab Data/all.mat')
for k, v in f.items():
    arrays[k] = np.array(v)

spikes = arrays['all']  # all has a shape of (11,23928,20,265)

spikes_mean = np.zeros((spikes.shape[0], spikes.shape[1], spikes.shape[3]))  # take a mean on the trials
spikes_mean = np.mean(spikes, 2)

spikes_resampled = spikes_mean[:, ::8, :]  # downsample to 3 khz

b = 30  # 10msec stride
a = 900  # 300msec window

activity_transpose = np.transpose(spikes_resampled,
                                  (1, 2, 0))  # transpose so the first axis is the time and second is the num of neurons

activity_windowed = LagGen(activity_transpose, b, a)
windowed_train = np.zeros((100, 900, 265, 8))
windowed_test = np.zeros((100, 900, 265, 3))
windowed_test[:, :, :, 0:1] = activity_windowed[:, :, :, 0:1]
windowed_train[:, :, :, 0:5] = activity_windowed[:, :, :, 1:6]
windowed_test[:, :, :, 1:3] = activity_windowed[:, :, :, 6:8]
windowed_train[:, :, :, 5:8] = activity_windowed[:, :, :, 8:11]

windowed_train_concat = np.zeros((800, 900, 265))
windowed_test_concat = np.zeros((300, 900, 265))
for i in range(8):
    windowed_train_concat[i * 100:(i + 1) * 100, :, :] = windowed_train[:, :, :, i]
for i in range(3):
    windowed_test_concat[i * 100:(i + 1) * 100, :, :] = windowed_test[:, :, :, i]

windowed_train_concat = np.expand_dims(windowed_train_concat, axis=3)
windowed_test_concat = np.expand_dims(windowed_test_concat, axis=3)

np.save('data/neural_data/rawData/windowed_train_concat.npy', windowed_train_concat)
np.save('data/neural_data/rawData/windowed_test_concat.npy', windowed_test_concat)
np.save('data/neural_data/rawData/windowed_train.npy', windowed_train)
np.save('data/neural_data/rawData/windowed_test.npy', windowed_test)

####my data
"""
Created on Thu Nov 21 17:39:05 2019

@author: hoss3301
"""

import numpy as np
import scipy.io as spio
import h5py


def LagGen(x, b, a):
    l, m, p = x.shape
    length = np.ceil(l / b).astype(int)
    x0 = np.pad(x, ((0, a), (0, 0), (0, 0)), 'edge')
    o = np.zeros((length, a, m, p))
    for i in range(length):
        for j in range(m):
            for c in range(p):
                o[i, :, j, c] = x0[i * b:i * b + a, j, c]
    return o


# D:\data\ICC\spikes\RawData

arrays = {}
f = h5py.File('C:/Users/hoss3301/work/deep_guinea_ears/data/data_mine/spikes_noisy_clean.mat', 'r')
for k, v in f.items():
    arrays[k] = np.array(v)

spikes = arrays['spikes']  # spikes has a shape of (11,23928,20,265)

spikes_mean = np.zeros((spikes.shape[0], spikes.shape[1], spikes.shape[3]))  # take a mean on the trials
spikes_mean = np.mean(spikes, 2)

spikes_resampled = spikes_mean[:, ::8, :]  # downsample to 3 khz

b = 30  # 10msec stride
a = 900  # 300msec window
activity_transpose = np.transpose(spikes_resampled,
                                  (1, 2, 0))  # transpose so the first axis is the time and second is the num of neurons

activity_windowed = LagGen(activity_transpose, b, a)
windowed_train = np.zeros((100, 900, 265, 8))
windowed_test = np.zeros((100, 900, 265, 3))
windowed_test[:, :, :, 0:1] = activity_windowed[:, :, :, 0:1]
windowed_train[:, :, :, 0:5] = activity_windowed[:, :, :, 1:6]
windowed_test[:, :, :, 1:3] = activity_windowed[:, :, :, 6:8]
windowed_train[:, :, :, 5:8] = activity_windowed[:, :, :, 8:11]

windowed_train_concat = np.zeros((800, 900, 265))
windowed_test_concat = np.zeros((300, 900, 265))
for i in range(8):
    windowed_train_concat[i * 100:(i + 1) * 100, :, :] = windowed_train[:, :, :, i]
for i in range(3):
    windowed_test_concat[i * 100:(i + 1) * 100, :, :] = windowed_test[:, :, :, i]

windowed_train_concat = np.expand_dims(windowed_train_concat, axis=3)
windowed_test_concat = np.expand_dims(windowed_test_concat, axis=3)

np.save('data/neural_data/rawData/windowed_train_concat.npy', windowed_train_concat)
np.save('data/neural_data/rawData/windowed_test_concat.npy', windowed_test_concat)
np.save('data/neural_data/rawData/windowed_train.npy', windowed_train)
np.save('data/neural_data/rawData/windowed_test.npy', windowed_test)


def LagGen_2d(x, stride, win):
    length, width = x.shape
    n_win = np.ceil(width / stride).astype(int)
    x0 = np.pad(x, ((0, 0), (0, win - 1)), 'edge')
    o = np.zeros((length, n_win, win))
    for i in range(length):
        for j in range(n_win):
            o[i, j, :] = x0[i, j * stride:j * stride + win]
    return o


str_stim = round(0.24 * 97656.25)  # 10msec stride
win_stim = round(0.3 * 97656.25)  # 300msec window
str_spk = round(0.2 * 24414.4)
win_spk = round(0.3 * 24414.4)
n_win = np.ceil(input_train.shape[1] / str_stim).astype(int)
input_test_windowed = np.zeros((input_test.shape[0], n_win, win_stim))

input_test_windowed = LagGen_2d(input_test, str_stim, win_stim)
input_test_windowed = LagGen_2d(input_test, str_stim, win_stim)

max_input_test_windowed = np.zeros((1, 79500))

for i in range(79500):
    max_input_test_windowed[:, i] = np.max(np.abs(input_test_windowed[i, :]))

input_test_windowed_1 = np.zeros((15900 * 5, 29297))
input_test_windowed_1 = np.reshape(input_test_windowed, (15900 * 5, 29297))
for i in rane(79500):
    if max_input_test_windowed[:, i] == 0:
        max_input_test_windowed[:, i] = 0.0000001

    #########  5-2-2020    this is how i created the old data, wrong normalization
import numpy as np

input_train = np.load('E:/resnet stuff/input_train.npy')
win = int(input_train.shape[1] / 3)
input_train_windowed = np.zeros((42400, 3, win))
for i in range(42400):
    for j in range(3):
        input_train_windowed[i, j, :] = input_train[i, j * win:j * win + win]

input_train_windowed_reshaped = np.zeros((42400 * 3, win))

input_train_windowed_reshaped = np.reshape(input_train_windowed, (42400 * 3, win))

max_input_train_windowed = np.zeros((1, 127200))
for i in range(127200):
    max_input_train_windowed[:, i] = np.max(np.abs(input_train_windowed_reshaped[i, :]))
    if max_input_train_windowed[:, i] == 0:
        max_input_train_windowed[:, i] = 0.0000001

input_train_windowed_reshaped_normalized = np.zeros(shape=(42400 * 3, win))

for i in range(127200):
    input_train_windowed_reshaped_normalized[i, :] = (input_train_windowed_reshaped[i, :]) / (
    max_input_train_windowed[:, i])

np.save('E:/resnet stuff/max_input_train_windowed', max_input_train_windowed)

import h5py

f = h5py.File('E:/resnet stuff/input_train_windowed_normalized.h5', 'w')

f.create_dataset('input_train', data=input_train_windowed_reshaped_normalized)

f.close()

spikes_train = np.load('E:/resnet stuff/spikes_train.npy')
win = int(spikes_train.shape[1] / 3)
spikes_train_windowed = np.zeros((42400, 3, win))
for i in range(42400):
    for j in range(3):
        spikes_train_windowed[i, j, :] = spikes_train[i, j * win:j * win + win]

spikes_train_windowed_reshaped = np.zeros((42400 * 3, win))

spikes_train_windowed_reshaped = np.reshape(spikes_train_windowed, (42400 * 3, win))

max_spikes_train_windowed = np.zeros((1, 127200))
for i in range(127200):
    max_spikes_train_windowed[:, i] = np.max(np.abs(spikes_train_windowed_reshaped[i, :]))
    if max_spikes_train_windowed[:, i] == 0:
        max_spikes_train_windowed[:, i] = 0.0000001

spikes_train_windowed_reshaped_normalized = np.zeros(shape=(42400 * 3, win))

for i in range(127200):
    spikes_train_windowed_reshaped_normalized[i, :] = (spikes_train_windowed_reshaped[i, :]) / (
    max_spikes_train_windowed[:, i])

np.save('E:/resnet stuff/max_spikes_train_windowed', max_spikes_train_windowed)

import h5py

f = h5py.File('E:/resnet stuff/spikes_train_windowed_reshaped_normalized.h5', 'w')

f.create_dataset('spikes_train', data=spikes_train_windowed_reshaped_normalized)

f.close()

import numpy as np

input_test = np.load('E:/resnet stuff/input_test.npy')
win = int(input_test.shape[1] / 3)
input_test_windowed = np.zeros((15900, 3, win))
for i in range(15900):
    for j in range(3):
        input_test_windowed[i, j, :] = input_test[i, j * win:j * win + win]

input_test_windowed_reshaped = np.zeros((15900 * 3, win))

input_test_windowed_reshaped = np.reshape(input_test_windowed, (15900 * 3, win))

max_input_test_windowed = np.zeros((1, 47700))
for i in range(47700):
    max_input_test_windowed[:, i] = np.max(np.abs(input_test_windowed_reshaped[i, :]))
    if max_input_test_windowed[:, i] == 0:
        max_input_test_windowed[:, i] = 0.0000001

input_test_windowed_reshaped_normalized = np.zeros(shape=(15900 * 3, win))

for i in range(47700):
    input_test_windowed_reshaped_normalized[i, :] = (input_test_windowed_reshaped[i, :]) / (
    max_input_test_windowed[:, i])

np.save('E:/resnet stuff/max_input_test_windowed', max_input_test_windowed)

import h5py

f = h5py.File('E:/resnet stuff/input_test_windowed_reshaped_normalized.h5', 'w')

f.create_dataset('input_test', data=input_test_windowed_reshaped_normalized)

f.close()

import numpy as np

spikes_test = np.load('E:/resnet stuff/spikes_test.npy')
win = int(spikes_test.shape[1] / 3)
spikes_test_windowed = np.zeros((15900, 3, win))
for i in range(15900):
    for j in range(3):
        spikes_test_windowed[i, j, :] = spikes_test[i, j * win:j * win + win]

spikes_test_windowed_reshaped = np.zeros((15900 * 3, win))

spikes_test_windowed_reshaped = np.reshape(spikes_test_windowed, (15900 * 3, win))

max_spikes_test_windowed = np.zeros((1, 47700))
for i in range(47700):
    max_spikes_test_windowed[:, i] = np.max(np.abs(spikes_test_windowed_reshaped[i, :]))
    if max_spikes_test_windowed[:, i] == 0:
        max_spikes_test_windowed[:, i] = 0.0000001

spikes_test_windowed_reshaped_normalized = np.zeros(shape=(15900 * 3, win))

for i in range(47700):
    spikes_test_windowed_reshaped_normalized[i, :] = (spikes_test_windowed_reshaped[i, :]) / (
    max_spikes_test_windowed[:, i])

np.save('E:/resnet stuff/max_spikes_test_windowed', max_spikes_test_windowed)

import h5py

f = h5py.File('E:/resnet stuff/spikes_test_windowed_reshaped_normalized.h5', 'w')

f.create_dataset('spikes_test', data=spikes_test_windowed_reshaped_normalized)

f.close()

###########################################################
# noisy model   wrong normalization
##########################################################
import numpy as np
import scipy.io as spio
import h5py

arrays = {}

f = h5py.File('D:/data/ICC/spikes/RawData/Raw_all_noisy_scream.mat')
for k, v in f.items():
    arrays[k] = np.array(v)

spikes_scream = arrays['Raw_all_noisy_scream']  # size=35160,40,256,12
spikes_scream_mean = np.mean(spikes_scream, 1)

spikes_scream_truncated = spikes_scream_mean[0:23928, :, :]
spikes_scream_truncated = np.transpose(spikes_scream_truncated, (1, 2, 0))

f = h5py.File('D:/data/ICC/spikes/RawData/Raw_all_noisy_tooth.mat')
for k, v in f.items():
    arrays[k] = np.array(v)

spikes_tooth = arrays['Raw_all_noisy_tooth']  # size=35160,40,256,12
spikes_tooth_mean = np.mean(spikes_tooth, 1)
spikes_tooth_truncated = spikes_tooth_mean[0:23928, :, :]
spikes_tooth_truncated = np.transpose(spikes_tooth_truncated, (1, 2, 0))

spikes_concat = np.concatenate((spikes_scream_truncated, spikes_tooth_truncated), axis=1)

spikes_noisy_train = np.zeros(shape=(256 * 24, 23928))
spikes_noisy_train = np.reshape(spikes_concat, (256 * 24, 23928))

max_spikes_noisy_train = np.zeros((1, 6144))
for i in range(6144):
    max_spikes_noisy_train[:, i] = np.max(np.abs(spikes_noisy_train[i, :]))
    if max_spikes_noisy_train[:, i] == 0:
        max_spikes_noisy_train[:, i] = 0.0000001

spikes_noisy_train_normalized = np.zeros(shape=(6144, 23928))
for i in range(6144):
    spikes_noisy_train_normalized[i, :] = (spikes_noisy_train[i, :]) / (max_spikes_noisy_train[:, i])

win = int(spikes_noisy_train_normalized.shape[1] / 3)
spikes_noisy_train_windowed = np.zeros((6144, 3, win))
for i in range(6144):
    for j in range(3):
        spikes_noisy_train_windowed[i, j, :] = spikes_noisy_train_normalized[i, j * win:j * win + win]

spikes_noisy_train_windowed_reshaped = np.zeros((6144 * 3, win))

spikes_noisy_train_windowed_reshaped = np.reshape(spikes_noisy_train_windowed, (6144 * 3, win))

f = h5py.File('D:/data/ICC/spikes/RawData/spikes_noisy_train_windowed_reshaped.h5', 'w')

f.create_dataset('spikes_train', data=spikes_noisy_train_windowed_reshaped)

f.close()

f = h5py.File('D:/data/ICC/spikes/RawData/Raw_all_noisy_squeal.mat')
arrays = {}
for k, v in f.items():
    arrays[k] = np.array(v)

spikes_squeal = arrays['Raw_all_noisy_squeal']  # size=35160,40,256,12
spikes_squeal_mean = np.mean(spikes_squeal, 1)
spikes_squeal_truncated = spikes_squeal_mean[0:23928, :, :]
spikes_squeal_truncated = np.transpose(spikes_squeal_truncated, (1, 2, 0))

spikes_noisy_test = np.zeros(shape=(256 * 12, 23928))
spikes_noisy_test = np.reshape(spikes_squeal_truncated, (256 * 12, 23928))

max_spikes_noisy_test = np.zeros((1, 3072))
for i in range(3072):
    max_spikes_noisy_test[:, i] = np.max(np.abs(spikes_noisy_test[i, :]))
    if max_spikes_noisy_test[:, i] == 0:
        max_spikes_noisy_test[:, i] = 0.0000001

spikes_noisy_test_normalized = np.zeros(shape=(3072, 23928))
for i in range(3072):
    spikes_noisy_test_normalized[i, :] = (spikes_noisy_test[i, :]) / (max_spikes_noisy_test[:, i])

win = int(spikes_noisy_test_normalized.shape[1] / 3)
spikes_noisy_test_windowed = np.zeros((3072, 3, win))
for i in range(3072):
    for j in range(3):
        spikes_noisy_test_windowed[i, j, :] = spikes_noisy_test_normalized[i, j * win:j * win + win]

spikes_noisy_test_windowed_reshaped = np.zeros((3072 * 3, win))

spikes_noisy_test_windowed_reshaped = np.reshape(spikes_noisy_test_windowed, (3072 * 3, win))

f = h5py.File('D:/data/ICC/spikes/RawData/spikes_noisy_test_windowed_reshaped.h5', 'w')

f.create_dataset('spikes_test', data=spikes_noisy_test_windowed_reshaped)

f.close()

import scipy.io as sio

path = 'D:/data/ICC/stim_orig.mat'
mat = sio.loadmat(path)
input_all = mat['original_stim']

input_scream_noisy = input_all[:, 0:12]

input_squeal_noisy = input_all[:, 24:36]

input_tooth_noisy = input_all[:, 48:60]

input_train_noisy = np.concatenate((input_scream_noisy, input_tooth_noisy), axis=1)

input_scream_clean = input_all[:, 72:74]

input_squeal_clean = input_all[:, 74:76]

input_tooth_clean = input_all[:, 76:78]

input_train_noisy = input_train_noisy[:, :, np.newaxis]
list_train = [input_train_noisy] * 256
train_input_noisy = np.concatenate(list_train, axis=2)
train_input_noisy = np.transpose(train_input_noisy, (2, 1, 0))
train_input_noisy_short = train_input_noisy[:, :, 195312:195312 + 191406]
train_input_noisy_short_resampled = train_input_noisy_short[:, :, ::2]

train_input_noisy_short_resampled_reshaped = np.zeros(shape=(256 * 24, 95703))
train_input_noisy_short_resampled_reshaped = np.reshape(train_input_noisy_short_resampled, (256 * 24, 95703))

max_input_noisy_train = np.zeros((1, 6144))
for i in range(6144):
    max_input_noisy_train[:, i] = np.max(np.abs(train_input_noisy_short_resampled_reshaped[i, :]))
    if max_input_noisy_train[:, i] == 0:
        max_input_noisy_train[:, i] = 0.0000001
input_noisy_train_normalized = np.zeros(shape=(6144, 95703))
for i in range(6144):
    input_noisy_train_normalized[i, :] = (train_input_noisy_short_resampled_reshaped[i, :]) / (
    max_input_noisy_train[:, i])

win = int(95703 / 3)

input_noisy_train_windowed = np.zeros((6144, 3, win))
for i in range(6144):
    for j in range(3):
        input_noisy_train_windowed[i, j, :] = input_noisy_train_normalized[i, j * win:j * win + win]

input_noisy_train_windowed_reshaped = np.zeros((6144 * 3, win))

input_noisy_train_windowed_reshaped = np.reshape(input_noisy_train_windowed, (6144 * 3, win))

f = h5py.File('D:/data/ICC/spikes/RawData/input_noisy_train.h5', 'w')

f.create_dataset('input_train', data=input_noisy_train_windowed_reshaped)

f.close()

input_test_noisy = input_squeal_noisy
input_test_noisy = input_test_noisy[:, :, np.newaxis]
list_test = [input_test_noisy] * 256
test_input_noisy = np.concatenate(list_test, axis=2)
test_input_noisy = np.transpose(test_input_noisy, (2, 1, 0))
test_input_noisy_short = test_input_noisy[:, :, 195312:195312 + 191406]
test_input_noisy_short_resampled = test_input_noisy_short[:, :, ::2]

test_input_noisy_short_resampled_reshaped = np.zeros(shape=(256 * 12, 95703))
test_input_noisy_short_resampled_reshaped = np.reshape(test_input_noisy_short_resampled, (256 * 12, 95703))

max_input_noisy_test = np.zeros((1, 3072))
for i in range(3072):
    max_input_noisy_test[:, i] = np.max(np.abs(test_input_noisy_short_resampled_reshaped[i, :]))
    if max_input_noisy_test[:, i] == 0:
        max_input_noisy_test[:, i] = 0.0000001
input_noisy_test_normalized = np.zeros(shape=(3072, 95703))
for i in range(3072):
    input_noisy_test_normalized[i, :] = (test_input_noisy_short_resampled_reshaped[i, :]) / (max_input_noisy_test[:, i])

win = int(95703 / 3)

input_noisy_test_windowed = np.zeros((3072, 3, win))
for i in range(3072):
    for j in range(3):
        input_noisy_test_windowed[i, j, :] = input_noisy_test_normalized[i, j * win:j * win + win]

input_noisy_test_windowed_reshaped = np.zeros((3072 * 3, win))

input_noisy_test_windowed_reshaped = np.reshape(input_noisy_test_windowed, (3072 * 3, win))

f = h5py.File('D:/data/ICC/spikes/RawData/input_noisy_test_windowed_reshaped.h5', 'w')

f.create_dataset('input_test', data=input_noisy_test_windowed_reshaped)

f.close()

input_squeal_clean_55 = input_squeal_clean[195312:195312 + 191406, 0:1]
input_squeal_clean_65 = input_squeal_clean[195312:195312 + 191406, 1:2]

list_squeal_clean_55 = [input_squeal_clean_55] * 2
input_squeal_clean_55_2 = np.concatenate(list_squeal_clean_55, axis=1)

list_squeal_clean_65 = [input_squeal_clean_65] * 2
input_squeal_clean_65_2 = np.concatenate(list_squeal_clean_65, axis=1)

input_squeal_clean_all = np.concatenate((input_squeal_clean_55_2, input_squeal_clean_65_2), axis=1)

list_squeal = [input_squeal_clean_all] * 3
input_test_clean = np.concatenate(list_squeal, axis=1)

input_scream_clean_55 = input_scream_clean[195312:195312 + 191406, 0:1]
input_scream_clean_65 = input_scream_clean[195312:195312 + 191406, 1:2]

list_scream_clean_55 = [input_scream_clean_55] * 2
input_scream_clean_55_2 = np.concatenate(list_scream_clean_55, axis=1)

list_scream_clean_65 = [input_scream_clean_65] * 2
input_scream_clean_65_2 = np.concatenate(list_scream_clean_65, axis=1)

input_scream_clean_all = np.concatenate((input_scream_clean_55_2, input_scream_clean_65_2), axis=1)

list_scream = [input_scream_clean_all] * 3
input_scream_clean_all_12 = np.concatenate(list_scream, axis=1)

input_tooth_clean_55 = input_tooth_clean[195312:195312 + 191406, 0:1]
input_tooth_clean_65 = input_tooth_clean[195312:195312 + 191406, 1:2]

list_tooth_clean_55 = [input_tooth_clean_55] * 2
input_tooth_clean_55_2 = np.concatenate(list_tooth_clean_55, axis=1)

list_tooth_clean_65 = [input_tooth_clean_65] * 2
input_tooth_clean_65_2 = np.concatenate(list_tooth_clean_65, axis=1)

input_tooth_clean_all = np.concatenate((input_tooth_clean_55_2, input_tooth_clean_65_2), axis=1)

list_tooth = [input_tooth_clean_all] * 3
input_tooth_clean_all_12 = np.concatenate(list_tooth, axis=1)

input_train_clean = np.concatenate((input_scream_clean_all_12, input_tooth_clean_all_12), axis=1)

input_train_clean = input_train_clean[:, :, np.newaxis]
list_train = [input_train_clean] * 256
input_train_clean_all = np.concatenate(list_train, axis=2)
input_train_clean_all = np.transpose(input_train_clean_all, (2, 1, 0))

input_train_clean_reshaped = np.zeros(shape=(256 * 24, 191406))
input_train_clean_reshaped = np.reshape(input_train_clean_all, (256 * 24, 191406))
input_train_clean_reshaped = input_train_clean_reshaped[:, ::2]

max_input_clean_train = np.zeros((1, 6144))
for i in range(6144):
    max_input_clean_train[:, i] = np.max(np.abs(input_train_clean_reshaped[i, :]))
    if max_input_clean_train[:, i] == 0:
        max_input_clean_train[:, i] = 0.0000001
input_clean_train_normalized = np.zeros(shape=(6144, 95703))
for i in range(6144):
    input_clean_train_normalized[i, :] = (input_train_clean_reshaped[i, :]) / (max_input_clean_train[:, i])

win = int(95703 / 3)

input_clean_train_windowed = np.zeros((6144, 3, win))
for i in range(6144):
    for j in range(3):
        input_clean_train_windowed[i, j, :] = input_clean_train_normalized[i, j * win:j * win + win]

input_clean_train_windowed_reshaped = np.zeros((6144 * 3, win))

input_clean_train_windowed_reshaped = np.reshape(input_clean_train_windowed, (6144 * 3, win))

f = h5py.File('D:/data/ICC/spikes/RawData/input_clean_train.h5', 'w')

f.create_dataset('input_train', data=input_clean_train_windowed_reshaped)

f.close()

input_test_clean = input_test_clean[:, :, np.newaxis]
list_test = [input_test_clean] * 256
input_test_clean_all = np.concatenate(list_test, axis=2)
input_test_clean_all = np.transpose(input_test_clean_all, (2, 1, 0))

input_test_clean_reshaped = np.zeros(shape=(256 * 12, 191406))
input_test_clean_reshaped = np.reshape(input_test_clean_all, (256 * 12, 191406))
input_test_clean_reshaped = input_test_clean_reshaped[:, ::2]

max_input_clean_test = np.zeros((1, 3072))
for i in range(3072):
    max_input_clean_test[:, i] = np.max(np.abs(input_test_clean_reshaped[i, :]))
    if max_input_clean_test[:, i] == 0:
        max_input_clean_test[:, i] = 0.0000001
input_clean_test_normalized = np.zeros(shape=(3072, 95703))
for i in range(3072):
    input_clean_test_normalized[i, :] = (input_test_clean_reshaped[i, :]) / (max_input_clean_test[:, i])

win = int(95703 / 3)

input_clean_test_windowed = np.zeros((3072, 3, win))
for i in range(3072):
    for j in range(3):
        input_clean_test_windowed[i, j, :] = input_clean_test_normalized[i, j * win:j * win + win]

input_clean_test_windowed_reshaped = np.zeros((3072 * 3, win))

input_clean_test_windowed_reshaped = np.reshape(input_clean_test_windowed, (3072 * 3, win))

f = h5py.File('D:/data/ICC/spikes/RawData/input_clean_test.h5', 'w')

f.create_dataset('input_test', data=input_clean_test_windowed_reshaped)

f.close()

########################################################################
#### to divide all by maximum of the whole matrix
#########################################################################

# thilos data


import scipy.io as spio
import numpy as np

mat = spio.loadmat('D:/data_workFolder/deep_guinea_ears/data/data_rawData/in samples/Matlab Data/input.mat')
input_train = mat['input_train']  # shape: (95703,8)
input_test = mat['input_test']  # shape: (95703,3)

input_test = input_test[:, :, np.newaxis, np.newaxis]  # shape: 95703,3,1,1,
list_test = [input_test] * 20
input_test = np.concatenate(list_test, axis=2)  # shape: 95703,3,20,1,

list_test = [input_test] * 265
input_test = np.concatenate(list_test, axis=3)  ##shape: 95703,3,20,265

input_train = input_train[:, :, np.newaxis, np.newaxis]

list_train = [input_train] * 20
input_train = np.concatenate(list_train, axis=2)
list_train = [input_train] * 265
input_train = np.concatenate(list_train, axis=3)

input_test = np.transpose(input_test, (2, 3, 1, 0))  # shape: 20,265,3,95703
input_train = np.transpose(input_train, (2, 3, 1, 0))

input_test_reshaped = np.zeros(shape=(20 * 265 * 3, 95703))
input_test_reshaped = np.reshape(input_test, (20 * 265 * 3, 95703))  # shape: 15900,95703

input_train_reshaped = np.zeros(shape=(20 * 265 * 8, 95703))
input_train_reshaped = np.reshape(input_train, (20 * 265 * 8, 95703))

max_input_test = np.max(np.abs(input_test_reshaped))

np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/max_input_test',
        max_input_test)

input_test_normalized = np.zeros(shape=(15900, 95703))

input_test_normalized = input_test_reshaped / max_input_test

max_input_train = np.max(np.abs(input_train_reshaped))

np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/max_input_train',
        max_input_train)

input_train_normalized = np.zeros(shape=(42400, 95703))

input_train_normalized = input_train_reshaped / max_input_train

input_test_cut = np.zeros(shape=(input_test_normalized.shape[0], 3, int(input_test_normalized.shape[1] / 3)))
win = int(input_test_normalized.shape[1] / 3)
for i in range(15900):
    for j in range(3):
        input_test_cut[i, j, :] = input_test_normalized[i, j * win:j * win + win]  # shape: 15900,3,31901

input_train_cut = np.zeros(shape=(input_train_normalized.shape[0], 3, int(input_train_normalized.shape[1] / 3)))
win = int(input_train_normalized.shape[1] / 3)
for i in range(input_train_normalized.shape[0]):
    for j in range(3):
        input_train_cut[i, j, :] = input_train_normalized[i, j * win:j * win + win]

input_test = np.zeros(shape=(15900 * 3, 31901))  # shape: 47700,31901
input_test = np.reshape(input_test_cut, (15900 * 3, 31901))

f = h5py.File('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/input_test.h5',
              'w')

f.create_dataset('input_test', data=input_test)

f.close()

input_train = np.zeros(shape=(input_train_cut.shape[0] * 3, 31901))
input_train = np.reshape(input_train_cut, (input_train_cut.shape[0] * 3, 31901))

f = h5py.File('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/input_train.h5',
              'w')

f.create_dataset('input_train', data=input_train)

f.close()

arrays = {}
f = h5py.File('D:/data_workFolder/deep_guinea_ears/data/data_rawData/in samples/Matlab Data/all.mat', 'r')
for k, v in f.items():
    arrays[k] = np.array(v)

spikes = arrays['all']  # all has a shape of (11,23928,20,265)
# spikes=np.transpose(spikes,(1,2,3,0))
spikes_test = np.zeros(shape=(3, 23928, 20, 265))  # shape of (3,23928,20,265)
spikes_train = np.zeros(shape=(8, 23928, 20, 265))

spikes_test[0:1, :, :, :] = spikes[0:1, :, :, :]
spikes_train[0:5, :, :, :] = spikes[1:6, :, :, :]
spikes_test[1:3, :, :, :] = spikes[6:8, :, :, :]
spikes_train[5:8, :, :, :] = spikes[8:11, :, :, :]

spikes_test = np.transpose(spikes_test, (2, 3, 0, 1))  # shape: 20,265,3,23928

spikes_train = np.transpose(spikes_train, (2, 3, 0, 1))

spikes_test_reshaped = np.zeros(shape=(20 * 265 * 3, 23928))  # shape: 15900,23928
spikes_test_reshaped = np.reshape(spikes_test, (15900, 23928))

spikes_train_reshaped = np.zeros(shape=(20 * 265 * 8, 23928))  # shape: 42400,23928
spikes_train_reshaped = np.reshape(spikes_train, (20 * 265 * 8, 23928))

max_spikes_test = np.max(np.abs(spikes_test_reshaped))
spikes_test_normalized = spikes_test_reshaped / max_spikes_test
np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/max_spikes_test',
        max_spikes_test)

spikes_test_cut = np.zeros(shape=(spikes_test_normalized.shape[0], 3, int(spikes_test_normalized.shape[1] / 3)))
win = int(spikes_test_normalized.shape[1] / 3)
for i in range(15900):
    for j in range(3):
        spikes_test_cut[i, j, :] = spikes_test_normalized[i, j * win:j * win + win]  # shape: 15900,3,31901

spikes_test = np.zeros(shape=(15900 * 3, 7976))
spikes_test = np.reshape(spikes_test_cut, (15900 * 3, 7976))

f = h5py.File('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/spikes_test.h5',
              'w')
f.create_dataset('spikes_test', data=spikes_test)
f.close()

max_spikes_train = np.max(np.abs(spikes_train_reshaped))
spikes_train_normalized = spikes_train_reshaped / max_spikes_train
np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/max_spikes_train',
        max_spikes_train)

spikes_train_cut = np.zeros(shape=(spikes_train_normalized.shape[0], 3, int(spikes_train_normalized.shape[1] / 3)))
win = int(spikes_train_normalized.shape[1] / 3)
for i in range(42400):
    for j in range(3):
        spikes_train_cut[i, j, :] = spikes_train_normalized[i, j * win:j * win + win]  # shape: 15900,3,31901

spikes_train = np.zeros(shape=(42400 * 3, 7976))
spikes_train = np.reshape(spikes_train_cut, (42400 * 3, 7976))

f = h5py.File(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/spikes_train.h5', 'w')
f.create_dataset('spikes_train', data=spikes_train)
f.close()

# my data
import numpy as np
import scipy.io as spio
import h5py

f = h5py.File('D:/data/ICC/spikes/RawData/Raw_all_noisy_squeal.mat')
arrays = {}
for k, v in f.items():
    arrays[k] = np.array(v)

spikes_squeal = arrays['Raw_all_noisy_squeal']  # size=35160,40,256,12
spikes_squeal_mean = np.mean(spikes_squeal, 1)  # 35160,256,12
spikes_squeal_truncated = spikes_squeal_mean[0:23928, :, :]  # 23928,256,12
spikes_squeal_truncated = np.transpose(spikes_squeal_truncated, (1, 2, 0))  # 256,12,23928

spikes_noisy_test = np.zeros(shape=(256 * 12, 23928))
spikes_noisy_test = np.reshape(spikes_squeal_truncated, (256 * 12, 23928))  # 3072,23928

max_spikes_noisy_test = np.max(np.abs(spikes_noisy_test))

spikes_noisy_test_normalized = np.zeros(shape=(3072, 23928))
spikes_noisy_test_normalized = spikes_noisy_test / max_spikes_noisy_test  # 3072,23928

win = int(spikes_noisy_test_normalized.shape[1] / 3)
spikes_noisy_test_windowed = np.zeros((3072, 3, win))  # 3072,3,7976
for i in range(3072):
    for j in range(3):
        spikes_noisy_test_windowed[i, j, :] = spikes_noisy_test_normalized[i, j * win:j * win + win]

spikes_noisy_test_windowed_reshaped = np.zeros((3072 * 3, win))  # 3072*3,7976

spikes_noisy_test_windowed_reshaped = np.reshape(spikes_noisy_test_windowed, (3072 * 3, win))

f = h5py.File(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/spikes_noisy_test.h5',
    'w')

f.create_dataset('spikes_test', data=spikes_noisy_test_windowed_reshaped)

f.close()

np.save(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/max_spikes_noisy_test',
    max_spikes_noisy_test)

arrays = {}
f = h5py.File('D:/data/ICC/spikes/RawData/Raw_all_noisy_scream.mat')
for k, v in f.items():
    arrays[k] = np.array(v)

spikes_scream = arrays['Raw_all_noisy_scream']  # size=35160,40,256,12
spikes_scream_mean = np.mean(spikes_scream, 1)  # size=35160,256,12

spikes_scream_truncated = spikes_scream_mean[0:23928, :, :]  # size=23928,256,12
spikes_scream_truncated = np.transpose(spikes_scream_truncated, (1, 2, 0))

arrays = {}
f = h5py.File('D:/data/ICC/spikes/RawData/Raw_all_noisy_tooth.mat')
for k, v in f.items():
    arrays[k] = np.array(v)

spikes_tooth = arrays['Raw_all_noisy_tooth']  # size=35160,40,256,12
spikes_tooth_mean = np.mean(spikes_tooth, 1)  # size=35160,256,12

spikes_tooth_truncated = spikes_tooth_mean[0:23928, :, :]  # size=23928,256,12
spikes_tooth_truncated = np.transpose(spikes_tooth_truncated, (1, 2, 0))  # size=256,12,23928

spikes_concat = np.concatenate((spikes_scream_truncated, spikes_tooth_truncated), axis=1)  # size=256,24,23928

spikes_noisy_train = np.zeros(shape=(256 * 24, 23928))  # size=256*24,23928
spikes_noisy_train = np.reshape(spikes_concat, (256 * 24, 23928))

max_spikes_noisy_train = np.max(np.abs(spikes_noisy_train))

np.save(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/max_spikes_noisy_train',
    max_spikes_noisy_train)

spikes_noisy_train_normalized = np.zeros(shape=(6144, 23928))

spikes_noisy_train_normalized = spikes_noisy_train / max_spikes_noisy_train

win = int(spikes_noisy_train_normalized.shape[1] / 3)
spikes_noisy_train_windowed = np.zeros((6144, 3, win))
for i in range(6144):
    for j in range(3):
        spikes_noisy_train_windowed[i, j, :] = spikes_noisy_train_normalized[i, j * win:j * win + win]

spikes_noisy_train_windowed_reshaped = np.zeros((6144 * 3, win))
win = 7976
spikes_noisy_train_windowed_reshaped = np.reshape(spikes_noisy_train_windowed, (6144 * 3, win))

f = h5py.File(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/spikes_noisy_train.h5',
    'w')

f.create_dataset('spikes_train', data=spikes_noisy_train_windowed_reshaped)

f.close()

import scipy.io as sio

path = 'D:/data/ICC/stim_orig.mat'
mat = sio.loadmat(path)
input_all = mat['original_stim']

input_scream_noisy = input_all[:, 0:12]

input_squeal_noisy = input_all[:, 24:36]

input_tooth_noisy = input_all[:, 48:60]

input_train_noisy = np.concatenate((input_scream_noisy, input_tooth_noisy), axis=1)

input_scream_clean = input_all[:, 72:74]

input_squeal_clean = input_all[:, 74:76]

input_tooth_clean = input_all[:, 76:78]

input_train_noisy = input_train_noisy[:, :, np.newaxis]
list_train = [input_train_noisy] * 256
train_input_noisy = np.concatenate(list_train, axis=2)  # 407070,24,256
train_input_noisy = np.transpose(train_input_noisy, (2, 1, 0))  # 256,24,407070
train_input_noisy_short = train_input_noisy[:, :, 195312:195312 + 191406]  # 256,24,191406
train_input_noisy_short_resampled = train_input_noisy_short[:, :, ::2]  # 256,24,95703

train_input_noisy_short_resampled_reshaped = np.zeros(shape=(256 * 24, 95703))
train_input_noisy_short_resampled_reshaped = np.reshape(train_input_noisy_short_resampled, (256 * 24, 95703))

max_input_noisy_train = np.max(np.abs(train_input_noisy_short_resampled_reshaped))
np.save(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/max_input_noisy_train',
    max_input_noisy_train)

input_noisy_train_normalized = np.zeros(shape=(6144, 95703))
input_noisy_train_normalized = train_input_noisy_short_resampled_reshaped / max_input_noisy_train

win = int(95703 / 3)

input_noisy_train_windowed = np.zeros((6144, 3, win))
for i in range(6144):
    for j in range(3):
        input_noisy_train_windowed[i, j, :] = input_noisy_train_normalized[i, j * win:j * win + win]

input_noisy_train_windowed_reshaped = np.zeros((6144 * 3, win))

input_noisy_train_windowed_reshaped = np.reshape(input_noisy_train_windowed, (6144 * 3, win))

f = h5py.File(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/input_noisy_train.h5',
    'w')

f.create_dataset('input_train', data=input_noisy_train_windowed_reshaped)

f.close()

input_test_noisy = input_squeal_noisy
input_test_noisy = input_test_noisy[:, :, np.newaxis]
list_test = [input_test_noisy] * 256
test_input_noisy = np.concatenate(list_test, axis=2)
test_input_noisy = np.transpose(test_input_noisy, (2, 1, 0))
test_input_noisy_short = test_input_noisy[:, :, 195312:195312 + 191406]
test_input_noisy_short_resampled = test_input_noisy_short[:, :, ::2]

test_input_noisy_short_resampled_reshaped = np.zeros(shape=(256 * 12, 95703))
test_input_noisy_short_resampled_reshaped = np.reshape(test_input_noisy_short_resampled, (256 * 12, 95703))

max_input_noisy_test = np.max(np.abs(test_input_noisy_short_resampled_reshaped))
np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/max_input_noisy_test',
        max_input_noisy_test)

input_noisy_test_normalized = np.zeros(shape=(3072, 95703))
input_noisy_test_normalized = test_input_noisy_short_resampled_reshaped / max_input_noisy_test

win = int(95703 / 3)

input_noisy_test_windowed = np.zeros((3072, 3, win))
for i in range(3072):
    for j in range(3):
        input_noisy_test_windowed[i, j, :] = input_noisy_test_normalized[i, j * win:j * win + win]

input_noisy_test_windowed_reshaped = np.zeros((3072 * 3, win))

input_noisy_test_windowed_reshaped = np.reshape(input_noisy_test_windowed, (3072 * 3, win))

f = h5py.File(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/input_noisy_test.h5', 'w')

f.create_dataset('input_test', data=input_noisy_test_windowed_reshaped)

f.close()

input_squeal_clean_55 = input_squeal_clean[195312:195312 + 191406, 0:1]
input_squeal_clean_65 = input_squeal_clean[195312:195312 + 191406, 1:2]

list_squeal_clean_55 = [input_squeal_clean_55] * 2
input_squeal_clean_55_2 = np.concatenate(list_squeal_clean_55, axis=1)

list_squeal_clean_65 = [input_squeal_clean_65] * 2
input_squeal_clean_65_2 = np.concatenate(list_squeal_clean_65, axis=1)

input_squeal_clean_all = np.concatenate((input_squeal_clean_55_2, input_squeal_clean_65_2), axis=1)

list_squeal = [input_squeal_clean_all] * 3
input_test_clean = np.concatenate(list_squeal, axis=1)

input_scream_clean_55 = input_scream_clean[195312:195312 + 191406, 0:1]
input_scream_clean_65 = input_scream_clean[195312:195312 + 191406, 1:2]

list_scream_clean_55 = [input_scream_clean_55] * 2
input_scream_clean_55_2 = np.concatenate(list_scream_clean_55, axis=1)

list_scream_clean_65 = [input_scream_clean_65] * 2
input_scream_clean_65_2 = np.concatenate(list_scream_clean_65, axis=1)

input_scream_clean_all = np.concatenate((input_scream_clean_55_2, input_scream_clean_65_2), axis=1)

list_scream = [input_scream_clean_all] * 3
input_scream_clean_all_12 = np.concatenate(list_scream, axis=1)

input_tooth_clean_55 = input_tooth_clean[195312:195312 + 191406, 0:1]
input_tooth_clean_65 = input_tooth_clean[195312:195312 + 191406, 1:2]

list_tooth_clean_55 = [input_tooth_clean_55] * 2
input_tooth_clean_55_2 = np.concatenate(list_tooth_clean_55, axis=1)

list_tooth_clean_65 = [input_tooth_clean_65] * 2
input_tooth_clean_65_2 = np.concatenate(list_tooth_clean_65, axis=1)

input_tooth_clean_all = np.concatenate((input_tooth_clean_55_2, input_tooth_clean_65_2), axis=1)

list_tooth = [input_tooth_clean_all] * 3
input_tooth_clean_all_12 = np.concatenate(list_tooth, axis=1)

input_train_clean = np.concatenate((input_scream_clean_all_12, input_tooth_clean_all_12), axis=1)

input_train_clean = input_train_clean[:, :, np.newaxis]
list_train = [input_train_clean] * 256
input_train_clean_all = np.concatenate(list_train, axis=2)
input_train_clean_all = np.transpose(input_train_clean_all, (2, 1, 0))

input_train_clean_reshaped = np.zeros(shape=(256 * 24, 191406))
input_train_clean_reshaped = np.reshape(input_train_clean_all, (256 * 24, 191406))
input_train_clean_reshaped = input_train_clean_reshaped[:, ::2]

max_input_clean_train = np.max(np.abs(input_train_clean_reshaped))
np.save(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/max_input_clean_train',
    max_input_clean_train)

input_clean_train_normalized = np.zeros(shape=(6144, 95703))
input_clean_train_normalized = input_train_clean_reshaped / max_input_clean_train

win = int(95703 / 3)

input_clean_train_windowed = np.zeros((6144, 3, win))
for i in range(6144):
    for j in range(3):
        input_clean_train_windowed[i, j, :] = input_clean_train_normalized[i, j * win:j * win + win]

input_clean_train_windowed_reshaped = np.zeros((6144 * 3, win))

input_clean_train_windowed_reshaped = np.reshape(input_clean_train_windowed, (6144 * 3, win))

f = h5py.File(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/input_clean_train.h5',
    'w')

f.create_dataset('input_train', data=input_clean_train_windowed_reshaped)

f.close()

# input_test_clean=input_squeal_clean
input_test_clean = input_test_clean[:, :, np.newaxis]
list_test = [input_test_clean] * 256
input_test_clean_all = np.concatenate(list_test, axis=2)
input_test_clean_all = np.transpose(input_test_clean_all, (2, 1, 0))

input_test_clean_reshaped = np.zeros(shape=(256 * 12, 191406))
input_test_clean_reshaped = np.reshape(input_test_clean_all, (256 * 12, 191406))
input_test_clean_reshaped = input_test_clean_reshaped[:, ::2]

max_input_clean_test = np.max(np.abs(input_test_clean_reshaped))
np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized/max_input_clean_test',
        max_input_clean_test)

input_clean_test_normalized = np.zeros(shape=(3072, 95703))

input_clean_test_normalized = input_test_clean_reshaped / max_input_clean_test
win = int(95703 / 3)

input_clean_test_windowed = np.zeros((3072, 3, win))
for i in range(3072):
    for j in range(3):
        input_clean_test_windowed[i, j, :] = input_clean_test_normalized[i, j * win:j * win + win]

input_clean_test_windowed_reshaped = np.zeros((3072 * 3, win))

input_clean_test_windowed_reshaped = np.reshape(input_clean_test_windowed, (3072 * 3, win))

f = h5py.File(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/time_domain/new_allMatrixNormalized//input_clean_test.h5',
    'w')

f.create_dataset('input_test', data=input_clean_test_windowed_reshaped)

f.close()

########################################### 23-3-2020
# this is the part to use when making gammatone representation of the stimuli and
# also we have to window the spikes with the same method we are windowing the stimuli fo gammatones
from gammatone.gtgram import gtgram

import scipy.io as spio
import numpy as np

mat = spio.loadmat('C:/Users/hoss3301/work/deep_guinea_ears/data/data/data_rawData/in samples/Matlab Data/input.mat')
input_train = mat['input_train']  # shape: (95703,8)
input_test = mat['input_test']  # shape: (95703,3)

input_test = input_test[:, :, np.newaxis, np.newaxis]  # shape: 95703,3,1,1,
list_test = [input_test] * 20
input_test = np.concatenate(list_test, axis=2)  # shape: 95703,3,20,1,

list_test = [input_test] * 265
input_test = np.concatenate(list_test, axis=3)  ##shape: 95703,3,20,265

list_train = [input_train] * 20
input_train = np.concatenate(list_train, axis=2)
input_train = input_train[:, :, np.newaxis, np.newaxis]
list_train = [input_train] * 265
input_train = np.concatenate(list_train, axis=3)

input_test = np.transpose(input_test, (2, 3, 1, 0))  # shape: 20,265,3,95703
input_train = np.transpose(input_train, (2, 3, 1, 0))

input_test_reshaped = np.zeros(shape=(20 * 265 * 3, 95703))
input_test_reshaped = np.reshape(input_test, (20 * 265 * 3, 95703))  # shape: 15900,95703

input_train_reshaped = np.zeros(shape=(20 * 265 * 8, 95703))
input_train_reshaped = np.reshape(input_train, (20 * 265 * 8, 95703))

# input_test_cut=np.zeros(shape=(input_test_reshaped.shape[0],3,int(input_test_reshaped.shape[1]/3)))
# win=int(input_test_reshaped.shape[1]/3)
# for i in range(15900):
#    for j in range(3):
#        input_test_cut[i,j,:]=input_test_reshaped[i,j*win:j*win+win]    #shape: 15900,3,31901


# input_train_cut=np.zeros(shape=(input_train_reshaped.shape[0],3,int(input_train_reshaped.shape[1]/3)))
# win=int(input_train_reshaped.shape[1]/3)
# for i in range(input_train_reshaped.shape[0]):
#    for j in range(3):
#        input_train_cut[i,j,:]=input_train_reshaped[i,j*win:j*win+win]


# input_test=np.zeros(shape=(15900*3,31901))   #shape: 47700,31901
# input_test=np.reshape(input_test_cut,(15900*3,31901))

# input_train=np.zeros(shape=(input_train_cut.shape[0]*3,31901))
# input_train=np.reshape(input_train_cut,(input_train_cut.shape[0]*3,31901))

n_channels = 128

# add_samples=np.round(0.025*97656.25)
# input_test=
gamma_waves_test = np.zeros(shape=(input_test_reshaped.shape[0], n_channels, 96))  # shape: 15900,128,96
gamma_waves_train = np.zeros(shape=(input_train_reshaped.shape[0], n_channels, 96))

"""
from gammatone.gtgram import gtgram_strides
from gammatone.filters import centre_freqs, make_erb_filters


cfs = centre_freqs(97656.25, 128, 20)
fcoefs = np.flipud(make_erb_filters(97656.25, cfs))
#xf = np.zeros((fcoefs[:,9].shape[0], input_test_reshaped.shape[0],input_test_reshaped.shape[1]))
#xf = np.zeros((fcoefs[:,9].shape[0], c.shape[0], c.shape[1]))
gain = fcoefs[:, 9]
# A0, A11, A2
As1 = fcoefs[:, (0, 1, 5)]
# A0, A12, A2
As2 = fcoefs[:, (0, 2, 5)]
# A0, A13, A2
As3 = fcoefs[:, (0, 3, 5)]
# A0, A14, A2
As4 = fcoefs[:, (0, 4, 5)]
# B0, B1, B2
Bs = fcoefs[:, 6:9]
from scipy import signal as sgn
test_wave= np.zeros((128,15900, 952))
for i in range(100):
    print(i)
    xf=np.zeros((fcoefs[:,9].shape[0], int(input_test_reshaped.shape[0]/100),input_test_reshaped.shape[1]))

    
    for idx in range(0, fcoefs.shape[0]):
    
        y1 = sgn.lfilter(As1[idx], Bs[idx], input_test_reshaped[i*159:i*159+159])
        y2 = sgn.lfilter(As2[idx], Bs[idx], y1)
        y3 = sgn.lfilter(As3[idx], Bs[idx], y2)
        y4 = sgn.lfilter(As4[idx], Bs[idx], y3)
        xf[idx, ::] = y4 / gain[idx]
        
    xe = np.power(xf, 2)   
    
    nwin, hop_samples, ncols = gtgram_strides(
        97656.25,
        0.025,
        0.001,
        xe.shape[2]
    )
    

    
    for cnum in range(ncols):
        segment = xe[:,:, cnum * hop_samples + np.arange(nwin)]
        test_wave[:,i*159:i*159+159, cnum] = np.sqrt(segment.mean(2))

"""
for i in range(input_test_reshaped.shape[0]):
    gamma_waves_test[i, :, :] = gtgram(input_test_reshaped[i, :], 97656.25, 0.025, 0.01, n_channels, 20)
    if i % 10 == 0:
        print(i)
for i in range(input_train.shape[0]):
    gamma_waves_train[i, :, :] = gtgram(input_train_reshaped[i, :], 97656.25, 0.025, 0.01, n_channels, 20)
    if i % 10 == 0:
        print(i)

gamma_waves_test = np.transpose(gamma_waves_test, (0, 2, 1))  # shape: 15900,96,128
gamma_waves_train = np.transpose(gamma_waves_train, (0, 2, 1))

max_gamma_train = np.max(np.abs(gamma_waves_train))
max_gamma_test = np.max(np.abs(gamma_waves_test))

gamma_train_normalized = gamma_waves_train / max_gamma_train;
gamma_test_normalized = gamma_waves_test / max_gamma_test;

np.save('data/original/gammatone/clean/max_gamma_test', max_gamma_test)

np.save('data/original/gammatone/clean/gamma_waves_test', gamma_waves_test)

f = h5py.File('data/original/gammatone/clean/gamma_waves_test_normalized.h5', 'w')

f.create_dataset('gamma_waves_test', data=gamma_test_normalized)
f.close()

np.save('data/original/gammatone/clean/max_gamma_train', max_gamma_train)

np.save('data/original/gammatone/clean/gamma_waves_train', gamma_waves_train)

f = h5py.File('data/original/gammatone/clean/gamma_waves_train_normalized.h5', 'w')

f.create_dataset('gamma_waves_train', data=gamma_train_normalized)
f.close()

import h5py
import numpy as np

arrays = {}
f = h5py.File('C:/Users/hoss3301/work/deep_guinea_ears/data/data/data_rawData/in samples/Matlab Data/all.mat', 'r')
for k, v in f.items():
    arrays[k] = np.array(v)

spikes = arrays['all']  # all has a shape of (11,23928,20,265)
# spikes=np.transpose(spikes,(1,2,3,0))
spikes_test = np.zeros(shape=(3, 23928, 20, 265))  # shape of (3,23928,20,265)
spikes_train = np.zeros(shape=(8, 23928, 20, 265))

spikes_test[0:1, :, :, :] = spikes[0:1, :, :, :]
spikes_train[0:5, :, :, :] = spikes[1:6, :, :, :]
spikes_test[1:3, :, :, :] = spikes[6:8, :, :, :]
spikes_train[5:8, :, :, :] = spikes[8:11, :, :, :]

spikes_test = np.transpose(spikes_test, (2, 3, 0, 1))  # shape: 20,265,3,23928

spikes_train = np.transpose(spikes_train, (2, 3, 0, 1))

spikes_test_reshaped = np.zeros(shape=(20 * 265 * 3, 23928))  # shape: 15900,23928
spikes_test_reshaped = np.reshape(spikes_test, (15900, 23928))

spikes_train_reshaped = np.zeros(shape=(20 * 265 * 8, 23928))  # shape: 42400,23928
spikes_train_reshaped = np.reshape(spikes_train, (20 * 265 * 8, 23928))

# spikes_test_cut=np.zeros(shape=(spikes_test_reshaped.shape[0],3,int(spikes_test_reshaped.shape[1]/3)))
# win=int(spikes_test_reshaped.shape[1]/3)
# for i in range(15900):
#    for j in range(3):
#       spikes_test_cut[i,j,:]=spikes_test_reshaped[i,j*win:j*win+win]    #shape: 15900,3,7976
#


# spikes_train_cut=np.zeros(shape=(spikes_train_reshaped.shape[0],3,int(spikes_train_reshaped.shape[1]/3)))
# win=int(spikes_train_reshaped.shape[1]/3)
# for i in range(spikes_train_reshaped.shape[0]):
#    for j in range(3):
#        spikes_train_cut[i,j,:]=spikes_train_reshaped[i,j*win:j*win+win]


# spikes_test=np.zeros(shape=(15900*3,7976))   #shape: 47700,7976
# spikes_test=np.reshape(spikes_test_cut,(15900*3,7976))

# spikes_train=np.zeros(shape=(spikes_train_cut.shape[0]*3,7976))
# spikes_train=np.reshape(input_train_cut,(spikes_train_cut.shape[0]*3,7976))

from gammatone.gtgram import gtgram_strides

nwin, hop_samples, ncols = gtgram_strides(
    24414.4,
    0.025,
    0.01,
    spikes_test_reshaped.shape[1]
)

spikes_test_windowed = np.zeros((spikes_test_reshaped.shape[0], ncols))

# for i in range(15900):
print(i)
for cnum in range(ncols):
    segment = spikes_test_reshaped[:, cnum * hop_samples + np.arange(nwin)]  # shape: 15900,96
    spikes_test_windowed[:, cnum] = segment.mean(1)

spikes_test_windowed = spikes_test_windowed[:, :, np.newaxis]  # 15900,96,1

list_test = [spikes_test_windowed] * 128

spikes_test = np.concatenate(list_test, axis=2)  # 15900,96,128

max_spikes_test = np.max(np.abs(spikes_test))
spikes_test_normalized = spikes_test / max_spikes_test

np.save('data/original/gammatone/clean/max_spikes_test', max_spikes_test)

f = h5py.File('data/original/gammatone/clean/spikes_test_normalized.h5', 'w')
f.create_dataset('spikes_test', data=spikes_test_normalized)
nwin, hop_samples, ncols = gtgram_strides(
    24414.4,
    0.025,
    0.01,
    spikes_train_reshaped.shape[1]
)

spikes_train_windowed = np.zeros((spikes_train_reshaped.shape[0], ncols))

for cnum in range(ncols):
    segment = spikes_train_reshaped[:, cnum * hop_samples + np.arange(nwin)]  # shape: 42400,96
    spikes_train_windowed[:, cnum] = segment.mean(1)

spikes_train_windowed = spikes_train_windowed[:, :, np.newaxis]  # 42400,96,1

list_train = [spikes_train_windowed] * 128

spikes_train = np.concatenate(list_train, axis=2)  # 42400,96,128

max_spikes_train = np.max(np.abs(spikes_train))
spikes_train_normalized = spikes_train / max_spikes_train

np.save('data/original/gammatone/clean/max_spikes_test', max_spikes_test)
f = h5py.File('data/original/gammatone/clean/spikes_train_normalized.h5', 'w')
f.create_dataset('spikes_train', data=spikes_train_normalized)
f.close()

###my data
import scipy.io as sio
import numpy as np

path = 'D:/data/ICC/stim_orig.mat'
mat = sio.loadmat(path)
input_all = mat['original_stim']

input_scream_noisy = input_all[:, 0:12]

input_squeal_noisy = input_all[:, 24:36]
input_test_noisy = input_squeal_noisy

input_tooth_noisy = input_all[:, 48:60]

input_train_noisy = np.concatenate((input_scream_noisy, input_tooth_noisy), axis=1)

input_scream_clean = input_all[:, 72:74]

input_squeal_clean = input_all[:, 74:76]

input_tooth_clean = input_all[:, 76:78]

# train data
# input_train_noisy=input_train_noisy[:,:,np.newaxis,np.newaxis]
input_train_noisy = input_train_noisy[:, :, np.newaxis]  # 47070,24,1
input_train_noisy = input_train_noisy[195312:195312 + 191406, :, :]

# input_train_noisy=input_train_noisy[195312:195312+191406,:,:,:]
# input_train_noisy=input_train_noisy[::2,:,:,:]
input_train_noisy = input_train_noisy[::2, :, :]  # 95703,24,1

list_train = [input_train_noisy] * 256
# train_input_noisy=np.concatenate(list_train,axis=3)
train_input_noisy = np.concatenate(list_train, axis=2)  # 95703,24,256

# list_train=[train_input_noisy]*15
# input_train_noisy=np.concatenate(list_train,axis=2)

input_train_noisy = np.transpose(input_train_noisy, (2, 1, 0))  # 256,24,95703
# input_train_noisy=np.transpose(input_train_noisy,(2,3,1,0))
# input_train_noisy_reshaped=np.zeros(shape=(256*24*15,95703))
# input_train_noisy_reshaped=np.reshape(input_train_noisy,(256*24*15,95703))
input_train_noisy_reshaped = np.zeros(shape=(256 * 24, 95703))
input_train_noisy_reshaped = np.reshape(input_train_noisy, (256 * 24, 95703))

n_channels = 128

gamma_waves_train = np.zeros(shape=(input_train_noisy_reshaped.shape[0], n_channels, 96))

from gammatone.gtgram import gtgram

for i in range(input_train_noisy_reshaped.shape[0]):
    gamma_waves_train[i, :, :] = gtgram(input_train_noisy_reshaped[i, :], 97656.25, 0.025, 0.01, n_channels, 20)
    if i % 10 == 0:
        print(i)

# test data noisy
input_test_noisy = input_test_noisy[:, :, np.newaxis]  # 47070,24,1
input_test_noisy = input_test_noisy[195312:195312 + 191406, :, :]

# input_train_noisy=input_train_noisy[195312:195312+191406,:,:,:]
# input_train_noisy=input_train_noisy[::2,:,:,:]
input_test_noisy = input_test_noisy[::2, :, :]  # 95703,24,1

list_test = [input_test_noisy] * 256
# train_input_noisy=np.concatenate(list_train,axis=3)
input_test_noisy = np.concatenate(list_test, axis=2)  # 95703,24,256

# list_train=[train_input_noisy]*15
# input_train_noisy=np.concatenate(list_train,axis=2)

input_test_noisy = np.transpose(input_test_noisy, (2, 1, 0))  # 256,24,95703
# input_train_noisy=np.transpose(input_train_noisy,(2,3,1,0))
# input_train_noisy_reshaped=np.zeros(shape=(256*24*15,95703))
# input_train_noisy_reshaped=np.reshape(input_train_noisy,(256*24*15,95703))
input_test_noisy_reshaped = np.zeros(shape=(256 * 12, 95703))
input_test_noisy_reshaped = np.reshape(input_test_noisy, (256 * 12, 95703))

n_channels = 128

gamma_waves_test = np.zeros(shape=(input_test_noisy_reshaped.shape[0], n_channels, 96))

from gammatone.gtgram import gtgram

for i in range(input_test_noisy_reshaped.shape[0]):
    gamma_waves_test[i, :, :] = gtgram(input_test_noisy_reshaped[i, :], 97656.25, 0.025, 0.01, n_channels, 20)
    if i % 10 == 0:
        print(i)

gamma_waves_test = np.transpose(gamma_waves_test, (0, 2, 1))  # 3072,96,128
gamma_waves_train = np.transpose(gamma_waves_train, (0, 2, 1))  # 6144,96,128

max_gamma_train = np.max(np.abs(gamma_waves_train))
max_gamma_test = np.max(np.abs(gamma_waves_test))

gamma_train_noisy_normalized = gamma_waves_train / max_gamma_train;
gamma_test_noisy_normalized = gamma_waves_test / max_gamma_test;

np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/max_gamma_test_noisy', max_gamma_test)

# np.save('data/original/gammatone/noisy/gamma_waves_test',gamma_waves_test)

f = h5py.File(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/gamma_waves_test_noisy_normalized.h5', 'w')

f.create_dataset('gamma_waves_test', data=gamma_test_noisy_normalized)
f.close()

np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/max_gamma_train_noisy', max_gamma_train)

# np.save('data/original/gammatone/clean/gamma_waves_train',gamma_waves_train)

f = h5py.File(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/gamma_waves_train_noisy_normalized.h5', 'w')

f.create_dataset('gamma_waves_train', data=gamma_train_noisy_normalized)
f.close()

input_scream_clean_55 = input_scream_clean[195312:195312 + 191406, 0:1]
input_scream_clean_65 = input_scream_clean[195312:195312 + 191406, 1:2]

list_scream_clean_55 = [input_scream_clean_55] * 2
input_scream_clean_55_2 = np.concatenate(list_scream_clean_55, axis=1)

list_scream_clean_65 = [input_scream_clean_65] * 2
input_scream_clean_65_2 = np.concatenate(list_scream_clean_65, axis=1)

input_scream_clean_all = np.concatenate((input_scream_clean_55_2, input_scream_clean_65_2), axis=1)

list_scream = [input_scream_clean_all] * 3
input_scream_clean_all_12 = np.concatenate(list_scream, axis=1)

input_tooth_clean_55 = input_tooth_clean[195312:195312 + 191406, 0:1]
input_tooth_clean_65 = input_tooth_clean[195312:195312 + 191406, 1:2]

list_tooth_clean_55 = [input_tooth_clean_55] * 2
input_tooth_clean_55_2 = np.concatenate(list_tooth_clean_55, axis=1)

list_tooth_clean_65 = [input_tooth_clean_65] * 2
input_tooth_clean_65_2 = np.concatenate(list_tooth_clean_65, axis=1)

input_tooth_clean_all = np.concatenate((input_tooth_clean_55_2, input_tooth_clean_65_2), axis=1)

list_tooth = [input_tooth_clean_all] * 3
input_tooth_clean_all_12 = np.concatenate(list_tooth, axis=1)

input_train_clean = np.concatenate((input_scream_clean_all_12, input_tooth_clean_all_12), axis=1)

input_train_clean = input_train_clean[:, :, np.newaxis]  # 191406,24,1
list_train = [input_train_clean] * 256
input_train_clean_all = np.concatenate(list_train, axis=2)
input_train_clean_all = np.transpose(input_train_clean_all, (2, 1, 0))

input_train_clean_all = input_train_clean_all[::2, :, :]  # 95703,24,256

input_train_clean_reshaped = np.zeros(shape=(256 * 24, 95703))
input_train_clean_reshaped = np.reshape(input_train_clean_all, (256 * 24, 95703))  # 6144,95703

n_channels = 128

gamma_waves_train_clean = np.zeros(shape=(input_train_clean_reshaped.shape[0], n_channels, 96))

from gammatone.gtgram import gtgram

for i in range(input_train_clean_reshaped.shape[0]):
    gamma_waves_train_clean[i, :, :] = gtgram(input_train_clean_reshaped[i, :], 97656.25, 0.025, 0.01, n_channels, 20)
    if i % 10 == 0:
        print(i)

input_squeal_clean_55 = input_squeal_clean[195312:195312 + 191406, 0:1]  # 191406,1
input_squeal_clean_65 = input_squeal_clean[195312:195312 + 191406, 1:2]

list_squeal_clean_55 = [input_squeal_clean_55] * 2
input_squeal_clean_55_2 = np.concatenate(list_squeal_clean_55, axis=1)

list_squeal_clean_65 = [input_squeal_clean_65] * 2
input_squeal_clean_65_2 = np.concatenate(list_squeal_clean_65, axis=1)

input_squeal_clean_all = np.concatenate((input_squeal_clean_55_2, input_squeal_clean_65_2), axis=1)
# 191406,4
list_squeal = [input_squeal_clean_all] * 3
input_test_clean = np.concatenate(list_squeal, axis=1)  # 191406,12

input_test_clean = input_test_clean[::2, :]  # 95703,12

input_test_clean = input_test_clean[:, :, np.newaxis]  # 95703,12,1
list_test = [input_test_clean] * 256
input_test_clean_all = np.concatenate(list_test, axis=2)
input_test_clean_all = np.transpose(input_test_clean_all, (2, 1, 0))  # 256,12,95703

input_test_clean_reshaped = np.zeros(shape=(256 * 12, 95703))
input_test_clean_reshaped = np.reshape(input_test_clean_all, (256 * 12, 95703))  # 6144,95703

n_channels = 128

gamma_waves_test_clean = np.zeros(shape=(input_test_clean_reshaped.shape[0], n_channels, 96))

from gammatone.gtgram import gtgram

for i in range(input_test_clean_reshaped.shape[0]):
    gamma_waves_test_clean[i, :, :] = gtgram(input_test_clean_reshaped[i, :], 97656.25, 0.025, 0.01, n_channels, 20)
    if i % 10 == 0:
        print(i)

gamma_waves_test_clean = np.transpose(gamma_waves_test_clean, (0, 2, 1))  # shape: 15900,96,128
gamma_waves_train_clean = np.transpose(gamma_waves_train_clean, (0, 2, 1))  # 6144,96,128

max_gamma_train = np.max(np.abs(gamma_waves_train_clean))
max_gamma_test = np.max(np.abs(gamma_waves_test_clean))

gamma_train_clean_normalized = gamma_waves_train_clean / max_gamma_train;
gamma_test_clean_normalized = gamma_waves_test_clean / max_gamma_test;

np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/max_gamma_test_clean', max_gamma_test)

# np.save('data/original/gammatone/noisy/gamma_waves_test',gamma_waves_test)

f = h5py.File(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/gamma_waves_test_clean_normalized.h5', 'w')

f.create_dataset('gamma_waves_test', data=gamma_test_clean_normalized)
f.close()

np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/max_gamma_train_clean', max_gamma_train)

# np.save('data/original/gammatone/clean/gamma_waves_train',gamma_waves_train)

f = h5py.File(
    'D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/gamma_waves_train_clean_normalized.h5', 'w')

f.create_dataset('gamma_waves_train', data=gamma_train_clean_normalized)
f.close()

import numpy as np
import scipy.io as spio
import h5py

arrays = {}

f = h5py.File('D:/data/ICC/spikes/RawData/Raw_all_noisy_scream.mat')
for k, v in f.items():
    arrays[k] = np.array(v)

spikes_scream = arrays['Raw_all_noisy_scream']  # size=35160,40,256,12
spikes_scream_mean = np.mean(spikes_scream, 1)  # 35160,256,12

spikes_scream_truncated = spikes_scream_mean[0:23928, :, :]  # 23928,256,12
spikes_scream_truncated = np.transpose(spikes_scream_truncated, (1, 2, 0))  # 256,12,23928

arrays = {}
f = h5py.File('D:/data/ICC/spikes/RawData/Raw_all_noisy_tooth.mat')
for k, v in f.items():
    arrays[k] = np.array(v)

spikes_tooth = arrays['Raw_all_noisy_tooth']  # size=35160,40,256,12
spikes_tooth_mean = np.mean(spikes_tooth, 1)
spikes_tooth_truncated = spikes_tooth_mean[0:23928, :, :]
spikes_tooth_truncated = np.transpose(spikes_tooth_truncated, (1, 2, 0))

spikes_concat = np.concatenate((spikes_scream_truncated, spikes_tooth_truncated), axis=1)

spikes_noisy_train = np.zeros(shape=(256 * 24, 23928))
spikes_noisy_train = np.reshape(spikes_concat, (256 * 24, 23928))

from gammatone.gtgram import gtgram_strides

nwin, hop_samples, ncols = gtgram_strides(
    24414.4,
    0.025,
    0.01,
    spikes_noisy_train.shape[1]
)

spikes_noisy_train_windowed = np.zeros((spikes_noisy_train.shape[0], ncols))
for cnum in range(ncols):
    segment = spikes_noisy_train[:, cnum * hop_samples + np.arange(nwin)]  # shape: 3072,96
    spikes_noisy_train_windowed[:, cnum] = segment.mean(1)

spikes_noisy_train_windowed = spikes_noisy_train_windowed[:, :, np.newaxis]  # 3076,96,1

list_train = [spikes_noisy_train_windowed] * 128

spikes_noisy_train = np.concatenate(list_train, axis=2)  # 3076,96,128

max_spikes_train = np.max(np.abs(spikes_noisy_train))
spikes_train_normalized = spikes_noisy_train / max_spikes_train

np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/max_spikes_train', max_spikes_train)

f = h5py.File('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/spikes_noisy_train_normalized.h5',
              'w')
f.create_dataset('spikes_train', data=spikes_train_normalized)
f.close()

f = h5py.File('D:/data/ICC/spikes/RawData/Raw_all_noisy_squeal.mat')
arrays = {}
for k, v in f.items():
    arrays[k] = np.array(v)

# test
spikes_squeal = arrays['Raw_all_noisy_squeal']  # size=35160,40,256,12
spikes_squeal_mean = np.mean(spikes_squeal, 1)  # 35160,256,12
spikes_squeal_truncated = spikes_squeal_mean[0:23928, :, :]  # 23928,256,12
spikes_squeal_truncated = np.transpose(spikes_squeal_truncated, (1, 2, 0))  # 256,12,23928

spikes_noisy_test = np.zeros(shape=(256 * 12, 23928))
spikes_noisy_test = np.reshape(spikes_squeal_truncated, (256 * 12, 23928))

from gammatone.gtgram import gtgram_strides

nwin, hop_samples, ncols = gtgram_strides(
    24414.4,
    0.025,
    0.01,
    spikes_noisy_test.shape[1]
)

spikes_noisy_test_windowed = np.zeros((spikes_noisy_test.shape[0], ncols))

for cnum in range(ncols):
    segment = spikes_noisy_test[:, cnum * hop_samples + np.arange(nwin)]  # shape: 3072,96
    spikes_noisy_test_windowed[:, cnum] = segment.mean(1)

spikes_noisy_test_windowed = spikes_noisy_test_windowed[:, :, np.newaxis]  # 3076,96,1

list_test = [spikes_noisy_test_windowed] * 128

spikes_noisy_test = np.concatenate(list_test, axis=2)  # 3076,96,128

max_spikes_test = np.max(np.abs(spikes_noisy_test))
spikes_test_normalized = spikes_noisy_test / max_spikes_test

np.save('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/max_spikes_test', max_spikes_test)

f = h5py.File('D:/data_workFolder/TrialsOfNeuralVocalRecon/original/gammatone/noisy/spikes_noisy_test_normalized.h5',
              'w')
f.create_dataset('spikes_test', data=spikes_test_normalized)
f.close()

arrays = {}
f = h5py.File('C:/Users/hoss3301/work/deep_guinea_ears/data/data/data_rawData/in samples/Matlab Data/all.mat', 'r')
for k, v in f.items():
    arrays[k] = np.array(v)

spikes = arrays['all']  # all has a shape of (11,23928,20,265)
# spikes=np.transpose(spikes,(1,2,3,0))
spikes_test = np.zeros(shape=(3, 23928, 20, 265))  # shape of (3,23928,20,265)
spikes_train = np.zeros(shape=(8, 23928, 20, 265))

spikes_test[0:1, :, :, :] = spikes[0:1, :, :, :]
spikes_train[0:5, :, :, :] = spikes[1:6, :, :, :]
spikes_test[1:3, :, :, :] = spikes[6:8, :, :, :]
spikes_train[5:8, :, :, :] = spikes[8:11, :, :, :]

spikes_test = np.transpose(spikes_test, (2, 3, 0, 1))  # shape: 20,265,3,23928

spikes_train = np.transpose(spikes_train, (2, 3, 0, 1))

spikes_test_reshaped = np.zeros(shape=(20 * 265 * 3, 23928))  # shape: 15900,23928
spikes_test_reshaped = np.reshape(spikes_test, (15900, 23928))

spikes_train_reshaped = np.zeros(shape=(20 * 265 * 8, 23928))  # shape: 42400,23928
spikes_train_reshaped = np.reshape(spikes_train, (20 * 265 * 8, 23928))

from gammatone.gtgram import gtgram_strides

nwin, hop_samples, ncols = gtgram_strides(
    24414.4,
    0.025,
    0.01,
    spikes_test_reshaped.shape[1]
)

spikes_test_windowed = np.zeros((spikes_test_reshaped.shape[0], ncols))

# for i in range(15900):
print(i)
for cnum in range(ncols):
    segment = spikes_test_reshaped[:, cnum * hop_samples + np.arange(nwin)]  # shape: 15900,96
    spikes_test_windowed[:, cnum] = segment.mean(1)

spikes_test_windowed = spikes_test_windowed[:, :, np.newaxis]  # 15900,96,1

list_test = [spikes_test_windowed] * 128

spikes_test = np.concatenate(list_test, axis=2)  # 15900,96,128

max_spikes_test = np.max(np.abs(spikes_test))
spikes_test_normalized = spikes_test / max_spikes_test

np.save('data/original/gammatone/clean/max_spikes_test', max_spikes_test)

f = h5py.File('data/original/gammatone/clean/spikes_test_normalized.h5', 'w')
f.create_dataset('spikes_test', data=spikes_test_normalized)
nwin, hop_samples, ncols = gtgram_strides(
    24414.4,
    0.025,
    0.01,
    spikes_train_reshaped.shape[1]
)

spikes_train_windowed = np.zeros((spikes_train_reshaped.shape[0], ncols))

for cnum in range(ncols):
    segment = spikes_train_reshaped[:, cnum * hop_samples + np.arange(nwin)]  # shape: 42400,96
    spikes_train_windowed[:, cnum] = segment.mean(1)

spikes_train_windowed = spikes_train_windowed[:, :, np.newaxis]  # 42400,96,1

list_train = [spikes_train_windowed] * 128

spikes_train = np.concatenate(list_train, axis=2)  # 42400,96,128

max_spikes_train = np.max(np.abs(spikes_train))
spikes_train_normalized = spikes_train / max_spikes_train

np.save('data/original/gammatone/clean/max_spikes_test', max_spikes_test)
f = h5py.File('data/original/gammatone/clean/spikes_train_normalized.h5', 'w')
f.create_dataset('spikes_train', data=spikes_train_normalized)
f.close()
