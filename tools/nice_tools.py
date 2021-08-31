# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:44:48 2020

@author: hoss3301
"""

import numpy as np
from scipy.interpolate import interp1d


def rms_normalize(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    audio= audio / rms
    return audio


def removeSilentFrames(x, y, dyn_range=40, N=256, K=256 / 2):
    #   [X_SIL Y_SIL] = REMOVESILENTFRAMES(X, Y, RANGE, N, K) X and Y
    #   are segmented with frame-length N and overlap K, where the maximum energy
    #   of all frames of X is determined, say X_MAX. X_SIL and Y_SIL are the
    #  reconstructed signals, excluding the frames, where the energy of a frame
    #   of X is smaller than X_MAX-RANGE

    x = x[:]
    y = y[:]
    # x=np.squeeze(x)
    # y=np.squeeze(y)
    x = np.reshape(x, (x.shape[0] * x.shape[1], 1))
    y = np.reshape(y, (y.shape[0] * y.shape[1], 1))
    frames = np.arange(0, x.shape[0] - N, K)
    w = np.hanning(N)
    msk = np.zeros(shape=frames.size)

    for j in range(frames.shape[0]):
        jj = np.arange(frames[j], frames[j] + N).astype('int')
        msk[j] = 20 * np.log10(np.linalg.norm(x[jj].T * w) / np.sqrt(N))

    msk = (msk - np.max(msk) + dyn_range) > 0;
    count = 0

    x_sil = np.zeros(shape=(x.shape))
    y_sil = np.zeros(shape=(y.shape))
    w = w[:, np.newaxis]
    for j in range(frames.shape[0]):
        if msk[j]:
            # Sprint(j)
            jj_i = np.arange(frames[j], frames[j] + N).astype('int')
            jj_o = np.arange(frames[count], frames[count] + N).astype('int')
            x_sil[jj_o] = x_sil[jj_o] + np.multiply(x[jj_i], w)
            y_sil[jj_o] = y_sil[jj_o] + np.multiply(y[jj_i], w)
            count = count + 1

    x_sil = x_sil[0:jj_o[-1], :]
    y_sil = y_sil[0:jj_o[-1], :]

    return x_sil[np.newaxis], y_sil[np.newaxis]


def resample_signal(x, factor, kind='linear'):  # for this to work x has to be one-dimensional
    size_new = np.ceil(x.size / factor)
    f = interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, int(size_new)))
