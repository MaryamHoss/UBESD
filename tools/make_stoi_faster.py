import numpy as np

import tensorflow.keras.backend as K
from timeit import default_timer as timer
from datetime import timedelta
import time

import tensorflow as tf
from TrialsOfNeuralVocalRecon.tools.utils.OBM import OBM as oooooo
import functools

window_fn = functools.partial(tf.signal.hann_window, periodic=True)
tf_signal = tf.signal
tf_log = tf.math.log


def thirdoct(fs, nfft, num_bands, min_freq):
    """ Returns the 1/3 octave band matrix and its center frequencies
    # Arguments :
        fs : sampling rate
        nfft : FFT size
        num_bands : number of 1/3 octave bands
        min_freq : center frequency of the lowest 1/3 octave band
    # Returns :
        obm : Octave Band Matrix
        cf : center frequencies
    """
    f = np.linspace(0, fs, nfft + 1)
    f = f[:int(nfft / 2 + 1)]
    k = np.array(range(num_bands)).astype(float)
    cf = np.power(2. ** (1. / 3), k) * min_freq
    freq_low = min_freq * np.power(2., (2 * k - 1) / 6)
    freq_high = min_freq * np.power(2., (2 * k + 1) / 6)
    obm = np.zeros((num_bands, len(f)))  # a verifier

    for i in range(len(cf)):
        # Match 1/3 oct band freq with fft frequency bin
        f_bin = np.argmin(np.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = np.argmin(np.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        # Assign to the octave band matrix
        obm[i, fl_ii:fh_ii] = 1
    return obm.astype(np.float32), cf


def log10(x):
    numerator = tf_log(x)
    denominator = tf_log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def stoi_loss_original(batch_size=8, nbf=200):
    def stoi_loss_inner(y_true, y_pred):
        y_true = K.squeeze(y_true, axis=-1)
        y_pred = K.squeeze(y_pred, axis=-1)
        y_pred_shape = K.shape(y_pred)

        stft_true = tf_signal.stft(y_true, 256, 128, 512, window_fn, pad_end=False)
        stft_pred = tf_signal.stft(y_pred, 256, 128, 512, window_fn, pad_end=False)

        N = 230  # 30  # length of temporal envelope vectors
        J = 15  # Number of one-third octave bands (cannot be varied)
        M = int(nbf - (N - 1))  # number of temporal envelope vectors
        smallVal = 0.0000000001  # To avoid divide by zero

        fs = 97656.25
        nfft = 512  # 256
        min_freq = 1150  # 1050
        OBM, _ = thirdoct(fs, nfft, J, min_freq)

        OBM1 = tf.convert_to_tensor(oooooo)
        OBM1 = K.tile(OBM1, [y_pred_shape[0], 1, ])
        OBM1 = K.reshape(OBM1, [y_pred_shape[0], 15, 257, ])

        OCT_pred = K.sqrt(tf.matmul(OBM1, K.square(K.abs(tf.transpose(stft_pred, perm=[0, 2, 1])))))
        OCT_true = K.sqrt(tf.matmul(OBM1, K.square(K.abs(tf.transpose(stft_true, perm=[0, 2, 1])))))

        doNorm = True

        c = K.constant(5.62341325, 'float')  # 10^(-Beta/20) with Beta = -15
        d = K.variable(0.0, 'float')
        for i in range(0, batch_size):  # Run over mini-batches
            for m in range(0, M):  # Run over temporal envelope vectors
                x = K.squeeze(tf.slice(OCT_true, [i, 0, m], [1, J, N]), axis=0)
                y = K.squeeze(tf.slice(OCT_pred, [i, 0, m], [1, J, N]), axis=0)
                if doNorm:
                    alpha = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True) / (
                        K.sum(K.square(y), axis=-1, keepdims=True)) + smallVal)
                    alpha = K.tile(alpha, [1, N, ])
                    ay = y * alpha
                    y = K.minimum(ay, x + x * c)
                xn = x - K.mean(x, axis=-1, keepdims=True)
                xn = xn / (K.sqrt(K.sum(xn * xn, axis=-1, keepdims=True)) + smallVal)
                yn = y - K.mean(y, axis=-1, keepdims=True)
                yn = yn / (K.sqrt(K.sum(yn * yn, axis=-1, keepdims=True)) + smallVal)
                di = K.sum(xn * yn, axis=-1, keepdims=True)
                d = d + K.sum(di, axis=0, keepdims=False)
        return 1 - (d / K.cast(batch_size * J * M, dtype='float'))

    return stoi_loss_inner



def stoi_loss_modified(batch_size=8, nbf=200):
    N = 230  # 30  # length of temporal envelope vectors
    J = 15  # Number of one-third octave bands (cannot be varied)
    M = int(nbf - (N - 1))  # number of temporal envelope vectors
    smallVal = 0.0000000001  # To avoid divide by zero

    fs = 97656.25
    nfft = 512  # 256
    min_freq = 1150  # 1050

    def stoi_loss_inner(y_true, y_pred):
        y_true = K.squeeze(y_true, axis=-1)
        y_pred = K.squeeze(y_pred, axis=-1)
        y_pred_shape = K.shape(y_pred)

        stft_true = tf_signal.stft(y_true, 256, 128, 512, window_fn, pad_end=False)
        stft_pred = tf_signal.stft(y_pred, 256, 128, 512, window_fn, pad_end=False)


        OBM, _ = thirdoct(fs, nfft, J, min_freq)

        OCT_pred = K.sqrt(tf.matmul(OBM, K.square(K.abs(tf.transpose(stft_pred, perm=[0, 2, 1])))))
        OCT_true = K.sqrt(tf.matmul(OBM, K.square(K.abs(tf.transpose(stft_true, perm=[0, 2, 1])))))

        doNorm = True

        c = K.constant(5.62341325, 'float')  # 10^(-Beta/20) with Beta = -15
        d = K.variable(0.0, 'float')
        for i in range(0, batch_size):  # Run over mini-batches
            for m in range(0, M):  # Run over temporal envelope vectors
                x = K.squeeze(tf.slice(OCT_true, [i, 0, m], [1, J, N]), axis=0)
                y = K.squeeze(tf.slice(OCT_pred, [i, 0, m], [1, J, N]), axis=0)
                if doNorm:
                    alpha = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True) / (
                        K.sum(K.square(y), axis=-1, keepdims=True)) + smallVal)
                    alpha = K.tile(alpha, [1, N, ])
                    ay = y * alpha
                    y = K.minimum(ay, x + x * c)
                xn = x - K.mean(x, axis=-1, keepdims=True)
                xn = xn / (K.sqrt(K.sum(xn * xn, axis=-1, keepdims=True)) + smallVal)
                yn = y - K.mean(y, axis=-1, keepdims=True)
                yn = yn / (K.sqrt(K.sum(yn * yn, axis=-1, keepdims=True)) + smallVal)
                di = K.sum(xn * yn, axis=-1, keepdims=True)
                d = d + K.sum(di, axis=0, keepdims=False)
        return 1 - (d / K.cast(batch_size * J * M, dtype='float'))

    return stoi_loss_inner


# DATADIR='./data'
def test():
    batch_size = 32
    time_steps = 30000  # 2100
    features = 1
    loss_matrix = np.zeros((3, 768))  # the matrix at the end, having the three losses in it
    np_true = np.random.rand(batch_size, time_steps, features).astype(np.float32)
    np_pred = np.random.rand(batch_size, time_steps, features).astype(np.float32)

    losses = {'MSE': tf.keras.losses.MSE,
              'stoi_loss_original': stoi_loss_original(batch_size=batch_size, nbf=200),
              'stoi_loss_modified': stoi_loss_modified(batch_size=batch_size, nbf=200),
              }

    for _ in range(4):
        print()
        for name_loss, loss in losses.items():
            start = timer()
            l = loss(np_true, np_pred)
            end = timer()
            print('{} took {}'.format(name_loss, timedelta(seconds=end - start)))
            # print('     value {}'.format(l))


if __name__ == '__main__':
    test()
