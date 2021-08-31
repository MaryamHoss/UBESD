import numpy as np

import tensorflow.keras.backend as K
from timeit import default_timer as timer
from datetime import timedelta

import tensorflow as tf

tf_log = tf.math.log


def mse_pearson_loss(a, b):
    a = tf.subtract(a, tf.reduce_mean(a))
    b = tf.subtract(b, tf.reduce_mean(b))
    tmp1 = tf.reduce_sum(tf.multiply(a, a))
    tmp2 = tf.reduce_sum(tf.multiply(b, b))
    tmp3 = tf.sqrt(tf.multiply(tmp1, tmp2))
    tmp4 = tf.reduce_sum(tf.multiply(a, b))
    r = -tf.divide(tmp4, tmp3)
    m = tf.reduce_mean(tf.square(tf.subtract(a, b)))
    rm = tf.add(r, m)
    return rm

def log10(x):
    numerator = tf_log(x)
    denominator = tf_log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def si_sdr_loss_original(y_true, y_pred):
    x = tf.squeeze(y_true, axis=-1)
    y = tf.squeeze(y_pred, axis=-1)
    smallVal = 0.000000001  # To avoid divide by zero
    x = x + 0.001
    y = y + 0.001
    a = K.sum(y * x, axis=-1, keepdims=True) / (K.sum(x * x, axis=-1, keepdims=True) + smallVal)

    xa = a * x
    xay = xa - y
    d = K.sum(xa * xa, axis=-1, keepdims=True) / (K.sum(xay * xay, axis=-1, keepdims=True) + smallVal)

    d = -K.mean(10 * log10(d + smallVal))
    return d


def si_sdr_loss_simplified(y_true, y_pred):
    x = tf.squeeze(y_true, axis=-1)
    y = tf.squeeze(y_pred, axis=-1)
    smallVal = 0.000000001  # To avoid divide by zero
    x = x + 0.001
    y = y + 0.001
    a = K.sum(y * x, axis=-1, keepdims=True) / (K.sum(x * x, axis=-1, keepdims=True) + smallVal)

    xa = a * x
    xay = xa - y
    d = K.sum(xa * xa, axis=-1, keepdims=True) / (K.sum(xay * xay, axis=-1, keepdims=True) + smallVal)

    d = -K.mean(10 * log10(d + 1))
    return d


def si_sdr_loss_modified(y_true, y_pred):
    x = tf.squeeze(y_true, axis=-1)
    y = tf.squeeze(y_pred, axis=-1)
    smallVal = 0.000000001  # To avoid divide by zero
    x = x + 0.001
    y = y + 0.001
    a = K.sum(y * x, axis=-1, keepdims=True) / (K.sum(x * x, axis=-1, keepdims=True) + smallVal)

    xa = a * x
    xay = xa - y
    d = K.sum(xa * xa, axis=-1, keepdims=True) / (K.sum(xay * xay, axis=-1, keepdims=True) + smallVal)
    # d1=tf.zeros(d.shape)
    d1 = d == 0
    d1 = 1 - tf.cast(d1, tf.float32)

    d = -K.mean(10 * d1 * log10(d + smallVal))
    return d


# DATADIR='./data'
def test():
    batch_size = 32
    time_steps = 20000 #30000  # 2100
    features = 1
    loss_matrix = np.zeros((3, 768))  # the matrix at the end, having the three losses in it
    np_true = np.random.rand(batch_size, time_steps, features).astype(np.float32)
    np_pred = np.random.rand(batch_size, time_steps, features).astype(np.float32)

    losses = {'si_sdr_loss_modified': si_sdr_loss_modified,
              'si_sdr_loss_original': si_sdr_loss_original,
              'si_sdr_loss_simplified': si_sdr_loss_simplified,
              'mesgarani': mse_pearson_loss,
              'MSE': tf.keras.losses.MSE}

    for _ in range(4):
        print()
        for name_loss, loss in losses.items():
            start = timer()
            l = loss(np_true, np_pred).numpy()
            end = timer()
            print('{} took {}'.format(name_loss, timedelta(seconds=end - start)))
            #print('     value {}'.format(l))


if __name__ == '__main__':
    test()
