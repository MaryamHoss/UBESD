import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def corr2(a, b):
    k = np.shape(a)
    H = k[0]
    W = k[1]
    c = np.zeros((H, W))
    d = np.zeros((H, W))
    e = np.zeros((H, W))

    # Calculating mean values
    AM = np.mean(a)
    BM = np.mean(b)

    # Calculating terms of the formula
    for ii in range(H):
        for jj in range(W):
            c[ii, jj] = (a[ii, jj] - AM) * (b[ii, jj] - BM)
            d[ii, jj] = (a[ii, jj] - AM) ** 2
            e[ii, jj] = (b[ii, jj] - BM) ** 2

    # Formula itself
    r = np.sum(c) / float(np.sqrt(np.sum(d) * np.sum(e)))
    return r


def corr2_mse_loss(a, b):
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


def perplexity_raw(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267
    """
    #     cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.cast(K.equal(K.max(y_true, axis=-1),
                                   K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                           K.floatx())
    perplexity = K.exp(cross_entropy)
    return perplexity


def perplexity(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = K.exp(cross_entropy)
    return perplexity
