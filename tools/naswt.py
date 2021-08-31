import tensorflow as tf
import numpy as np

EMPIRICAL_MAX = -2.5e2
EMPIRICAL_MIN = -6.5e6


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v = np.linalg.eigvals(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1. / (v + k))


def IWPJS(model, batch, return_dict=None):
    # inv_weighted_parameters_jacobian_score
    x, y = batch
    # define the loss inside here
    with tf.GradientTape(persistent=True) as tape:
        outputs = model(x)
        l = model.loss(y, outputs)
    variables = model.trainable_variables
    parameters_grads = tape.gradient(l, variables)  # (outputs, variables)
    len_p = len(parameters_grads)

    s = 0
    j = 0
    for i, g in enumerate(reversed(parameters_grads)):
        if not g is None:
            if len(g.shape) > 1:
                batch_size = g.shape[0]
                if batch_size > 1:
                    j += 1
                    try:
                        g = g.numpy().reshape(batch_size, -1)
                        s += j * eval_score(g, None)
                    except:
                        pass

    if s == 0:
        s = np.nan
    else:
        normalizer_weights = len_p * (len_p + 1) / 2
        s /= j * normalizer_weights

        s = 1 - (s-EMPIRICAL_MIN) / (EMPIRICAL_MAX-EMPIRICAL_MIN)

    if not return_dict is None:
        return_dict['score'] = s.real
    return s.real
