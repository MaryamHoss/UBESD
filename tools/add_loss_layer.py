import tensorflow as tf

from TrialsOfNeuralVocalRecon.tools.utils.losses import si_sdr_loss


class SisdrLossLayer(tf.keras.layers.Layer):

    def __init__(self, coef=1., **kwargs):
        super().__init__(**kwargs)
        self.coef = coef

    def call(self, inputs, training=None):
        pred_output_sound, true_output_sound = inputs

        loss = self.coef * si_sdr_loss(true_output_sound, pred_output_sound)
        self.add_loss(loss)
        self.add_metric(loss, name='aux_si_sdr_loss_{}'.format(self.name), aggregation='mean')
        mse = tf.keras.losses.MSE(true_output_sound, pred_output_sound)
        self.add_metric(mse, name='aux_mse_{}'.format(self.name), aggregation='mean')
        return pred_output_sound


    def get_config(self):
        config = {
            'coef': self.coef,
        }

        return dict(list(super().get_config().items()) + list(config.items()))
