import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from GenericTools.KerasTools.esoteric_layers.contrastive_loss_language import ContrastiveLossLayer
from TrialsOfNeuralVocalRecon.tools.add_loss_layer import SisdrLossLayer
from TrialsOfNeuralVocalRecon.tools.utils.losses import si_sdr_loss


class SelectOutput(tf.keras.layers.Layer):

    def call(self, inputs, training=None):
        test_output, train_output = inputs

        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        is_train = tf.cast(tf.keras.backend.learning_phase(), tf.float32)
        # print(training, is_train)
        output = is_train * train_output + (1 - is_train) * test_output
        # print(output.shape)
        return output

def MakeMultirresolution(model, data_type):
    inputs = model.inputs
    if len(inputs) == 3:
        input_sound, input_spike, output_sound = inputs
    else:
        input_sound, output_sound = inputs

    sound_names = [l.name for l in model.layers if 'sound' in l.name]
    ns = len(sound_names)
    output = model.outputs[0]

    for i, ln in enumerate(sound_names):
        pool_size = int(300 * (ns - i - 1) / ns + (i + 1) / ns * 2)
        downsampled_clean = AveragePooling1D(pool_size=pool_size, strides=1, padding='same')(output_sound)
        sound = model.get_layer(ln).output
        downsampled_pred = Conv1D(1, 32, padding='causal', kernel_initializer='orthogonal')(sound)
        decoded = SisdrLossLayer(.1)([downsampled_pred, downsampled_clean])
        output = Add()([decoded, output])

    # decoded = SisdrLossLayer()([model.outputs[0], output_sound])

    if 'contrastive' in data_type:
        coef_disorder, coef_random = .1, .1
        decoded = ContrastiveLossLayer(coef_disorder, coef_random, loss=si_sdr_loss)([output_sound, decoded])

    output = SelectOutput()([decoded, output])
    model = Model(inputs, output)
    # model.get_layer(layer_name).output

    return model
