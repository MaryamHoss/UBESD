import tensorflow as tf

from GenericTools.KerasTools.esoteric_layers.contrastive_loss_language import ContrastiveLossLayer
from GenericTools.KerasTools.noise_curriculum import InputSensitiveGaussianNoise
from TrialsOfNeuralVocalRecon.neural_models.models_transformer_classic import TransformerClassic
from TrialsOfNeuralVocalRecon.neural_models.models_transformer_crossed_stairs import TransformerCrossedStairs
from TrialsOfNeuralVocalRecon.neural_models.models_transformer_parallel import TransformerParallel
from TrialsOfNeuralVocalRecon.neural_models.models_transformer_stairs import TransformerStairs
from TrialsOfNeuralVocalRecon.tools.utils.losses import si_sdr_loss

transformers = {
    'transformer_classic': TransformerClassic,
    'transformer_parallel': TransformerParallel,
    'transformer_stairs': TransformerStairs,
    'transformer_crossed_stairs': TransformerCrossedStairs
}

def build_transformer(
        data_type='WithSpikes',
        learning_rate=0.001,
        sound_shape=(31900, 1),
        spike_shape=(7975, 1),
        downsample_sound_by=3,
        num_layers=4,  # 12
        d_model=64,  # 128,  # 512
        dff=128,  # 2048
        num_heads=4,  # 8
        dropout_rate=0.1,
        *args
):
    initializer = 'orthogonal' if 'orthogonal' in data_type else 'glorot_normal'

    t_name = [k for k in transformers.keys() if k in data_type][0]
    t = transformers[t_name]

    transformer = t(num_layers, d_model, num_heads, dff,
                    pe_input=sound_shape[0], pe_target=sound_shape[0],
                    rate=dropout_rate, kernel_initializer=initializer)


    input_sound = tf.keras.layers.Input(shape=(None, sound_shape[1]))
    input_spike = tf.keras.layers.Input(shape=(None, spike_shape[1]))
    output_sound = tf.keras.layers.Input(shape=(None, sound_shape[1]))

    sound = input_sound
    spikes = input_spike
    if 'noiseinput' in data_type:
        isgn = InputSensitiveGaussianNoise(0.)
        sound = isgn(sound)
        spikes = isgn(spikes)
        curriculum_noise = isgn.stddev

    sound = tf.keras.layers.Dense(d_model, kernel_initializer=initializer)(sound)
    spikes = tf.keras.layers.Dense(d_model, kernel_initializer=initializer)(spikes)
    predictions, _ = transformer([spikes, sound])
    predictions = tf.keras.layers.Lambda(lambda x: x, name='network_prediction')(predictions)

    # decoded = SisdrLossLayer()([predictions, output_sound])

    if 'contrastive' in data_type:
        coef_disorder, coef_random = .1, .1
        decoded = ContrastiveLossLayer(coef_disorder, coef_random, loss=si_sdr_loss)([output_sound, decoded])

    # define autoencoder
    inputs = [input_sound, input_spike, output_sound] if 'WithSpikes' in data_type \
        else [input_sound, output_sound]
    autoencoder = tf.keras.models.Model(inputs=inputs, outputs=decoded)

    if 'noiseinput' in data_type:
        autoencoder.curriculum_noise = curriculum_noise
    return autoencoder
