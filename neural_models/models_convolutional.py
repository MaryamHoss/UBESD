
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from UBESD.neural_models.fusions import FiLM_Fusion, Fusion
from UBESD.tools.utils.losses import si_sdr_loss

def select_from_string(string, id='convblock:', output_type='string'):
    block_seq = [s for s in string.split('_') if id in s][0].replace(id, '')

    if output_type == 'int':
        if ':' in block_seq:
            block_seq = [int(o) for o in block_seq.split(':')]
        else:
            block_seq = int(block_seq)

    return block_seq

esoteric_initializers_list = [
    'BiGamma', 'BiGamma10', 'BiGammaOrthogonal', 'CauchyOrthogonal', 'GlorotCauchyOrthogonal', 'GlorotOrthogonal',
    'GlorotTanh', 'MoreVarianceScalingAndOrthogonal', 'TanhBiGamma10', 'TanhBiGamma10Orthogonal', 'orthogonal',
]


def ConvBlock(n_filters, kernel_size, activation, kernel_initializer, data_type, name='', i=1):
    # sound = Conv1D(n_filters, kernel_size, strides=1, activation=activation, padding='causal',
    #                kernel_initializer=kernel_initializer)(sound)
    # sound = Dropout(0.3)(LayerNormalization(axis=[2])(LeakyReLU()(sound)))  # second version

    # sound = Dropout(0.3)(LeakyReLU()(LayerNormalization(axis=[2])(sound)))  # first version
    # sound = LayerNormalization(axis=[2])(Dropout(0.3)(sound)) #third

    if 'convblock:' in data_type:
        block_seq = select_from_string(data_type, id='convblock:')
    else:
        block_seq = 'crnd'

    padding = 'causal' if not 'noncausal' in data_type else 'same'

    if 'separable' in data_type and 'dilation' in data_type:
        dilation_rate = 2 ** i
        conv = SeparableConv1D(n_filters, kernel_size, activation=activation, padding=padding,
                               depthwise_initializer=kernel_initializer, pointwise_initializer=kernel_initializer,
                               dilation_rate=dilation_rate)
    elif 'dilation:' in data_type:
        # dilation_rate = select_from_string(data_type, id='dilation:', output_type='int')
        dilation_rate = 2 ** i
        conv = Conv1D(n_filters, kernel_size, activation=activation, padding=padding, dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer)

    else:
        conv = Conv1D(n_filters, kernel_size, activation=activation, padding=padding,
                      kernel_initializer=kernel_initializer)
    lr = LeakyReLU()
    ln = LayerNormalization(axis=[2])
    do = Dropout(0.3)
    identity = Lambda(lambda x: x, name=name)

    dict_layers = {'c': conv, 'r': lr, 'n': ln, 'd': do}

    def call(inputs):
        output = inputs
        for s in block_seq:
            output = dict_layers[s](output)
        # output = identity(do(ln(lr(conv(inputs)))))
        return identity(output)

    return call


def build_conv_with_fusion(
        data_type='WithSpikes',
        learning_rate=0.001,
        sound_shape=(31900, 1),
        spike_shape=(7975, 1),
        downsample_sound_by=3,
        activation_encode='relu',  # snake, # 'relu', #
        activation_spikes='relu',  # snake, # 'relu', #
        activation_decode='relu',  # snake, #'relu', #
        activation_all='tanh',  # snake, #'tanh', #
        n_convolutions=3,
        min_filters=5,
        max_filters=100,
        kernel_size=25, #25
        weight_decay=.1,
        clipnorm=1.
):
    # first version:
    # activation_encode = 'relu'
    # activation_spikes = 'relu'
    # activation_decode = 'relu'
    # activation_all = 'tanh'
    # n_convolutions = 3
    # min_filters = 5
    # max_filters = 100
    # kernel_size = 25

    # activation_encode = 'linear' #second versions
    # activation_spikes = 'linear'
    # activation_decode = 'linear'
    # activation_all = 'linear'
    # activation_encode = 'relu'
    # activation_spikes = 'relu'
    # activation_decode = 'relu'
    # activation_all = 'linear'

    ks = kernel_size
    initializer = [d.replace('initializer:', '') for d in data_type.split('_')
                   if 'initializer:' in d][0] if 'initializer' in data_type else 'glorot_uniform'

    if initializer in esoteric_initializers_list:
        initializer = getattr(esoteric_initializers, initializer)()
        # initializer = getattr(thismodule, initializer)

    print(initializer)
    padding = 'causal' if not 'noncausal' in data_type else 'same'

    if 'mmfilter:' in data_type:
        min_filters, max_filters = select_from_string(data_type, id='mmfilter:', output_type='int')

    if 'nconvs:' in data_type:
        n_convolutions = select_from_string(data_type, id='nconvs:', output_type='int')

    filters = np.linspace(min_filters, max_filters, n_convolutions).astype(int)
    c = filters[::-1]
    c_end = c[-1]
    c = c[0:c.shape[0] - 1]
    decay_rate = learning_rate / 150

    # downsampled_sound_shape = (int(sound_shape[0] / downsample_sound_by), sound_shape[1])
    input_sound = Input(shape=(None, sound_shape[1]))
    input_spike = Input(shape=(None, spike_shape[1]))
    output_sound = Input(shape=(None, sound_shape[1]))

    sound = input_sound
    spike = input_spike
    if 'noiseinput' in data_type:
        isgn = InputSensitiveGaussianNoise(0.)
        sound = isgn(sound)
        spike = isgn(spike)
        curriculum_noise = isgn.stddev

    unet_layers = []
    for i, n_filters in enumerate(reversed(filters[1:])):
    #for i, n_filters in enumerate((filters)):

        in_sound, in_spike = sound, spike
        sound = ConvBlock(n_filters, ks, activation_encode, initializer,
                          name='encoder_sound_{}_{}'.format(n_filters, i), data_type=data_type, i=i)(in_sound)
        spike = ConvBlock(n_filters, ks, activation_encode, initializer,
                          name='encoder_spike_{}_{}'.format(n_filters, i), data_type=data_type, i=i)(in_spike)

        if 'WithSpikes' in data_type:
            sound, spike = FiLM_Fusion(sound.shape[2], data_type, initializer, n_filters)([sound, spike])

        if 'resnet2' in data_type:
            sound = ConvBlock(n_filters, ks, activation_encode, initializer,
                              name='encoder_sound_{}_{}_2'.format(n_filters, i), data_type=data_type, i=i)(sound)
            spike = ConvBlock(n_filters, ks, activation_encode, initializer,
                              name='encoder_spike_{}_{}_2'.format(n_filters, i), data_type=data_type, i=i)(spike)
            if 'WithSpikes' in data_type:
                sound, spike = FiLM_Fusion(sound.shape[2], data_type, initializer, n_filters)([sound, spike])

        if 'soundencoderresnet' in data_type:
            # sound = ConvBlock()(sound)
            sound = Add()([in_sound, sound])
            sound = Activation('relu')(sound)

        if 'spikeencoderresnet' in data_type:
            in_spike = Lambda(lambda x: tf.expand_dims(x, -1))(in_spike)
            in_spike = Permute((3, 2, 1))(in_spike)
            in_spike = Resizing(1, n_filters)(in_spike)
            in_spike = Permute((3, 2, 1))(in_spike)
            in_spike = Lambda(lambda x: tf.squeeze(x, -1))(in_spike)

            spike = Add()([in_spike, spike])
            spike = Activation('relu')(spike)

        if not 'OnlySpikes' in data_type:
            unet_layers.append(sound)
        else:
            unet_layers.append(spike)

    sound = ConvBlock(c_end, ks, activation_all, initializer,
                      name='encoder_sound_{}_{}'.format(c_end, i + 1), data_type=data_type, i=i)(sound)
    spike = ConvBlock(c_end, ks, activation_all, initializer,
                      name='encoder_spike_{}_{}'.format(c_end, i + 1), data_type=data_type, i=i)(spike)

    sound = Fusion(data_type)([sound, spike])

    #for j, n_filters in enumerate(reversed(filters)):
    for j, n_filters in enumerate((filters[1:])):

        in_sound = sound

        if 'unet' in data_type:
            prev = unet_layers[-j-1]

            if not 'sumunet' in data_type:
                sound = Concatenate()([prev, sound])
            else:
                sound = Add()([prev, sound])

        sound = ConvBlock(n_filters, ks, activation_decode, initializer,
                          name='decoder_sound_{}_{}'.format(n_filters, j), data_type=data_type, i=j)(sound)

        if 'sounddecoderresnet' in data_type:
            tg_d = sound.shape[2]
            in_sound = Lambda(lambda x: tf.expand_dims(x, -1))(in_sound)
            in_sound = Permute((3, 2, 1))(in_sound)
            in_sound = Resizing(1, tg_d)(in_sound)
            in_sound = Permute((3, 2, 1))(in_sound)
            in_sound = Lambda(lambda x: tf.squeeze(x, -1))(in_sound)

            if 'resnet2' in data_type:
                sound = ConvBlock(n_filters, ks, activation_encode, initializer,
                                  name='decoder_sound_{}_{}_2'.format(n_filters, j), data_type=data_type, i=j)(sound)

            sound = Add()([in_sound, sound])
            sound = Activation('relu')(sound)

    sound = Conv1D(1, ks, activation=activation_all, padding=padding,
                   kernel_initializer=initializer)(sound)
    decoded = Activation('tanh', name='network_prediction')(sound)  # snake)(decoded) #'tanh')(decoded)


    # define autoencoder
    if 'noSpikes' in data_type:
        inputs = [input_sound, output_sound]
    elif 'WithSpikes' in data_type:
        inputs = [input_sound, input_spike, output_sound]
    elif 'OnlySpikes' in data_type:
        inputs = [input_spike, output_sound]
    else:
        raise NotImplementedError

    autoencoder = Model(inputs=inputs, outputs=decoded)


    return autoencoder




def build_conv_with_fusion_skip(data_type='WithSpikes',
                                learning_rate=0.001,
                                sound_shape=(31900, 1),
                                spike_shape=(7975, 1),
                                downsample_sound_by=3,
                                activation_encode='relu',  # snake, # 'relu', #
                                activation_spikes='relu',  # snake, # 'relu', #
                                activation_decode='relu',  # snake, #'relu', #
                                activation_all='tanh',  # snake, #'tanh', #
                                n_convolutions=3,
                                min_filters=5,
                                max_filters=100,
                                kernel_size=25,#25
                                weight_decay=.1,
                                clipnorm=1.
                                ):
    ks = kernel_size

    initializer = [d.replace('initializer:', '') for d in data_type.split('_')
                   if 'initializer:' in d][0] if 'initializer' in data_type else 'glorot_uniform'

    if initializer in esoteric_initializers_list:
        initializer = getattr(esoteric_initializers, initializer)()

    print(initializer)
    # initializer = 'orthogonal' if 'orthogonal' in data_type else 'glorot_uniform'
    padding = 'causal' if not 'noncausal' in data_type else 'same'

    if 'mmfilter:' in data_type:
        min_filters, max_filters = select_from_string(data_type, id='mmfilter:', output_type='int')
    if 'nconvs:' in data_type:
        n_convolutions = select_from_string(data_type, id='nconvs:', output_type='int')

    filters = np.linspace(min_filters, max_filters, n_convolutions).astype(int)
    c = filters[::-1]
    c_end = c[-1]
    c = c[0:c.shape[0] - 1]
    decay_rate = learning_rate / 150

    # downsampled_sound_shape = (int(sound_shape[0] / downsample_sound_by), sound_shape[1])
    input_sound = Input(shape=(None, sound_shape[1]))
    input_spike = Input(shape=(None, spike_shape[1]))
    output_sound = Input(shape=(None, sound_shape[1]))

    sound = input_sound
    spike = input_spike

    sound1 = ConvBlock(filters[0], ks, activation_encode, initializer,
                       name='encoder_sound_block_1_{}'.format(filters[0]), data_type=data_type, i=0)(sound)

    spike1 = ConvBlock(filters[0], ks, activation_encode, initializer,
                       name='encoder_spikes_block_1_{}'.format(filters[0]), data_type=data_type, i=0)(spike)

    if 'WithSpikes' in data_type:
        sound1, spike1 = FiLM_Fusion(sound1.shape[2], data_type, initializer)([sound1, spike1])

    sound2 = ConvBlock(filters[1], ks, activation_encode, initializer,
                       name='encoder_sound_block_2_{}'.format(filters[1]), data_type=data_type, i=1)(sound1)

    spike2 = ConvBlock(filters[1], ks, activation_encode, initializer,
                       name='encoder_spikes_block_2_{}'.format(filters[1]), data_type=data_type, i=1)(spike1)

    if 'WithSpikes' in data_type:
        sound2, spike2 = FiLM_Fusion(sound2.shape[2], data_type, initializer)([sound2, spike2])

    if 'resnet' in data_type:
        sound2 = Add()([sound2, sound])

        sound2 = Activation('relu')(sound2)
        
    if 'resnet2' in data_type:
        
        spike = Lambda(lambda x: tf.expand_dims(x, -1))(spike)
        spike = Permute((3, 2, 1))(spike)
        spike = Resizing(1, filters[0])(spike)
        spike = Permute((3, 2, 1))(spike)
        spike = Lambda(lambda x: tf.squeeze(x, -1))(spike)

        
        spike2 = Add()([spike2, spike])

        spike2 = Activation('relu')(spike2)

    sound3 = ConvBlock(filters[2], ks, activation_encode, initializer,
                       name='encoder_sound_block_3_{}'.format(filters[2]), data_type=data_type, i=2)(sound2)

    spike3 = ConvBlock(filters[2], ks, activation_encode, initializer,
                       name='encoder_spikes_block_3_{}'.format(filters[2]), data_type=data_type, i=2)(spike2)

    if 'WithSpikes' in data_type:
        sound3, spike3 = FiLM_Fusion(sound3.shape[2], data_type, initializer)([sound3, spike3])

    sound4 = ConvBlock(filters[3], ks, activation_encode, initializer,
                       name='encoder_sound_block_4_{}'.format(filters[3]), data_type=data_type, i=3)(sound3)

    spike4 = ConvBlock(filters[3], ks, activation_encode, initializer,
                       name='encoder_spikes_block_4_{}'.format(filters[3]), data_type=data_type, i=3)(spike3)

    if 'WithSpikes' in data_type:
        sound4, spike4 = FiLM_Fusion(sound4.shape[2], data_type, initializer)([sound4, spike4])

    if 'resnet' in data_type:
        sound4 = Add()([sound2, sound4])

        sound4 = Activation('relu')(sound4)

    if 'resnet2' in data_type:
        
        spike4 = Add()([spike2, spike4])

        spike4 = Activation('relu')(spike4)
        
        

    decoded = Fusion(data_type)([sound4, spike4])
    
    if 'separable' in data_type:
        decoded1 = SeparableConv1D(filters[2], kernel_size, strides=1, activation=activation_decode, padding=padding,
                     depthwise_initializer=initializer, pointwise_initializer=initializer, name='decoder1')(decoded)
    else:
        decoded1 = Conv1D(filters[2], kernel_size, strides=1, activation=activation_decode, padding=padding,
                      kernel_initializer=initializer, name='decoder1')(decoded)
    if 'unet' in data_type:
        decoded1 = Concatenate(axis=-1)([decoded1, sound4])
    decoded1 = Dropout(0.3)(LeakyReLU()(LayerNormalization(axis=[2])(decoded1)))  # first version
    # decoded = Dropout(0.3)(LayerNormalization(axis=[2])(LeakyReLU()(decoded)))  # second version
    # decoded = LayerNormalization(axis=[2])(Dropout(0.3)(decoded)) #third


    if 'separable' in data_type:
        decoded2 = SeparableConv1D(filters[2], kernel_size, strides=1, activation=activation_decode, padding=padding,
                     depthwise_initializer=initializer, pointwise_initializer=initializer, name='decoder2')(decoded1)
        
    else:
        decoded2 = Conv1D(filters[2], kernel_size, strides=1, activation=activation_decode, padding=padding,
                      kernel_initializer=initializer, name='decoder2')(decoded1)
        
        
    if 'unet' in data_type:
        decoded2 = Concatenate(axis=-1)([decoded2, sound3])
    decoded2 = Dropout(0.3)(LeakyReLU()(LayerNormalization(axis=[2])(decoded2)))  # first version
    # decoded = Dropout(0.3)(LayerNormalization(axis=[2])(LeakyReLU()(decoded)))  # second version
    # decoded = LayerNormalization(axis=[2])(Dropout(0.3)(decoded)) #third

    if 'separable' in data_type:
        decoded3 = SeparableConv1D(filters[2], kernel_size, strides=1, activation=activation_decode, padding=padding,
                     depthwise_initializer=initializer, pointwise_initializer=initializer, name='decoder3')(decoded2)
    else:
        decoded3 = Conv1D(filters[2], kernel_size, strides=1, activation=activation_decode, padding=padding,
                      kernel_initializer=initializer, name='decoder3')(decoded2)

    if 'unet' in data_type:
        decoded3 = Concatenate(axis=-1)([decoded3, sound2])
    decoded3 = Dropout(0.3)(LeakyReLU()(LayerNormalization(axis=[2])(decoded3)))  # first version
    # decoded = Dropout(0.3)(LayerNormalization(axis=[2])(LeakyReLU()(decoded)))  # second version
    # decoded = LayerNormalization(axis=[2])(Dropout(0.3)(decoded)) #third

    if 'unet' in data_type:
        if 'separable' in data_type:
            decoded4 = SeparableConv1D(filters[2], kernel_size, strides=1, activation=activation_decode, padding=padding,
                     depthwise_initializer=initializer, pointwise_initializer=initializer, name='decoder4')(decoded3)
        else:
            decoded4 = Conv1D(filters[2], kernel_size, strides=1, activation=activation_decode, padding=padding,
                         kernel_initializer=initializer, name='decoder4')(decoded3)

        decoded4 = Concatenate(axis=-1)([decoded4, sound1])
        decoded4 = Dropout(0.3)(LeakyReLU()(LayerNormalization(axis=[2])(decoded4)))  # first version

        decoded = Conv1D(1, kernel_size, strides=1, activation=activation_all, padding=padding,
                         kernel_initializer=initializer, name='last_layer')(decoded4)

    else:
        decoded = Conv1D(1, kernel_size, strides=1, activation=activation_all, padding=padding,
                         kernel_initializer=initializer, name='last_layer')(decoded3)

    decoded = Activation('tanh')(decoded)


    # define autoencoder
    inputs = [input_sound, input_spike, output_sound] if 'WithSpikes' in data_type \
        else [input_sound, output_sound]
    autoencoder = Model(inputs=inputs, outputs=decoded)

    return autoencoder