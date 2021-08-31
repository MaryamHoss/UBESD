import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, LocallyConnected2D, AveragePooling1D, \
    BatchNormalization, Flatten, Lambda, Conv1D, LeakyReLU, ELU, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from TrialsOfNeuralVocalRecon.neural_models.metrics_and_losses import corr2_mse_loss


def build_model(shp_in, shp_out):
    reg = .0005
    inputs = Input(shape=shp_in)
    x = LocallyConnected2D(1, kernel_size=[5, 5], padding='valid', kernel_initializer='he_normal',
                           kernel_regularizer=l2(reg))(inputs)
    x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = LocallyConnected2D(1, kernel_size=[3, 3], padding='valid', kernel_initializer='he_normal',
                           kernel_regularizer=l2(reg))(x)
    x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = LocallyConnected2D(2, kernel_size=[1, 1], padding='valid', kernel_initializer='he_normal',
                           kernel_regularizer=l2(reg))(x)
    x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = Flatten()(x)

    x_MLP = Flatten()(inputs)
    x_MLP = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
    x_MLP = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x_MLP)))
    x_MLP = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
    x_MLP = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x_MLP)))

    x = concatenate([x, x_MLP], axis=1)

    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x)))
    x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x)))
    x = Dense(shp_out, kernel_initializer='he_normal')(x)
    coded_preds = Activation('tanh', name='coded_preds')(x)
    model = Model(inputs, coded_preds)
    adam = Adam(lr=.0001)
    model.compile(loss=corr2_mse_loss, optimizer=adam)

    return model


class build_autoencoder():
    # initialization
    def __init__(self,
                 feats_shp,
                 spec_shp,
                 lat_dim):
        self.adam = Adam(lr=.0001)
        self.reg = .001
        self.feats_shp = feats_shp
        self.spec_shp = spec_shp
        self.lat_dim = lat_dim

        # building models
        self.encoder = self.build_encoder()
        self.encoder.compile(loss=corr2_mse_loss, optimizer=self.adam)

        self.decoder = self.build_decoder()
        self.decoder.compile(loss=corr2_mse_loss, optimizer=self.adam)

        inputs = Input(shape=(self.feats_shp,))
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        decoded = [self.renamer(decoded[0], 'spec'), self.renamer(decoded[1], 'aper'), self.renamer(decoded[2], 'f0'),
                   self.renamer(decoded[3], 'vuv')]
        self.autoencoder = Model(inputs, decoded)
        self.autoencoder.compile(loss=corr2_mse_loss, optimizer=self.adam)
        # self.autoencoder.summary()

    def renamer(self, x, name):
        renamer_lambda = Lambda(lambda x: x, name=name)
        return renamer_lambda(x)

    # encoder part of auto-encoder
    def build_encoder(self):
        in_encoder = Input(shape=(self.feats_shp,))
        x = Dense(512)(in_encoder)
        x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(400)(x)
        x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(300)(x)
        x = LeakyReLU()(BatchNormalization()(x))
        # x = Dense(200)(x)
        # x = LeakyReLU()(BatchNormalization()(x))
        # x = Dense(100)(x)
        # x = LeakyReLU()(BatchNormalization()(x))
        # x = Dense(50)(x)
        # x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(self.lat_dim)(x)
        x = Activation('tanh')(BatchNormalization()(x))
        out_encoder = x  # noise.GaussianNoise(.2)(x)

        encoder = Model(in_encoder, out_encoder)
        return encoder

    # decoder part of auto-encoder
    def build_decoder(self):
        in_decoder = Input(shape=(self.lat_dim,))
        x = Dense(300)(in_decoder)
        x = LeakyReLU()(BatchNormalization()(x))
        # x = Dense(100)(x)
        # x = LeakyReLU()(BatchNormalization()(x))
        # x = Dense(200)(x)
        # x = LeakyReLU()(BatchNormalization()(x))
        # x = Dense(300)(x)
        # x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(400)(x)
        x = LeakyReLU()(BatchNormalization()(x))
        x = Dense(512)(x)
        x = LeakyReLU()(BatchNormalization()(x))

        # spec branch
        x_spec = Dense(512)(x)
        x_spec = LeakyReLU()(x_spec)
        x_spec = Activation('relu', name='spec')(Dense(self.spec_shp)(x_spec))

        # aperiodicity branch
        x_aper = Dense(32)(x)
        x_aper = LeakyReLU()(BatchNormalization()(x_aper))
        x_aper = Dense(16)(x_aper)
        x_aper = LeakyReLU()(BatchNormalization()(x_aper))
        x_aper = Dense(8)(x_aper)
        x_aper = LeakyReLU()(BatchNormalization()(x_aper))
        x_aper = Dense(4)(x_aper)
        x_aper = LeakyReLU()(x_aper)
        x_aper = Activation('relu', name='aper')(Dense(1)(x_aper))

        # f0 branch
        x_f0 = Dense(32)(x)
        x_f0 = LeakyReLU()(BatchNormalization()(x_f0))
        x_f0 = Dense(8)(x_f0)
        x_f0 = LeakyReLU()(x_f0)
        x_f0 = Activation('relu', name='f0')(Dense(1)(x_f0))

        # vuv branch
        x_vuv = Dense(32)(x)
        x_vuv = LeakyReLU()(BatchNormalization()(x_vuv))
        x_vuv = Dense(8)(x_vuv)
        x_vuv = LeakyReLU()(x_vuv)
        x_vuv = Activation('relu', name='vuv')(Dense(1)(x_vuv))

        decoder = Model(in_decoder, [x_spec, x_aper, x_f0, x_vuv])
        return decoder


def resnet_layer(inputs,
                 num_filters=25,
                 kernel_size=5,
                 strides=1,
                 activation='relu',
                 batch_normalization=False,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='causal',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is 'leakyrelu':
            x = LeakyReLU(alpha=0.3)(x)
        elif activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is 'leakyrelu':
            x = LeakyReLU(alpha=0.3)(x)
        elif activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10, top_classifier=False):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tensorflow.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    if top_classifier == True:
        x = AveragePooling1D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)
    else:
        outputs = x

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def simplified_resnet(input_shape, depth, n_filter, last_filter=1):
    # Start model definition.
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs,
                     activation='leakyrelu')
    # Instantiate the stack of residual units
    for _ in range(depth):
        y = resnet_layer(inputs=x,
                         num_filters=n_filter,
                         activation='leakyrelu')
        x = tensorflow.keras.layers.add([x, y])
        x = LeakyReLU(alpha=0.3)(x)
        # x = Activation('relu')(x)
    x = resnet_layer(inputs=x,
                     num_filters=last_filter,
                     activation='leakyrelu')
    outputs = Dense(last_filter)(x)
    # outputs = LeakyReLU(alpha=0.3)(x)
    outputs = Activation('tanh')(outputs)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def gradual_resnet(input_shape, filters):
    # Start model definition.
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs,
                     activation='leakyrelu',
                     num_filters=filters[0])

    # Instantiate the stack of residual units
    for filter in filters:
        y = resnet_layer(inputs=x,
                         num_filters=filter,
                         activation='leakyrelu')
        x = tensorflow.keras.layers.add([x, y])
        x = LeakyReLU(alpha=0.3)(x)
        # x = Activation('relu')(x)
    x = resnet_layer(inputs=x,
                     num_filters=filters[-1],
                     activation='leakyrelu')
    x = Dense(filters[-1])(x)
    outputs = LeakyReLU(alpha=0.3)(x)
    # outputs = Activation('tanh')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    lat_dim = 256  # 128
    depth = 8  # 5#3
    epochs = 2000
    finetuning_epochs = 0
    sound_len = 31900  # 95704 # prediction:95700 reconstruction:95704 #31900 for when we cut the data into three parts
    spike_len = 7975  # 23926 #prediction:23925 reconstruction 23926 #7975 for when it is divided by three
    data_type = 'real_prediction'  # 'real_reconstruction'  #  #'random'
    batch_size = 32  # 64
    n_filters = 25

    # initialize shared weights
    filters = [3, 40, 3, 1]
    #sound2latent_model = gradual_resnet((sound_len, 1), filters)
    sound2latent_model = simplified_resnet((sound_len, lat_dim), depth, n_filters, 3)

    sound2latent_model.summary()

    """
    latent2sound_model = simplified_resnet((sound_len, lat_dim), depth, n_filters, 1)

    # define sound2sound
    input_sound = Input((sound_len, 1))
    latent_sound = sound2latent_model(input_sound)
    output = latent2sound_model(latent_sound)
    sound2sound = Model(input_sound, output)
    """
