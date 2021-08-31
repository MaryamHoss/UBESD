'''
This module describes the contrastive predictive coding model from DeepMind:

Oord, Aaron van den, Yazhe Li, and Oriol Vinyals.
"Representation Learning with Contrastive Predictive Coding."
arXiv preprint arXiv:1807.03748 (2018).
'''
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.models import Model

from GenericTools.KerasTools.esoteric_layers.convenience_layers import AverageOverAxis, ExpandDims, Slice
from TrialsOfNeuralVocalRecon.neural_models.layers_transformer import MultiHeadAttention


def network_encoder(x, code_size):
    ''' Define the network mapping images to embeddings '''

    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x


def get_smaller_overshoot(initial_size, final_size, n_upsamplings=3):
    smallest = 1e10
    u_best = 0
    f_best = 0
    for f in [32, 64, 128]:
        for u in range(4):
            U = u ** n_upsamplings
            overshoot = initial_size * U * f - final_size
            if overshoot < smallest and overshoot > 0:
                u_best = u
                f_best = f
                smallest = overshoot

    return u_best, f_best, smallest


def network_decoder_v2(x, code_size, sound_shape):
    n_upsamplings = 3
    u_best, f_best, smallest_overshoot = get_smaller_overshoot(code_size, sound_shape[0], n_upsamplings)

    x = ExpandDims(2)(x)

    for _ in range(n_upsamplings):
        x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='linear')(x)
        x = keras.layers.BatchNormalization()(x)
        x = UpSampling1D(size=u_best)(x)
        x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=f_best, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = keras.layers.Flatten()(x)
    x = Slice(axis=1, initial=smallest_overshoot, final=None)(x)
    x = ExpandDims(2)(x)

    return x


def network_encoder_v2(x, code_size):
    ''' Define the network mapping images to embeddings
    adding a multihead attention at the end to reduce the number of parameters'''

    n_head, d_model, d_k, d_v, dropout = 6, code_size, code_size, code_size, .2
    mha = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv1D(filters=code_size, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)

    # TODO: check if mha layer improves performance or not
    x, _ = mha(x, x, x)
    x = AverageOverAxis(1)(x)

    return x


class DecoderNet(object):
    def __init__(self, code_size):
        input_decoder = keras.layers.Input((code_size,))
        x = keras.layers.Dense(units=256, activation='linear')(input_decoder)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dense(units=1, activation='linear', name='future_generation')(x)
        self.decoder = Model(input_decoder, x)

    def __call__(self, x):
        x = self.decoder(x)
        return x


class Latent2SoundNet(object):
    def __init__(self, code_size, sound_shape):
        snd_decoder_input_2 = keras.layers.Input((code_size,))
        snd_decoder_output_2 = network_decoder_v2(snd_decoder_input_2, code_size, sound_shape)
        self.decoder = keras.models.Model(snd_decoder_input_2, snd_decoder_output_2, name='sound_decoder')
        self.decoder.summary()

    def __call__(self, x):
        x = self.decoder(x)
        return x


class EncoderNet(object):
    def __init__(self, code_size, input_shape, name='encoder'):
        encoder_input = keras.layers.Input(input_shape)
        encoder_output = network_encoder_v2(encoder_input, code_size)
        self.encoder = keras.models.Model(encoder_input, encoder_output, name=name)
        self.encoder.summary()

    def __call__(self, x):
        x = self.encoder(x)
        return x


class AutoregressiveNet(object):
    def __init__(self, code_size, type='MHA'):
        if type == 'GRU':
            self.ar = keras.layers.GRU(units=code_size, return_sequences=False, name='ar_context')
        elif type == 'MHA':
            # TODO: check if mha layer improves performance or not
            n_head, d_model, d_k, d_v, dropout = 6, code_size, code_size, code_size, .2
            mha = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            input_layer = keras.layers.Input((None, code_size,))
            x, _ = mha(input_layer, input_layer, input_layer)
            output = AverageOverAxis(1)(x)
            self.ar = Model(input_layer, output)
        else:
            raise NotImplementedError

    def __call__(self, x):
        x = self.ar(x)
        return x


class PredictionNet(object):
    ''' Define the network mapping context to multiple embeddings '''

    def __init__(self, code_size, predict_terms):
        self.predict_terms = predict_terms

        input_prediction = keras.layers.Input((code_size,))
        self.outputs = []
        for i in range(predict_terms):
            self.outputs.append(
                keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(input_prediction))

        if len(self.outputs) == 1:
            output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(self.outputs[0])
        else:
            output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(self.outputs)

        self.prediction_model = Model(input_prediction, output)

    def __call__(self, context):
        output = self.prediction_model([context] * self.predict_terms)
        return output


class CPCLayer(keras.layers.Layer):
    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def network_cpc_v1(sound_shape, spike_shape, terms, predict_terms, code_size, learning_rate):
    """ CPC net with Decoder part """
    ''' Define the CPC network combining encoder and autoregressive model '''

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # initialize weights of shared nets
    network_autoregressive = AutoregressiveNet(code_size)
    network_decoder = DecoderNet(code_size)
    network_prediction = PredictionNet(code_size, predict_terms)

    # Define sound encoder model (not shared)
    snd_encoder_model = EncoderNet(code_size, sound_shape)

    # Define spike encoder model (not shared)
    spk_encoder_model = EncoderNet(code_size, spike_shape)

    ''' Define the sound2sound net '''
    x_input = keras.layers.Input((terms, sound_shape[0], sound_shape[1]))
    x_encoded = keras.layers.TimeDistributed(snd_encoder_model)(x_input)
    context = network_autoregressive(x_encoded)
    preds = network_prediction(context)

    y_input = keras.layers.Input((predict_terms, sound_shape[0], sound_shape[1]))
    y_encoded = keras.layers.TimeDistributed(snd_encoder_model)(y_input)

    # Loss
    dot_product_probs = CPCLayer()([preds, y_encoded])

    # decoder
    generation = network_decoder(context)

    # Model
    sound2sound_cpc = keras.models.Model(inputs=[x_input, y_input], outputs=[dot_product_probs, generation],
                                         name='sound2sound')

    # Compile model
    sound2sound_cpc.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=['binary_crossentropy', 'mse'],
        metrics=['binary_accuracy']
    )
    sound2sound_cpc.summary()

    ''' Define the spike2sound net '''

    x_input = keras.layers.Input((terms, spike_shape[0], spike_shape[1]))
    x_encoded = keras.layers.TimeDistributed(spk_encoder_model)(x_input)
    context = network_autoregressive(x_encoded)
    preds = network_prediction(context)

    y_input = keras.layers.Input((predict_terms, sound_shape[0], sound_shape[1]))
    y_encoded = keras.layers.TimeDistributed(snd_encoder_model)(y_input)

    # Loss
    dot_product_probs = CPCLayer()([preds, y_encoded])

    # decoder
    generation = network_decoder(context)

    # Model
    spike2sound_cpc = keras.models.Model(inputs=[x_input, y_input], outputs=[dot_product_probs, generation],
                                         name='spike2sound')

    # Compile model
    spike2sound_cpc.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=['binary_crossentropy', 'mse'],
        metrics=['binary_accuracy']
    )
    spike2sound_cpc.summary()

    return sound2sound_cpc, spike2sound_cpc,


def network_cpc_v2(sound_shape, spike_shape, terms, predict_terms, code_size, learning_rate):
    sound2sound, spike2sound, sound2sound_cpc, spike2sound_cpc = [], [], [], []

    """ CPC net with Decoder part """
    ''' Define the CPC network combining encoder and autoregressive model '''

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # initialize weights of shared nets
    network_autoregressive = AutoregressiveNet(code_size)
    latent2sound = Latent2SoundNet(code_size, sound_shape)

    # Define spike and sound encoder models (not shared)
    snd_encoder_model = EncoderNet(code_size, sound_shape, name='sound_encoder').encoder
    spk_encoder_model = EncoderNet(code_size, spike_shape, name='spike_encoder').encoder

    if terms > 0:
        network_prediction = PredictionNet(code_size, predict_terms)

        ''' Define the sound2sound net '''
        x_input = keras.layers.Input((terms, sound_shape[0], sound_shape[1]))
        x_encoded = keras.layers.TimeDistributed(snd_encoder_model, input_shape=(sound_shape[0], sound_shape[1]))(x_input)
        context = network_autoregressive(x_encoded)
        preds = network_prediction(context)

        y_input = keras.layers.Input((predict_terms, sound_shape[0], sound_shape[1]))
        y_encoded = keras.layers.TimeDistributed(snd_encoder_model)(y_input)

        # Loss
        dot_product_probs = CPCLayer()([preds, y_encoded])

        # Model
        sound2sound_cpc = keras.models.Model(inputs=[x_input, y_input], outputs=dot_product_probs, name='sound2sound')

        # Compile model
        sound2sound_cpc.compile(
            optimizer=keras.optimizers.Adam(lr=learning_rate),
            loss=['binary_crossentropy'],
            metrics=['binary_accuracy']
        )
        sound2sound_cpc.summary()

        ''' Define the spike2sound net '''

        x_input = keras.layers.Input((terms, spike_shape[0], spike_shape[1]))
        x_encoded = keras.layers.TimeDistributed(spk_encoder_model)(x_input)
        context = network_autoregressive(x_encoded)
        preds = network_prediction(context)

        y_input = keras.layers.Input((predict_terms, sound_shape[0], sound_shape[1]))
        y_encoded = keras.layers.TimeDistributed(snd_encoder_model)(y_input)

        # Loss
        dot_product_probs = CPCLayer()([preds, y_encoded])

        # Model
        spike2sound_cpc = keras.models.Model(inputs=[x_input, y_input], outputs=dot_product_probs, name='spike2sound')

        # Compile model
        spike2sound_cpc.compile(
            optimizer=keras.optimizers.Adam(lr=learning_rate),
            loss=['binary_crossentropy'],
            metrics=['binary_accuracy']
        )
        spike2sound_cpc.summary()

    ''' Define prediction models '''

    # Define sound2sound model (not shared)
    snd_encoder_input = keras.layers.Input(sound_shape)
    snd_encoder_output = snd_encoder_model(snd_encoder_input)
    sound = latent2sound(snd_encoder_output)
    sound2sound = Model(snd_encoder_input, sound)

    # Define spike encoder model (not shared)
    spk_encoder_input = keras.layers.Input(spike_shape)
    spk_encoder_output = spk_encoder_model(spk_encoder_input)
    sound = latent2sound(spk_encoder_output)
    spike2sound = Model(spk_encoder_input, sound)

    # Compile model
    sound2sound.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=['mse'],
    )
    # Compile model
    spike2sound.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=['mse'],
    )
    return sound2sound, spike2sound, sound2sound_cpc, spike2sound_cpc
