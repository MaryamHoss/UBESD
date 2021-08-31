import tensorflow as tf
from TrialsOfNeuralVocalRecon.neural_models.layers_transformer import positional_encoding, EncoderLayer, DecoderLayer, \
    create_look_ahead_mask


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1, kernel_initializer = 'glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate,
                             num_layers=num_layers, maximum_position_encoding=maximum_position_encoding,
                             kernel_initializer=kernel_initializer)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate, kernel_initializer, name='{}_{}'.format(kwargs['name'], i))
            for i in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None):
        # adding embedding and position encoding.
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, :seq_len, :]
        x += positional_encoding(tf.shape(x)[1], self.d_model, tf.shape(x)[0])

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # (batch_size, input_seq_len, d_model)

    def get_config(self):
        return {"d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "num_layers": self.num_layers,
                "maximum_position_encoding": self.maximum_position_encoding,
                "rate": self.rate,
                "kernel_initializer": self.kernel_initializer,
                }


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1, kernel_initializer = 'glorot_uniform', **kwargs):
        super().__init__(**kwargs)

        self.__dict__.update(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate,
                             num_layers=num_layers, maximum_position_encoding=maximum_position_encoding,
                             kernel_initializer=kernel_initializer)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate, kernel_initializer, name='{}_{}'.format(kwargs['name'], i))
            for i in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        x, enc_output = inputs
        attention_weights = {}

        # x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += positional_encoding(tf.shape(x)[1], self.d_model, tf.shape(x)[0])

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i]([x, enc_output])

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

    def get_config(self):
        return {"d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "num_layers": self.num_layers,
                "maximum_position_encoding": self.maximum_position_encoding,
                "rate": self.rate,
                "kernel_initializer": self.kernel_initializer,
                }


class TransformerClassic(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 pe_input, pe_target, rate=0.1, kernel_initializer = 'glorot_uniform'):
        super().__init__()

        self.__dict__.update(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate,
                             num_layers=num_layers, pe_input=pe_input, pe_target=pe_target,
                             kernel_initializer=kernel_initializer)

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate, kernel_initializer, name='spike')
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_target, rate, kernel_initializer, name='sound')

        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        inp, tar = inputs
        # look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        enc_output = self.encoder(inp)  # (batch_size, inp_seq_len, d_model)
        dec_output, attention_weights = self.decoder([tar, enc_output])

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        final_output = tf.keras.layers.Activation('tanh')(final_output)

        return final_output, attention_weights

    def get_config(self):
        return {"d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "num_layers": self.num_layers,
                "pe_input": self.pe_input,
                "pe_target": self.pe_target,
                "rate": self.rate,
                "kernel_initializer": self.kernel_initializer,
                }
