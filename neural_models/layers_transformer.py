import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def get_angles(pos, i, d_model):
    pos = tf.cast(pos, tf.float32)
    i = tf.cast(i, tf.float32)
    A = pos * (1 / tf.pow(10000., (2 * (i / 2)) / d_model))
    B = tf.cast(pos, tf.float32) * np.pi / 2
    return A + B


def positional_encoding(position, d_model, batch_size):
    angle_rads = get_angles(tf.range(position)[:, None],
                            tf.range(d_model)[None],
                            d_model)
    angle_rads = tf.tile(angle_rads[None], [batch_size, 1, 1])
    pos_encoding = tf.math.cos(angle_rads)
    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


phi = tf.keras.activations.sigmoid  # tf.keras.layers.ELU()(x) + 1


def attention_modified(q, k, v):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    s = tf.math.cumsum(tf.matmul(phi(k), v, transpose_a=True), axis=1)  # (2, 3, dk, vd)
    # z = tf.math.cumsum(phi(k), axis=1)
    num = tf.matmul(phi(q), s, name='this_mul')

    # den = tf.reduce_mean(
    #     tf.matmul(phi(q), z, transpose_b=True, name='maybe_this_mul'),
    #     axis=-1)[..., None]
    #
    # inv_den = 1 / den  # tf.linalg.inv(den)

    output = num  # * inv_den
    # output = tf.matmul(num, inv_den, transpose_a=True)
    return output, tf.zeros_like(output)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def get_config(self):
        return {"d_model": self.d_model,                "num_heads": self.num_heads,                }

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def ProjectionLinformer(d_model):
    input_layer = Input((None, d_model))
    conv = Conv1D(32, 16, 4, padding='causal')(input_layer)
    conv = Conv1D(32, 16, 4, padding='causal')(conv)
    conv = Conv1D(d_model, 16, 4, padding='causal')(conv)
    model = Model(input_layer, conv)
    return model


class LinearSynthesizerInspiredMultiHeadAttention(tf.keras.layers.Layer):
    """
    techniques from:
    # Linformer: Self-Attention with Linear Complexity
    # Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
    # SYNTHESIZER: RETHINKING SELF-ATTENTION FOR TRANSFORMER MODELS

    """

    def __init__(self, d_model, num_heads, kernel_initializer='glorot_uniform'):
        super(LinearSynthesizerInspiredMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        # self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(1 * num_heads, kernel_initializer=kernel_initializer)
        self.wk = tf.keras.layers.Dense(1 * num_heads, kernel_initializer=kernel_initializer)
        # self.wq = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer)
        # self.wk = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer)
        self.wv = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer)

        self.dense = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer)
        # self.time_projection_v = ProjectionLinformer(d_model)
        # self.time_projection_k = ProjectionLinformer(4 * num_heads)
        self.time_projection_v = lambda x: x
        self.time_projection_k = lambda x: x

    def get_config(self):
        return {"d_model": self.d_model,
                "num_heads": self.num_heads,
                }

    def split_heads(self, x, batch_size, depth):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.time_projection_k(self.wk(k))  # (batch_size, seq_len, d_model)
        v = self.time_projection_v(self.wv(v))  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size, 1)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size, 1)  # (batch_size, num_heads, seq_len_k, depth)
        # q = self.split_heads(q, batch_size, self.d_model // self.num_heads)  # (batch_size, num_heads, seq_len_q, depth)
        # k = self.split_heads(k, batch_size, self.d_model // self.num_heads)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size, self.d_model // self.num_heads)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = attention_modified(q, k, v)
        # scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, None)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)

        self.d_model, self.num_heads, self.dff, self.rate = d_model, num_heads, dff, rate

        self.mha = LinearSynthesizerInspiredMultiHeadAttention(d_model, num_heads, kernel_initializer)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None):
        attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    def get_config(self):
        return {"d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "rate": self.rate,
                }


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.dff, self.rate = d_model, num_heads, dff, rate

        self.mha1 = LinearSynthesizerInspiredMultiHeadAttention(d_model, num_heads,
                                                                kernel_initializer=kernel_initializer)
        self.mha2 = LinearSynthesizerInspiredMultiHeadAttention(d_model, num_heads,
                                                                kernel_initializer=kernel_initializer)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        x, enc_output = inputs
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

    def get_config(self):
        return {"d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "rate": self.rate,
                }


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
