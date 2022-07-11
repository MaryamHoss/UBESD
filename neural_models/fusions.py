from tensorflow.keras.layers import *

from GenericTools.KerasTools.configuration_performer_attention import PerformerAttentionConfig
from GenericTools.KerasTools.modeling_tf_performer_attention import TFPerformerAttention

import itertools


def prime_factors(n):
    for i in itertools.chain([2], itertools.count(3, 2)):
        if n <= 1:
            break
        while n % i == 0:
            n //= i
            yield i


def FiLM_Fusion(size, data_type='sum', initializer='orthogonal', n_filters=0):
    def fuse(inputs):
        sound, spikes = inputs
        if 'FiLM_v1' in data_type or 'FiLM_v2' in data_type:
            # FiLM starts -------
            beta_snd = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(spikes)
            gamma_snd = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(spikes)

            beta_spk = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(sound)
            gamma_spk = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(sound)

            # changes: 20-8-20 instead of + I made a layer with ADD
            # sound = Multiply()([sound, gamma_snd]) + beta_snd
            sound = Add()([Multiply()([sound, gamma_snd]), beta_snd])
            # spikes = Multiply()([spikes, gamma_spk]) + beta_spk
            spikes = Add()([Multiply()([spikes, gamma_spk]), beta_spk])

            # FiLM ends ---------

            layer = [sound, spikes]

        elif 'FiLM_v3' in data_type or 'FiLM_v4' in data_type:
            # FiLM starts -------
            beta_snd = Dense(size)(spikes)
            gamma_snd = Dense(size)(spikes)

            beta_spk = Dense(size)(sound)
            gamma_spk = Dense(size)(sound)
            # changes: 20-8-20 instead of + I made a layer with ADD

            # sound = Multiply()([sound, gamma_snd]) + beta_snd
            sound = Add()([Multiply()([sound, gamma_snd]), beta_snd])
            # spikes = Multiply()([spikes, gamma_spk]) + beta_spk
            spikes = Add()([Multiply()([spikes, gamma_spk]), beta_spk])

            # FiLM ends ---------

            layer = [sound, spikes]
            
        elif 'FiLM_v5' in data_type:
            
            #just modulating the sound
            # FiLM starts -------
            beta_snd = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(spikes)
            gamma_snd = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(spikes)


            # changes: 20-8-20 instead of + I made a layer with ADD
            # sound = Multiply()([sound, gamma_snd]) + beta_snd
            sound = Add()([Multiply()([sound, gamma_snd]), beta_snd])
            # spikes = Multiply()([spikes, gamma_spk]) + beta_spk

            # FiLM ends ---------

            layer = [sound, spikes]        
            
        elif 'FiLM_v6' in data_type:
            # FiLM starts -------

            #just modulating the spikes
            beta_spk = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(sound)
            gamma_spk = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(sound)

            # changes: 20-8-20 instead of + I made a layer with ADD
            # sound = Multiply()([sound, gamma_snd]) + beta_snd
            # spikes = Multiply()([spikes, gamma_spk]) + beta_spk
            spikes = Add()([Multiply()([spikes, gamma_spk]), beta_spk])

            # FiLM ends ---------

            layer = [sound, spikes]       
            
            

        elif 'performer' in data_type:
            config = PerformerAttentionConfig()
            p = list(prime_factors(n_filters))
            config.d_model = n_filters
            config.num_heads = 1 if (p[-1] == p[0] and len(p) == 1) else p[-1]
            config.causal = True
            config.use_orthogonal_features = True

            performer_sound = TFPerformerAttention(config)
            query, key = spikes, sound
            per_sound, = performer_sound([query, key, key], mask=None, head_mask=None, output_attentions=False)
            performer_spike = TFPerformerAttention(config)
            query, key = sound, spikes
            per_spike, = performer_spike([query, key, key], mask=None, head_mask=None, output_attentions=False)
            layer = [per_sound, per_spike]
        else:
            layer = inputs
        return layer

    return fuse


def Fusion(data_type='Add'):
    def fuse(inputs):
        sound, spikes = inputs

        if 'noSpike' in data_type:
            layer = sound

        elif 'OnlySpike' in data_type:
            layer = spikes

        elif '_add' in data_type:
            layer = Add()(inputs)

        elif '_concatenate' in data_type:
            layer = Concatenate(axis=-1)(inputs)

        elif '_choice' in data_type:
            layer = ChoiceGated()(inputs)

        elif 'FiLM_v1' in data_type or 'FiLM_v3' in data_type or 'FiLM_v5' in data_type or 'FiLM_v6' in data_type:
            layer = Concatenate(axis=-1)(inputs)

        elif 'FiLM_v2' in data_type or 'FiLM_v4' in data_type:
            layer = sound

        elif '_gating' in data_type:
            gated = Multiply()([sound, spikes])
            gated = Activation('softmax')(gated)
            gated = Multiply()([gated, sound])

            concat = Concatenate(axis=-1)([sound, gated])
            layer = concat


        else:
            raise NotImplementedError

        return layer

    return fuse


def ChoiceGated():
    # output =  probs*sound + (1-probs)*spike
    def gating(inputs):
        sound, spikes = inputs
        mul = Multiply()([sound, spikes])
        probs = Activation('softmax')(mul)
        p_snd = Multiply()([probs, sound])
        # changed from sound to spikes
        # p_spk = Multiply()([1 - probs, sound])
        p_spk = Multiply()([1 - probs, spikes])

        output = Add()([p_snd, p_spk])
        return output

    return gating
