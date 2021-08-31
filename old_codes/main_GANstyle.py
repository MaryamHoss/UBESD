import numpy as np
from keras import Input
from keras.layers import MaxPooling1D
from keras.models import Model
from DNNBrainSpeechRecon.data_processing.convenience_tools import getRandomData
from DNNBrainSpeechRecon.neural_tools.neural_models import resnet_v1, simplified_resnet
"""
TODO

- probably better to have a few smaller pooling interleaved with convolutions in the middle instead of a massive pooling 
"""

lat_dim = 16
depth = 3
epochs = 3

spike_train = getRandomData(data_type='spike')[:,:,np.newaxis]
sound_train = getRandomData(data_type='sound')[:,:,np.newaxis]

sound_len = sound_train.shape[1]
spike_len = spike_train.shape[1]

ratio = int(spike_len/sound_len)

print('sound_len', sound_len)
print('spike_len', spike_len)
print(ratio)

# reshape them so they are an integer times the other
# 23,928/5+1 =  3,988‬

sound_train = sound_train[:, :3988, :]
sound_len = sound_train.shape[1]


# initialize shared weights
sound2latent_model = simplified_resnet((sound_len, 1), depth, lat_dim)
latent2sound_model = simplified_resnet((sound_len, lat_dim), depth, 1)
spike2latent_model = simplified_resnet((sound_len, 1), depth, lat_dim)

# define spike2sound
input_spike = Input((spike_len, 1))
# downsample to match the shape of the sound
pooled_spike = MaxPooling1D(pool_size=ratio+1, strides=None, padding='valid', data_format='channels_last')(input_spike)
latent_spike = spike2latent_model(pooled_spike)
output = latent2sound_model(latent_spike)
spike2sound = Model(input_spike, output)
spike2sound.compile('adam', 'mse')

# define sound2sound
input_sound = Input((sound_len, 1))
latent_sound = sound2latent_model(input_sound)
output = latent2sound_model(latent_sound)
sound2sound = Model(input_sound, output)
sound2sound.compile('adam', 'mse')

for _ in range(epochs):
    sound2sound.fit(sound_train, sound_train)
    spike2sound.fit(spike_train, sound_train)