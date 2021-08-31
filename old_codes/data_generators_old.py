# done, changed to correct

import h5py
import numpy as np
from numpy.random import seed
# from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow as tf

# from keras.utils.data_utils import Sequence
# from keras.utils import Sequence
# seed(14)
# from tensorflow import keras
# from keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.data_utils import Sequence

'''if tf.__version__[:2] == '1.':
    from tensorflow import compat
    compat.v1.set_random_seed(14)
elif tf.__version__[:2] == '2.':
    tf.random.set_seed(14)'''


def split_to_timedistributed(batch, terms):
    batch_k = np.split(batch, terms, axis=1)
    batch_k = [np.expand_dims(split, axis=1) for split in batch_k]
    timedistributed_batch = np.concatenate(batch_k, axis=1)
    return timedistributed_batch


def split_and_mix_terms(batch_in_1, batch_out, terms, predict_terms, batch_in_2=None):
    batch_size = batch_in_1.shape[0]
    half_batch_size = int(batch_size / 2)

    batch_in_1 = split_to_timedistributed(batch_in_1, terms)
    batch_out = split_to_timedistributed(batch_out, predict_terms)

    # mix only half of the batch, for the incorrect predictions
    randomfuture_batch = batch_out[:half_batch_size]
    rest_batch = batch_out[half_batch_size:]

    negative_samples = np.zeros(half_batch_size)
    positive_samples = np.ones(batch_size - half_batch_size)
    type_sample = np.concatenate([negative_samples, positive_samples])

    # bring axis 1 to axis 0 cause np.permute only works on the axis 0
    randomfuture_batch = np.swapaxes(randomfuture_batch, 0, 1)
    randomfuture_batch = np.random.permutation(randomfuture_batch)
    randomfuture_batch = np.swapaxes(randomfuture_batch, 0, 1)

    # concatenate correct future with incorrect future and mix in the batch
    batch_out = np.concatenate([randomfuture_batch, rest_batch], axis=0)

    reorder_array = np.random.permutation(range(batch_size))
    batch_out = batch_out[reorder_array]
    type_sample = type_sample[reorder_array]
    batch_in_1 = batch_in_1[reorder_array]

    if not batch_in_2 is None:
        batch_in_2 = split_to_timedistributed(batch_in_2, terms)
        batch_in_2 = batch_in_2[reorder_array]

    return batch_in_1, batch_out, type_sample, batch_in_2


class Prediction_Generator(Sequence):
    def __init__(self,
                 filepath_input_first,
                 filepath_input_second,
                 filepath_output,
                 sound_shape=(3, 1),
                 spike_shape=(3, 1),
                 batch_size=32,
                 data_type='',
                 downsample_sound_by=3):

        self.__dict__.update(filepath_input_first=filepath_input_first,
                             filepath_input_second=filepath_input_second,
                             filepath_output=filepath_output,
                             sound_shape=sound_shape,
                             spike_shape=spike_shape,
                             batch_size=batch_size,
                             data_type=data_type,
                             downsample_sound_by=downsample_sound_by
                             )

        self.input_len_1 = sound_shape[0]
        self.input_len_2 = spike_shape[0]
        self.output_len = spike_shape[0] if 'spike' in filepath_output else sound_shape[0]

        self.batch_size = batch_size
        self.batch_index = 0

        self.count_lines_in_file()
        self.on_epoch_end()

    def __len__(self):
        self.steps_per_epoch = int(np.floor((self.nb_lines) / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.filepath_input_first, 'r')
        for key in f.keys():
            for line in range(len(f[key])):
                self.nb_lines += 1

    def __getitem__(self, index=0):
        return self.batch_generation()

    def on_epoch_end(self):
        self.batch_index = 0
        self.input_file_first = h5py.File(self.filepath_input_first, 'r')
        self.input_file_second = h5py.File(self.filepath_input_second, 'r')
        self.output_file = h5py.File(self.filepath_output, 'r')

        self.input_1_key = [key for key in self.input_file_first][0]
        self.input_2_key = [key for key in self.input_file_second][0]
        self.output_key = [key for key in self.output_file][0]

    def batch_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size
        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1

        input_batch_first = self.input_file_first[self.input_1_key][batch_start:batch_stop, :self.input_len_1]
        input_batch_second = self.input_file_second[self.input_2_key][batch_start:batch_stop, :self.input_len_2]
        output_batch = self.output_file[self.output_key][batch_start:batch_stop, 1:self.output_len + 1]

        if 'noSpikes' in self.data_type:
            return input_batch_first[:, ::self.downsample_sound_by], \
                   output_batch[:, ::self.downsample_sound_by]
        elif 'WithSpikes' in self.data_type:
            return [input_batch_first[:, ::self.downsample_sound_by], input_batch_second], \
                   output_batch[:, ::self.downsample_sound_by]
        else:
            raise NotImplementedError


class Reconstruction_Generator(Prediction_Generator):
    def batch_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size
        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1

        input_batch_first = self.input_file_first[self.input_1_key][batch_start:batch_stop, :self.input_len_1]
        input_batch_second = self.input_file_second[self.input_2_key][batch_start:batch_stop, :self.input_len_2]
        output_batch = self.output_file[self.output_key][batch_start:batch_stop, 1:self.output_len + 1]

        return [input_batch_first[:, :, np.newaxis], input_batch_second[:, :, np.newaxis]], \
               output_batch[:, :, np.newaxis]


class Random_Generator(Sequence):
    def __init__(self,
                 sound_shape=None,
                 spike_shape=None,
                 batch_size=32,
                 data_type=0, #'noSpikes',
                 downsample_sound_by=3):

        self.__dict__.update(batch_size=batch_size, data_type=data_type, downsample_sound_by=downsample_sound_by)

        if data_type == 1:
            self.input_shape = spike_shape
            self.output_shape = sound_shape
        elif data_type == 0:
            self.input_shape = sound_shape
            self.output_shape = sound_shape
        else:
            raise NotImplementedError

    def __len__(self):
        return 2

    def __getitem__(self, index=0):
        return self.batch_generation()

    def batch_generation(self):
        input_batch = np.array(np.random.rand(self.batch_size, *self.input_shape), dtype='float32')
        output_batch = np.array(np.random.rand(self.batch_size, *self.output_shape), dtype='float32')

        if self.data_type == 0:
            output = (input_batch[:, ::self.downsample_sound_by, :],
                      output_batch[:, ::self.downsample_sound_by, :])
        elif self.data_type == 1:
            output = ((output_batch[:, ::self.downsample_sound_by, :], input_batch),
                      output_batch[:, ::self.downsample_sound_by, :])
        else:
            raise NotImplementedError

        return output


class CPC_Generator(Sequence):
    def __init__(self,
                 filepath_input_first,
                 filepath_input_second,
                 filepath_output,
                 sound_shape=None,
                 spike_shape=None,
                 batch_size=32,
                 terms=3, predict_terms=3,
                 data_type='noSpike'):

        if 'random' in data_type:
            self.SourceGenerator = Random_Generator(sound_shape=(sound_shape[0] * terms + 2, 1),
                                                    spike_shape=(spike_shape[0] * terms + 2, 1),
                                                    batch_size=batch_size,
                                                    data_type=data_type)
        else:
            self.SourceGenerator = Reconstruction_Generator(
                filepath_input_first=filepath_input_first,
                filepath_input_second=filepath_input_second,
                filepath_output=filepath_output,
                sound_shape=(sound_shape[0] * terms + 2, 1),
                spike_shape=(spike_shape[0] * terms + 2, 1),
                batch_size=batch_size
            )

        self.__dict__.update(filepath_input_first=filepath_input_first,
                             filepath_input_second=filepath_input_second,
                             filepath_output=filepath_output,
                             batch_size=batch_size,
                             terms=terms, predict_terms=predict_terms,
                             data_type=data_type,
                             sound_shape=sound_shape,
                             spike_shape=spike_shape
                             )

    def __len__(self):
        return self.SourceGenerator.__len__()

    def __getitem__(self, index=0):
        return self.batch_generation()

    def on_epoch_end(self):
        self.SourceGenerator.on_epoch_end()

    def batch_generation(self):
        if 'noSpike' in self.data_type:
            input_batch, output_batch = self.SourceGenerator.batch_generation()

            input_batch = input_batch[:, :-2]
            output_batch = output_batch[:, 1:-1]

            batch_in, batch_out, type_sample, _ = split_and_mix_terms(input_batch, output_batch,
                                                                      self.terms, self.predict_terms)

            type_sample = np.array(type_sample, dtype='float32')
            return [batch_in, batch_out, input_batch[:, -self.sound_shape[0] - 1:-1, :]], \
                   [type_sample[:, np.newaxis], input_batch[:, -self.sound_shape[0]:, :]]

        elif 'WithSpikes' in self.data_type:
            [input_batch_1, input_batch_2], output_batch = self.SourceGenerator.batch_generation()

            input_batch_1 = input_batch_1[:, :-2]
            input_batch_2 = input_batch_2[:, :-2]
            output_batch = output_batch[:, 1:-1]

            batch_in_1, batch_out, type_sample, batch_in_2 = split_and_mix_terms(input_batch_1, output_batch,
                                                                                 self.terms, self.predict_terms,
                                                                                 input_batch_2)

            type_sample = np.array(type_sample, dtype='float32')
            return [batch_in_1, batch_in_2, batch_out,
                    input_batch_1[:, -self.sound_shape[0] - 1:-1, :], input_batch_2[:, -self.spike_shape[0] - 1:-1, :]], \
                   [type_sample[:, np.newaxis], input_batch_1[:, -self.sound_shape[0]:]]
        else:
            raise NotImplementedError
