import h5py
import numpy as np
from numpy.random import seed
from tensorflow.keras.utils import Sequence

seed(14)
from tensorflow import compat

compat.v1.set_random_seed(14)


class Prediction_Generator(Sequence):
    def __init__(self,
                 filepath_input,
                 filepath_output,
                 sound_shape=(3, 1),
                 spike_shape=(3, 1),
                 batch_size=64):
        self.__dict__.update(
            sound_shape=sound_shape,
            spike_shape=spike_shape,
            filepath_input=filepath_input,
            filepath_output=filepath_output,
            batch_size=batch_size,
        )
        self.input_len = spike_shape[0] if 'spike' in filepath_input else sound_shape[0]
        self.output_len = spike_shape[0] if 'spike' in filepath_output else sound_shape[0]

        self.count_lines_in_file()
        self.on_epoch_end()

    def __len__(self):
        self.steps_per_epoch = int(np.floor((self.nb_lines) / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.filepath_input, 'r')
        for key in f.keys():

            for line in range(len(f[key])):
                self.nb_lines += 1

    def __getitem__(self, index=0):
        return self.batch_generation()

    def on_epoch_end(self):
        self.batch_index = 0
        self.input_file = h5py.File(self.filepath_input, 'r')
        self.output_file = h5py.File(self.filepath_output, 'r')

    def batch_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size
        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1
        list_keys = [key for key in self.input_file]
        key = list_keys[0]
        input_batch = self.input_file[key][batch_start:batch_stop, ::]

        list_keys = [key for key in self.output_file]
        key = list_keys[0]
        output_batch = self.output_file[key][batch_start:batch_stop, :]

        input_batch = input_batch[:, :self.input_len]
        output_batch = output_batch[:, 1:self.output_len+1]
        return input_batch[:, :, np.newaxis], output_batch[:, :, np.newaxis]


class Reconstruction_Generator(Prediction_Generator):
    def batch_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size
        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1

        key = [key for key in self.input_file][0]
        input_batch = self.input_file[key][batch_start:batch_stop, :self.input_len]

        key = [key for key in self.output_file][0]
        output_batch = self.output_file[key][batch_start:batch_stop, :self.output_len]

        return input_batch[:, :, np.newaxis], output_batch[:, :, np.newaxis]


class Random_Generator(Sequence):
    def __init__(self,
                 sound_shape=None,
                 spike_shape=None,
                 filepath_input='',
                 filepath_output='',
                 batch_size=32):
        self.__dict__.update(batch_size=batch_size)
        self.input_len = spike_shape[0] if 'spike' in filepath_input else sound_shape[0]
        self.output_len = spike_shape[0] if 'spike' in filepath_output else sound_shape[0]

    def __len__(self):
        return 10

    def __getitem__(self, index=0):
        return self.batch_generation()

    def batch_generation(self):
        input_batch = np.random.rand(self.batch_size, self.input_len)
        output_batch = np.random.rand(self.batch_size, self.output_len)

        return input_batch[:, :, np.newaxis], output_batch[:, :, np.newaxis]


class CPC_Generator(Sequence):
    def __init__(self,
                 filepath_input,
                 filepath_output,
                 sound_shape=None,
                 spike_shape=None,
                 batch_size=32,
                 terms=3, predict_terms=3,
                 is_random=False):

        if is_random:
            self.SourceGenerator = Random_Generator(
                sound_shape=(sound_shape[0] * terms + 2, 1),
                spike_shape=(spike_shape[0] * terms + 2, 1),
                filepath_input=filepath_input,
                filepath_output=filepath_output,
                batch_size=batch_size)
        else:
            self.SourceGenerator = Reconstruction_Generator(
                sound_shape=(sound_shape[0] * terms + 2, 1),
                spike_shape=(spike_shape[0] * terms + 2, 1),
                filepath_input=filepath_input,
                filepath_output=filepath_output,
                batch_size=batch_size)

        self.__dict__.update(filepath_input=filepath_input,
                             filepath_output=filepath_output,
                             batch_size=batch_size,
                             terms=terms, predict_terms=predict_terms
                             )

    def __len__(self):
        return self.SourceGenerator.__len__()

    def __getitem__(self, index=0):
        return self.batch_generation()

    def on_epoch_end(self):
        self.SourceGenerator.on_epoch_end()

    def batch_generation(self):
        input_batch, output_batch = self.SourceGenerator.batch_generation()
        input_batch = input_batch[:, :-2]
        output_batch = output_batch[:, 1:-1]

        batch_in, batch_out, type_sample = split_and_mix_terms(input_batch, output_batch,
                                                               self.terms, self.predict_terms)

        return [batch_in, batch_out], type_sample


def split_to_timedistributed(batch, terms):
    batch_k = np.split(batch, terms, axis=1)
    batch_k = [np.expand_dims(split, axis=1) for split in batch_k]
    timedistributed_batch = np.concatenate(batch_k, axis=1)
    return timedistributed_batch


def split_and_mix_terms(batch_in, batch_out, terms, predict_terms):
    batch_size = batch_in.shape[0]
    half_batch_size = int(batch_size / 2)

    batch_in = split_to_timedistributed(batch_in, terms)
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
    batch_in = batch_in[reorder_array]

    return batch_in, batch_out, type_sample
