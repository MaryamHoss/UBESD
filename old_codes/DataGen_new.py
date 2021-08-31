import h5py
import numpy as np
import tensorflow.keras
from numpy.random import seed

seed(14)
from tensorflow import compat

compat.v1.set_random_seed(14)


class SimpleGenerator_prediction(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 filepath_input_first,
                 filepath_input_second,
                 filepath_output,
                 batch_size=32):
        self.__dict__.update(filepath_input_first=filepath_input_first,
                             filepath_input_second=filepath_input_second,
                             filepath_output=filepath_output,
                             batch_size=batch_size,
                             )

        self.batch_size = batch_size
        self.batch_index = 0
        self.count_lines_in_file()

        self.on_epoch_end()

    def __len__(self):
        # steps_per_epoch, if unspecified, will use the len(generator) as a number of steps.
        # hence this
        self.steps_per_epoch = int(np.floor((self.nb_lines) / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.filepath_input_first, 'r')
        for key in f.keys():

            for line in range(len(f[key])):
                self.nb_lines += 1

    def __getitem__(self, index=0):
        # 'Generate one batch of data'
        # Generate data

        return self.batch_generation()

    def on_epoch_end(self):
        self.input_file_first = h5py.File(self.filepath_input_first, 'r')
        self.input_file_second = h5py.File(self.filepath_input_second, 'r')
        self.output_file = h5py.File(self.filepath_output, 'r')

    def batch_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size
        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1

        list_keys = [key for key in self.input_file_first]
        key = list_keys[0]
        input_batch_first = self.input_file_first[key][batch_start:batch_stop, ::]

        input_batch_first = input_batch_first[:, :-1]

        list_keys = [key for key in self.input_file_second]
        key = list_keys[0]
        input_batch_second = self.input_file_second[key][batch_start:batch_stop, ::]

        input_batch_second = input_batch_second[:, :-1]

        list_keys = [key for key in self.output_file]
        key = list_keys[0]
        # output_batch = self.output_file[key][batch_start:batch_stop, 1:,:]  # [0:self.batch_size,::]
        output_batch = self.output_file[key][batch_start:batch_stop, ::]
        # output_batch = self.output_file[key][batch_start:batch_stop, :13,:]
        output_batch = output_batch[:, 1:]
        return [input_batch_first[:, :, np.newaxis], input_batch_second[:, :, np.newaxis]], output_batch[:, :,
                                                                                            np.newaxis]

class SimpleGenerator_prediction_NoSpike(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 filepath_input,
                 filepath_output,
                 batch_size=32):
        self.__dict__.update(filepath_input=filepath_input,
                             filepath_output=filepath_output,
                             batch_size=batch_size,
                             )

        self.batch_size = batch_size
        self.batch_index = 0
        self.count_lines_in_file()

        self.on_epoch_end()

    def __len__(self):
        # steps_per_epoch, if unspecified, will use the len(generator) as a number of steps.
        # hence this
        self.steps_per_epoch = int(np.floor((self.nb_lines) / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.filepath_input, 'r')
        for key in f.keys():

            for line in range(len(f[key])):
                self.nb_lines += 1

    def __getitem__(self, index=0):
        # 'Generate one batch of data'
        # Generate data

        return self.batch_generation()

    def on_epoch_end(self):
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

        
        list_keys=[key for key in self.input_file]
        key=list_keys[0]
        input_batch = self.input_file[key][batch_start:batch_stop, ::]
              
        input_batch=input_batch[:,:-1]

        
        
        list_keys=[key for key in self.output_file]
        key=list_keys[0]
        #output_batch = self.output_file[key][batch_start:batch_stop, 1:,:]  # [0:self.batch_size,::]
        output_batch = self.output_file[key][batch_start:batch_stop, ::]
        #output_batch = self.output_file[key][batch_start:batch_stop, :13,:]
        output_batch=output_batch[:,1:]
        return input_batch[:,:,np.newaxis],output_batch[:,:,np.newaxis]


class SimpleGenerator_reconstruction(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 filepath_input,
                 filepath_output,
                 batch_size=10):
        self.__dict__.update(filepath_input=filepath_input,
                             filepath_output=filepath_output,
                             batch_size=batch_size,
                             )

        self.batch_size = batch_size
        self.batch_index = 0
        self.count_lines_in_file()

        self.on_epoch_end()

    def __len__(self):
        # steps_per_epoch, if unspecified, will use the len(generator) as a number of steps.
        # hence this
        self.steps_per_epoch = int(np.floor(self.nb_lines / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.filepath_input, 'r')
        for key in f.keys():

            for line in range(len(f[key])):
                self.nb_lines += 1

    def __getitem__(self, index=0):
        # 'Generate one batch of data'
        # Generate data

        return self.batch_generation()

    def on_epoch_end(self):
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

        # key = self.input_file.keys()[0] not able to do this!
        # key = next(iter(self.input_file)) one way to do it
        list_keys = [key for key in self.input_file]
        key = list_keys[0]
        n_rejected_samples = 2
        # input_batch = self.input_file[key][batch_start:batch_stop, :-1,:]  # [0:self.batch_size,::]
        input_batch = self.input_file[key][batch_start:batch_stop, ::]
        # if the input is the spikes, remove the last 2 time_steps to have an integer ratio with the number of sound time_steps
        if input_batch.shape[1] == 23928:
            input_batch = input_batch[:, :23926, :]

        # key = self.output_file.keys()[0]
        list_keys = [key for key in self.output_file]
        key = list_keys[0]
        # output_batch = self.output_file[key][batch_start:batch_stop, 1:,:]  # [0:self.batch_size,::]
        output_batch = self.output_file[key][batch_start:batch_stop, ::]

        print(input_batch.shape)
        return input_batch[:, :, np.newaxis], output_batch[:, :, np.newaxis]


class RandomGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 batch_size=32,
                 type_generator='spk2snd'):
        self.__dict__.update(batch_size=batch_size,
                             type_generator=type_generator
                             )

        self.sound_len = 31900 #95700
        self.spike_len = 7975 #23925

    def __len__(self):
        return 10

    def __getitem__(self, index=0):
        return self.batch_generation()

    def batch_generation(self):
        if self.type_generator == 'spk2snd':
            input_batch = np.random.rand(self.batch_size, self.spike_len, 1)
            output_batch = np.random.rand(self.batch_size, self.sound_len, 1)
        elif self.type_generator == 'snd2snd':
            input_batch = np.random.rand(self.batch_size, self.sound_len, 1)
            output_batch = np.random.rand(self.batch_size, self.sound_len, 1)
        elif self.type_generator == 'sndspk2snd':
            input_batch = [np.random.rand(self.batch_size, self.sound_len, 1),
                           np.random.rand(self.batch_size, self.spike_len, 1)
                           ]
            output_batch = np.random.rand(self.batch_size, self.sound_len, 1)
        else:
            raise NotImplementedError

        return input_batch, output_batch


class SimpleGenerator_denoising(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 filepath_input,
                 filepath_output,
                 batch_size=32):
        self.__dict__.update(filepath_input=filepath_input,
                             filepath_output=filepath_output,
                             batch_size=batch_size,
                             )

        self.batch_size = batch_size
        self.batch_index = 0
        self.count_lines_in_file()

        self.on_epoch_end()

    def __len__(self):
        # steps_per_epoch, if unspecified, will use the len(generator) as a number of steps.
        # hence this
        self.steps_per_epoch = int(np.floor(self.nb_lines / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.filepath_input, 'r')
        for key in f.keys():

            for line in range(len(f[key])):
                self.nb_lines += 1

    def __getitem__(self, index=0):
        # 'Generate one batch of data'
        # Generate data

        return self.batch_generation()

    def on_epoch_end(self):
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

        # key = self.input_file.keys()[0] not able to do this!
        # key = next(iter(self.input_file)) one way to do it
        list_keys = [key for key in self.input_file]
        key = list_keys[0]
        # input_batch = self.input_file[key][batch_start:batch_stop, :-1,:]  # [0:self.batch_size,::]
        input_batch = self.input_file[key][batch_start:batch_stop, :, :]
        # if the input is the spikes, remove the last 2 time_steps to have an integer ratio with the number of sound time_steps
        if input_batch.shape[1] == 23928:
            input_batch = input_batch[:, :23926, :]
            # input_batch = input_batch[:, :4,:]
        if input_batch.shape[1] == 95704:
            input_batch = input_batch[:, :95701, :]
            # input_batch = input_batch[:, :12,:]

        input_batch = input_batch[:, :-1, :]
        # key = self.output_file.keys()[0]
        list_keys = [key for key in self.output_file]
        key = list_keys[0]
        # output_batch = self.output_file[key][batch_start:batch_stop, 1:,:]  # [0:self.batch_size,::]
        output_batch = self.output_file[key][batch_start:batch_stop, :95701, :]
        # output_batch = self.output_file[key][batch_start:batch_stop, :13,:]
        output_batch = output_batch[:, 1:, :]
        return input_batch, output_batch


class SpectrogramGenerator_prediction(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 filepath_input_first,
                 filepath_input_second,
                 filepath_output,
                 batch_size=32):
        self.__dict__.update(filepath_input_first=filepath_input_first,
                             filepath_input_second=filepath_input_second,
                             filepath_output=filepath_output,
                             batch_size=batch_size,
                             )

        self.batch_size = batch_size
        self.batch_index = 0
        self.count_lines_in_file()

        self.on_epoch_end()

    def __len__(self):
        # steps_per_epoch, if unspecified, will use the len(generator) as a number of steps.
        # hence this
        self.steps_per_epoch = int(np.floor((self.nb_lines) / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.filepath_input_first, 'r')
        for key in f.keys():

            for line in range(len(f[key])):
                self.nb_lines += 1

    def __getitem__(self, index=0):
        # 'Generate one batch of data'
        # Generate data

        return self.batch_generation()

    def on_epoch_end(self):
        self.input_file_first = h5py.File(self.filepath_input_first, 'r')
        self.input_file_second = h5py.File(self.filepath_input_second, 'r')
        self.output_file = h5py.File(self.filepath_output, 'r')

    def batch_generation(self):
        batch_start = self.batch_index * self.batch_size
        batch_stop = batch_start + self.batch_size
        if batch_stop > self.nb_lines:
            self.batch_index = 0
            batch_start = self.batch_index * self.batch_size
            batch_stop = batch_start + self.batch_size

        self.batch_index += 1

        list_keys = [key for key in self.input_file_first]
        key = list_keys[0]
        input_batch_first = self.input_file_first[key][batch_start:batch_stop, :,:]

        input_batch_first = input_batch_first[:, :-1,:]

        list_keys = [key for key in self.input_file_second]
        key = list_keys[0]
        input_batch_second = self.input_file_second[key][batch_start:batch_stop, :,:]

        input_batch_second = input_batch_second[:, :-1,:]

        list_keys = [key for key in self.output_file]
        key = list_keys[0]
        # output_batch = self.output_file[key][batch_start:batch_stop, 1:,:]  # [0:self.batch_size,::]
        output_batch = self.output_file[key][batch_start:batch_stop, :,:]
        # output_batch = self.output_file[key][batch_start:batch_stop, :13,:]
        output_batch = output_batch[:, 1:,:]
        return [input_batch_first[:, :,:, np.newaxis], input_batch_second[:, :,:, np.newaxis]], output_batch[:, :,:, np.newaxis]

class SpectrogramGenerator_prediction_NoSpike(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 filepath_input,
                 filepath_output,
                 batch_size=32):
        self.__dict__.update(filepath_input=filepath_input,
                             filepath_output=filepath_output,
                             batch_size=batch_size,
                             )

        self.batch_size = batch_size
        self.batch_index = 0
        self.count_lines_in_file()

        self.on_epoch_end()

    def __len__(self):
        # steps_per_epoch, if unspecified, will use the len(generator) as a number of steps.
        # hence this
        self.steps_per_epoch = int(np.floor((self.nb_lines) / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.filepath_input, 'r')
        for key in f.keys():

            for line in range(len(f[key])):
                self.nb_lines += 1

    def __getitem__(self, index=0):
        # 'Generate one batch of data'
        # Generate data

        return self.batch_generation()

    def on_epoch_end(self):
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

        
        list_keys=[key for key in self.input_file]
        key=list_keys[0]
        input_batch = self.input_file[key][batch_start:batch_stop, :,:]
              
        input_batch=input_batch[:,:-1,:]

        
        
        list_keys=[key for key in self.output_file]
        key=list_keys[0]
        #output_batch = self.output_file[key][batch_start:batch_stop, 1:,:]  # [0:self.batch_size,::]
        output_batch = self.output_file[key][batch_start:batch_stop, :,:]
        #output_batch = self.output_file[key][batch_start:batch_stop, :13,:]
        output_batch=output_batch[:,1:,:]
        return input_batch[:,:,:,np.newaxis],output_batch[:,:,:,np.newaxis]