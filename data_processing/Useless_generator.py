import h5py, os
import numpy as np
from numpy.random import seed

from tensorflow.python.keras.utils.data_utils import Sequence

'''if tf.__version__[:2] == '1.':
    from tensorflow import compat
    compat.v1.set_random_seed(14)
elif tf.__version__[:2] == '2.':
    tf.random.set_seed(14)'''




class BaseGenerator(Sequence):
    def __init__(self, data_type=''):
        self.__dict__.update(data_type=data_type)

    def __getitem__(self, index=0):

        batch = self.batch_generation()

        input_batch_first = batch['noisy_sound']
        input_batch_second = batch['input_spikes']
        output_batch = batch['clean_sound']

        if 'noSpikes' in self.data_type:
            return [input_batch_first, output_batch], output_batch
        elif 'WithSpikes' in self.data_type:
            return [input_batch_first, input_batch_second, output_batch], output_batch
        elif 'OnlySpikes' in self.data_type:
            return [input_batch_second, output_batch], output_batch
        else:
            raise NotImplementedError




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
                    input_batch_1[:, -self.sound_shape[0] - 1:-1, :],
                    input_batch_2[:, -self.spike_shape[0] - 1:-1, :]], \
                   [type_sample[:, np.newaxis], input_batch_1[:, -self.sound_shape[0]:]]
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




class Prediction_Generator_yamnet(BaseGenerator):
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
        self.output_len = spike_shape[0] * 62

        self.batch_size = batch_size
        self.batch_index = 0

        self.count_lines_in_file()
        self.on_epoch_end()
        self.ratio_sound_spike = int(spike_shape[0]) / sound_shape[0]
        assert self.ratio_sound_spike.is_integer()
        self.select_subject(None)

    def __len__(self):
        self.steps_per_epoch = int(np.floor((self.nb_lines) / self.batch_size))
        return self.steps_per_epoch

    def count_lines_in_file(self):
        self.nb_lines = 0
        f = h5py.File(self.filepath_input_first, 'r')
        for key in f.keys():
            for line in range(len(f[key])):
                self.nb_lines += 1

    def select_subject(self, subject=2):
        self.samples_of_interest = range(self.input_file_first[self.input_1_key].shape[
                                             0])  # this takes the range of the number of test samples in the data
        if not subject == None:  # if we have a subject
            head, tail = os.path.split(self.filepath_output)  # separates the folder from the file name
            set = [s for s in ['train', 'val', 'test'] if s in tail][
                0]  # takes the name of the test data (clean_test.h5)
            subject_path = os.path.join(*[head, 'subjects_{}.h5'.format(
                set)])  # adds the subjects_test.h5 to the end of the path, so it gives us the subject list path
            subject_file = h5py.File(subject_path, 'r')
            subject_key = [key for key in subject_file][0]
            self.samples_of_interest = [i for i, s in enumerate(subject_file[subject_key][:]) if s == subject]
            # takes the list of subjects, finds all the indexes of the array items that correspons to the subject
        self.nb_lines = len(
            self.samples_of_interest)  # returns the number of the samples in the test data that correspond to the subject of interest
        # if the method select_subject with a subject of interest is not called, this will return all the samples of the data, no problem for train and validation

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

        samples = self.samples_of_interest[
                  batch_start:batch_stop]  # if batch=1, takes each samples index belonging to the subject of interest
        input_batch_first = self.input_file_first[self.input_1_key][samples, :self.input_len_1]  # and load them
        input_batch_second = self.input_file_second[self.input_2_key][samples, :self.input_len_2]
        output_batch = self.output_file[self.output_key][samples, 1:self.output_len + 1]

        input_batch_first = np.repeat(input_batch_first, self.ratio_sound_spike, 1)

        return {'input_spikes': input_batch_second, 'noisy_sound': input_batch_first, 'clean_sound': output_batch}
    
    
    
def getData_yamnet(
        sound_shape=(3, 1),
        spike_shape=(3, 1),
        sound_shape_test=(3, 1),
        spike_shape_test=(3, 1),
        data_type='real_prediction',
        batch_size=128,
        downsample_sound_by=3,
        terms=3, predict_terms=3, test_type='speaker_independent'):
    if not 'random' in data_type:
        data_paths = getDataPaths_yamnet(data_type, test_type)

    generators = {}
    if not 'random' in data_type:

        generator_train, generator_val, generator_test = [
            Prediction_Generator_yamnet(
                filepath_input_first=data_paths['in1_{}'.format(set_name)],
                filepath_input_second=data_paths['in2_{}'.format(set_name)],
                filepath_output=data_paths['out_{}'.format(set_name)],
                sound_shape=sound_shape,
                spike_shape=spike_shape,
                batch_size=batch_size,
                data_type=data_type,
                downsample_sound_by=downsample_sound_by)
            for set_name in ['train', 'val', 'test']]

        try:
            generator_test_unattended = Prediction_Generator_yamnet(
                filepath_input_first=data_paths['in1_test'],
                filepath_input_second=data_paths['in2_test'],
                filepath_output=data_paths['out_test_unattended'],
                sound_shape=sound_shape_test,
                spike_shape=spike_shape_test,
                batch_size=batch_size,
                data_type=data_type,
                downsample_sound_by=downsample_sound_by)

            generators.update(test_unattended=generator_test_unattended)
        except:
            print('Run preprocessed_to_h5.py again to generate the unattended_x.h5')

    elif 'random' in data_type:
        test_sound_shape = (sound_shape[0] * 15, sound_shape[1])  # *30 makes it 60 s
        generator_train, generator_val, generator_test = [Random_Generator(sound_shape=ss,
                                                                           spike_shape=spike_shape,
                                                                           batch_size=b,
                                                                           data_type=data_type,
                                                                           downsample_sound_by=downsample_sound_by)
                                                          for b, ss in zip([batch_size, batch_size, 1],
                                                                           [sound_shape, sound_shape,
                                                                            test_sound_shape])]
        generators.update(test_unattended=generator_test)
    else:
        raise NotImplementedError

    generators.update(
        train=generator_train,
        val=generator_val,
        test=generator_test)
    return generators
    
    
    
