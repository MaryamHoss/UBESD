import time

import numpy as np

from TrialsOfNeuralVocalRecon.data_processing.DataGen_new import SimpleGenerator_prediction, RandomGenerator, \
    SimpleGenerator_reconstruction,SpectrogramGenerator_prediction,\
        SimpleGenerator_prediction_NoSpike,SpectrogramGenerator_prediction_NoSpike







def timeStructured():
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S", named_tuple)
    return time_string


def getRandomData(data_type='raw_signal', lat_dim=16):
    if data_type == 'vocoder':
        random_aperi = np.random.rand(784, 1)
        random_vuv = np.random.rand(784, 1)
        random_f0 = np.random.rand(784, 1)
        random_spect = np.random.rand(784, 4097)
        return random_spect, random_aperi, random_vuv, random_f0

    elif data_type == 'latent_space':
        return np.random.rand(784, lat_dim)

    elif data_type == 'mesgarani_sound':
        random_sound = np.random.rand(784, 4100)
        return random_sound

    elif data_type == 'mesgarani_spike':
        random_act_1 = np.random.rand(784, 900, 265, 1)
        random_act_2 = np.random.rand(784, 900 * 265 * 1)
        return random_act_1

    elif data_type == 'spike':
        # (11, 23928/8, 20, 265) # 11 sounds, 23928 time_steps, /8 downsampling, 20 trials, 265 neurons
        random = np.random.rand(20 * 265 * 11, 23928)
        random = np.random.rand(4, 23928)
        return random

    elif data_type == 'sound':
        random_sound = np.random.rand(20 * 265 * 11, 4100)
        random_sound = np.random.rand(4, 4100)
        return random_sound

    else:
        raise NotImplementedError



#ok
def getRealData_prediction(batch_size,filepath_input_first,filepath_input_second,filepath_output):
    
    generator = SimpleGenerator_prediction(filepath_input_first=filepath_input_first,
                                           filepath_input_second=filepath_input_second,
                                    filepath_output=filepath_output,
                                    batch_size=batch_size)


    return generator


#ok
def getRealData_prediction_NoSpike(batch_size,filepath_input,filepath_output):
    
    generator = SimpleGenerator_prediction_NoSpike(filepath_input=filepath_input,
                                           filepath_output=filepath_output,
                                           batch_size=batch_size)


def getRealData_reconstruction(filepath_spikes, filepath_stim, data_type):
    if data_type == 'generated data_spk2snd':
        generator = SimpleGenerator_reconstruction(filepath_input=filepath_spikes,
                                                   filepath_output=filepath_stim,
                                                   batch_size=32)

    elif data_type == 'generated data_snd2snd':
        generator = SimpleGenerator_reconstruction(filepath_input=filepath_stim,
                                                   filepath_output=filepath_stim,
                                                   batch_size=32)
    else:
        raise NotImplementedError

    return generator

#ok
def getRealSpectrogramData_prediction(batch_size, filepath_input_first, filepath_input_second,
                                      filepath_output):
    generator = SpectrogramGenerator_prediction(filepath_input_first=filepath_input_first,
                                           filepath_input_second=filepath_input_second,
                                           filepath_output=filepath_output,
                                           batch_size=batch_size)

    return generator 

#ok
def getRealSpectrogramData_prediction_NoSpike(batch_size,filepath_input,filepath_output):
    
    generator = SpectrogramGenerator_prediction_NoSpike(filepath_input=filepath_input,
                                           filepath_output=filepath_output,
                                           batch_size=batch_size)


    return generator


#ok
def getData(data_type='real_prediction', batch_size=128):
    

    if data_type == 'real_prediction':
        """
        
            
        filepath_spikes_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/spikes_train_windowed_normalized.h5'
        filepath_spikes_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/spikes_test_windowed_normalized.h5'
        filepath_stim_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_train_windowed_normalized.h5'
        filepath_stim_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_test_windowed_normalized.h5'
"""
        filepath_spikes_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/spikes_train.h5'
        filepath_spikes_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/spikes_test.h5'
        filepath_stim_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_train.h5'
        filepath_stim_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_test.h5'

        generator_train = getRealData_prediction(batch_size, filepath_stim_train, filepath_spikes_train, filepath_stim_train)
        generator_test = getRealData_prediction(batch_size, filepath_stim_test,filepath_spikes_test, filepath_stim_test)

        
    elif data_type == 'real_reconstruction':
        
        filepath_spikes_train = './data/spikes_train_windowed_normalized.h5'
        filepath_spikes_test = './data/spikes_test_windowed_normalized.h5'
        filepath_stim_train = './data/input_train_windowed_normalized.h5'
        filepath_stim_test = './data/input_test_windowed_normalized.h5'

        generator_train_spk2snd = getRealData_reconstruction(filepath_spikes_train, filepath_stim_train,
                                              data_type='generated data_spk2snd')
        generator_test_spk2snd = getRealData_reconstruction(filepath_spikes_test, filepath_stim_test,
                                             data_type='generated data_spk2snd')

        generator_train_snd2snd = getRealData_reconstruction(filepath_stim_train, filepath_stim_train,
                                              data_type='generated data_snd2snd')
        generator_test_snd2snd = getRealData_reconstruction(filepath_stim_test, filepath_stim_test,
                                             data_type='generated data_snd2snd')
    
    elif data_type == 'denoising':

        filepath_noisy_spikes_train =  '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/spikes_noisy_train.h5'
        filepath_noisy_spikes_test =  '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/spikes_noisy_test.h5'
        filepath_stim_noisy_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_noisy_train.h5'
        filepath_stim_noisy_test =  '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_noisy_test.h5'
        filepath_stim_clean_train =  '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_clean_train.h5'
        filepath_stim_clean_test =  '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_clean_test.h5'
        
        generator_train = getRealData_prediction(batch_size, filepath_stim_noisy_train, filepath_noisy_spikes_train, filepath_stim_clean_train)
        generator_test = getRealData_prediction(batch_size, filepath_stim_noisy_test,filepath_noisy_spikes_test, filepath_stim_clean_test)

        
    elif data_type == 'random':
        generator_train_spk2snd = RandomGenerator(batch_size=batch_size, type_generator='spk2snd')
        generator_test_spk2snd = RandomGenerator(batch_size=batch_size, type_generator='spk2snd')
        generator_train_snd2snd = RandomGenerator(batch_size=batch_size, type_generator='snd2snd')
        generator_test_snd2snd = RandomGenerator(batch_size=batch_size, type_generator='snd2snd')
        
    
    elif data_type == 'real_prediction_NoSpike':
        """
            
        filepath_stim_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_train_windowed_normalized.h5'
        filepath_stim_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_test_windowed_normalized.h5'
"""
        filepath_stim_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_train.h5'
        filepath_stim_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_test.h5'

        generator_train = getRealData_prediction_NoSpike(batch_size, filepath_stim_train, filepath_stim_train)
        generator_test = getRealData_prediction_NoSpike(batch_size, filepath_stim_test,filepath_stim_test)
    elif data_type == 'denoising_NoSpike':
            
        filepath_stim_noisy_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_noisy_train.h5'
        filepath_stim_noisy_test =  '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_noisy_test.h5'
        filepath_stim_clean_train =  '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_clean_train.h5'
        filepath_stim_clean_test =  '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/input_clean_test.h5'
        
        generator_train = getRealData_prediction_NoSpike(batch_size, filepath_stim_noisy_train, filepath_stim_clean_train)
        generator_test = getRealData_prediction_NoSpike(batch_size, filepath_stim_noisy_test,filepath_stim_clean_test)

    else:
        raise NotImplementedError

    return generator_train, generator_test 




#ok
def getSpectrogramData(data_type='real_prediction', batch_size=128):
    
    if data_type == 'real_prediction':
        filepath_spikes_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/spikes_train_normalized.h5'
        filepath_spikes_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/spikes_test_normalized.h5'
        filepath_stim_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_train_normalized.h5'
        filepath_stim_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_test_normalized.h5'

 
        generator_train = getRealSpectrogramData_prediction(batch_size, filepath_stim_train, filepath_spikes_train, filepath_stim_train)
        generator_test = getRealSpectrogramData_prediction(batch_size, filepath_stim_test, filepath_spikes_test, filepath_stim_test)


    
    elif data_type == 'denoising':
        
        filepath_noisy_spikes_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/spikes_noisy_train_normalized.h5'
        filepath_noisy_spikes_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/spikes_noisy_test_normalized.h5'
        filepath_stim_noisy_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_train_noisy_normalized.h5'
        filepath_stim_noisy_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_test_noisy_normalized.h5'
        filepath_stim_clean_train='/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_train_clean_normalized.h5'
        filepath_stim_clean_test='/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_test_clean_normalized.h5'


        generator_train = getRealSpectrogramData_prediction(batch_size, filepath_stim_noisy_train, filepath_noisy_spikes_train,filepath_stim_clean_train)
        generator_test = getRealSpectrogramData_prediction(batch_size, filepath_stim_noisy_test, filepath_noisy_spikes_test,filepath_stim_clean_test)


    elif data_type == 'real_prediction_NoSpike':
            
        filepath_stim_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_train_normalized.h5'
        filepath_stim_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_test_normalized.h5'

        generator_train = getRealSpectrogramData_prediction_NoSpike(batch_size, filepath_stim_train, filepath_stim_train)
        generator_test = getRealSpectrogramData_prediction_NoSpike(batch_size, filepath_stim_test,filepath_stim_test)
    elif data_type == 'denoising_NoSpike':
            
        filepath_stim_noisy_train = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_train_noisy_normalized.h5'
        filepath_stim_noisy_test = '/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_test_noisy_normalized.h5'
        filepath_stim_clean_train='/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_train_clean_normalized.h5'
        filepath_stim_clean_test='/home/hosseinim/work/TrialsOfNeuralVocalRecon/data/gammatones/gamma_waves_test_clean_normalized.h5'
        
        generator_train = getRealSpectrogramData_prediction_NoSpike(batch_size, filepath_stim_noisy_train, filepath_stim_clean_train)
        generator_test = getRealSpectrogramData_prediction_NoSpike(batch_size, filepath_stim_noisy_test,filepath_stim_clean_test)


    else:
        raise NotImplementedError

    return generator_train, generator_test

