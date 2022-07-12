import time, os, itertools


from UBESD.data_processing.data_generators import Prediction_Generator

CDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data']))


def timeStructured():
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S", named_tuple)
    return time_string


def getDataPaths(data_type, test_type):
    CDIR = os.path.dirname(os.path.realpath(__file__))
    DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data']))

    eeg_folder = 'eeg'

    if 'new' in data_type:
        eeg_folder += '/new'
    
    if 'reduced_ch' in data_type:
        eeg_folder += '/reduced_ch'
    if 'FBC' in data_type:
        eeg_folder = eeg_folder.replace('eeg', 'fbc')

    if 'RAW' in data_type:
        eeg_folder = eeg_folder.replace('eeg', 'raw_eeg')

    if 'speaker_specific' in test_type:
        eeg_folder += '/speaker_specific'
        
    if 'filtered' in data_type:
        eeg_folder += '/filtered'
        
        

    data_paths = {}
    for set in ['train', 'val', 'test']:
        time_folder = '60s' if set == 'test' else '2s'

        if 'kuleuven' in data_type:
            EEG_h5_DIR = os.path.abspath(os.path.join(*[DATADIR, 'kuleuven', 'Normalized']))
        else:
            EEG_h5_DIR = os.path.abspath(os.path.join(*[DATADIR, 'Cocktail_Party', 'Normalized']))


        if 'eeg' in data_type:
            if 'small_eeg_' in data_type:
                data_paths['in1_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, '2s', 'eeg', 'noisy_{}.h5'.format(set)])
                data_paths['in2_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, '2s', 'eeg', 'eegs_{}.h5'.format(set)])
                data_paths['out_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, '2s', 'eeg', 'clean_{}.h5'.format(set)])
                data_paths['out_{}_unattended'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, '2s', 'eeg', 'unattended_{}.h5'.format(set)])

            elif 'denoising' in data_type:
                

                if 'filtered' in data_type:

                    data_paths['in1_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, eeg_folder, 'noisy_{}.h5'.format(set)])
                    data_paths['in2_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, eeg_folder, 'eegs_{}.h5'.format(set)])
                    data_paths['out_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, eeg_folder, 'clean_{}.h5'.format(set)])
                    data_paths['out_{}_unattended'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, eeg_folder, 'unattended_{}.h5'.format(set)])
                    
                    print(data_paths['in1_{}'.format(set)])
                    
                else:
                    
                    
                    data_paths['in1_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, eeg_folder, 'noisy_{}.h5'.format(set)])
                    data_paths['in2_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, eeg_folder, 'eegs_{}.h5'.format(set)])
                    data_paths['out_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, eeg_folder, 'clean_{}.h5'.format(set)])
                    data_paths['out_{}_unattended'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, eeg_folder, 'unattended_{}.h5'.format(set)])
                    
                    print(data_paths['in1_{}'.format(set)])
            else:
                raise NotImplementedError
                
    print(data_paths)


    return data_paths


def getDataPaths_kuleuven(data_type, test_type):
    CDIR = os.path.dirname(os.path.realpath(__file__))
    DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data']))

    eeg_folder = 'eeg'

 

    data_paths = {}
    for set in ['train', 'test']:
        time_folder = '60s' if set == 'test' else '2s'

        EEG_h5_DIR = os.path.abspath(os.path.join(*[DATADIR, 'kuleuven']))




        if 'eeg' in data_type:


            if 'denoising' in data_type:
                

                data_paths['in1_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, time_folder, eeg_folder, 'noisy_{}.h5'.format(set)])
                data_paths['in2_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, time_folder, eeg_folder, 'eegs_{}.h5'.format(set)])
                data_paths['out_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, time_folder, eeg_folder, 'clean_{}.h5'.format(set)])

            else:
                raise NotImplementedError
                
    print(data_paths)


    return data_paths





def getDataPaths_mes(data_type, test_type):
    CDIR = os.path.dirname(os.path.realpath(__file__))
    DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data']))

    eeg_folder = 'eeg'

    if 'new' in data_type:
        if 'short' in data_type:
            eeg_folder += '/new_short'
        elif 'long' in data_type:
            eeg_folder += '/new_long'
            


    data_paths = {}
    for set in ['train', 'test', 'test']:
        time_folder = '4s' #'60s' if set == 'test' else '2s'

        if 'kuleuven' in data_type:
            EEG_h5_DIR = os.path.abspath(os.path.join(*[DATADIR, 'kuleuven', 'Normalized']))
        else:
            EEG_h5_DIR = os.path.abspath(os.path.join(*[DATADIR, 'Mesgarani']))

        if 'eeg' in data_type:
            if 'small_eeg_' in data_type:
                data_paths['in1_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, '2s', 'eeg', 'noisy_{}.h5'.format(set)])
                data_paths['in2_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, '2s', 'eeg', 'eegs_{}.h5'.format(set)])
                data_paths['out_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, '2s', 'eeg', 'clean_{}.h5'.format(set)])
                data_paths['out_{}_unattended'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, '2s', 'eeg', 'unattended_{}.h5'.format(set)])

            elif 'denoising' in data_type:

                data_paths['in1_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, time_folder, eeg_folder, 'noisy_{}.h5'.format(set)])
                data_paths['in2_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, time_folder, eeg_folder, 'eegs_{}.h5'.format(set)])
                data_paths['out_{}'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, time_folder, eeg_folder, 'clean_{}.h5'.format(set)])
                data_paths['out_{}_unattended'.format(set)] = os.path.join(
                    *[EEG_h5_DIR, time_folder, eeg_folder, 'unattended_{}.h5'.format(set)])
            else:
                raise NotImplementedError

    return data_paths



def getData(
        sound_shape=(3, 1),
        spike_shape=(3, 1),
        sound_shape_test=(3, 1),
        spike_shape_test=(3, 1),
        data_type='real_prediction',
        batch_size=128,
        downsample_sound_by=3,
        test_type='speaker_independent',
        cross_number='1st_fold'):
    if not 'random' in data_type:
        data_paths = getDataPaths(data_type, test_type)
        
    #if 'fold' in data_type:
        #data_paths= getDataPaths_cross_val(data_type, test_type,cross_number)

    generators = {}
    if not 'random' in data_type:
        if 'fold' in data_type:

            generator_train, generator_test = [
                Prediction_Generator(
                    filepath_input_first=data_paths['in1_{}'.format(set_name)],
                    filepath_input_second=data_paths['in2_{}'.format(set_name)],
                    filepath_output=data_paths['out_{}'.format(set_name)],
                    sound_shape=sound_shape,
                    spike_shape=spike_shape,
                    batch_size=batch_size,
                    data_type=data_type,
                    downsample_sound_by=downsample_sound_by)
                for set_name in ['train', 'test']]
    
            try:
                generator_test_unattended = Prediction_Generator(
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

        else:
            
            generator_train, generator_val, generator_test = [
                Prediction_Generator(
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
                generator_test_unattended = Prediction_Generator(
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
        generator_train, generator_val, generator_test = [
            Random_Generator(sound_shape=ss, spike_shape=spike_shape, batch_size=b, data_type=data_type,
                             downsample_sound_by=downsample_sound_by)
            for b, ss in zip([batch_size, batch_size, 1],
                             [sound_shape, sound_shape, sound_shape])]
        generators.update(test_unattended=generator_test)
    else:
        raise NotImplementedError
        
    if 'fold' in data_type:
        generators.update(
        train=generator_train,
        test=generator_test)
        
    else:
        generators.update(
        train=generator_train,
        val=generator_val,
        test=generator_test)
    return generators




def getData_mes(
        sound_shape=(3, 1),
        spike_shape=(3, 1),
        sound_shape_test=(3, 1),
        spike_shape_test=(3, 1),
        data_type='real_prediction',
        batch_size=128,
        downsample_sound_by=3,
        test_type='speaker_independent'):
    if not 'random' in data_type:
        data_paths = getDataPaths_mes(data_type, test_type)
        

    generators = {}
    if not 'random' in data_type:

            
        generator_train, generator_val, generator_test = [
            Prediction_Generator(
                filepath_input_first=data_paths['in1_{}'.format(set_name)],
                filepath_input_second=data_paths['in2_{}'.format(set_name)],
                filepath_output=data_paths['out_{}'.format(set_name)],
                sound_shape=sound_shape,
                spike_shape=spike_shape,
                batch_size=batch_size,
                data_type=data_type,
                downsample_sound_by=downsample_sound_by)
            for set_name in ['train', 'test', 'test']]

        try:
            generator_test_unattended = Prediction_Generator(
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
        generator_train, generator_val, generator_test = [
            Random_Generator(sound_shape=ss, spike_shape=spike_shape, batch_size=b, data_type=data_type,
                             downsample_sound_by=downsample_sound_by)
            for b, ss in zip([batch_size, batch_size, 1],
                             [sound_shape, sound_shape, sound_shape])]
        generators.update(test_unattended=generator_test)
    else:
        raise NotImplementedError
        

    generators.update(
    train=generator_train,
    val=generator_val,
    test=generator_test)
    return generators



def getData_kuleuven(
        sound_shape=(3, 1),
        spike_shape=(3, 1),
        sound_shape_test=(3, 1),
        spike_shape_test=(3, 1),
        data_type='real_prediction',
        batch_size=128,
        downsample_sound_by=3,
        test_type='speaker_independent',
        cross_number='1st_fold'):
        

    data_paths = getDataPaths_kuleuven(data_type, test_type)
    generators = {}
    if not 'random' in data_type:       
                   
        generator_train = Prediction_Generator(
                filepath_input_first=data_paths['in1_train'],
                filepath_input_second=data_paths['in2_train'],
                filepath_output=data_paths['out_train'],
                sound_shape=sound_shape,
                spike_shape=spike_shape,
                batch_size=batch_size,
                data_type=data_type,
                downsample_sound_by=downsample_sound_by)
            #for set_name in ['train']]
        
        generator_val=  Prediction_Generator(
                filepath_input_first=data_paths['in1_test'],
                filepath_input_second=data_paths['in2_test'],
                filepath_output=data_paths['out_test'],
                sound_shape=sound_shape_test,
                spike_shape=spike_shape_test,
                batch_size=batch_size,
                data_type=data_type,
                downsample_sound_by=downsample_sound_by)
        generator_test = generator_val
           
           # for set_name in ['test', 'test']]

        try:
            generator_test_unattended = Prediction_Generator(
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

    else:
        raise NotImplementedError
        
    if 'fold' in data_type:
        generators.update(
        train=generator_train,
        test=generator_test)
        
    else:
        generators.update(
        train=generator_train,
        val=generator_val,
        test=generator_test)
    return generators







if __name__ == '__main__':
    data_type = 'denoising_eeg_'
    segment_length = ''
    batch_size = 10
    downsample_sound_by = 2
    generators = getData(sound_shape=(3 * downsample_sound_by, 1),
                         spike_shape=(3, 128),
                         data_type=data_type,
                         segment_length=segment_length,
                         batch_size=batch_size,
                         downsample_sound_by=downsample_sound_by)

    print(generators.keys())
