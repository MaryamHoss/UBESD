import os, itertools, argparse, time, json, datetime

# wmt14 has 281799 steps_per_epoch at a batch size of 16: 26h per epoch

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

parser = argparse.ArgumentParser(description='main')

# types: spiking_transformers, send, summary, scancel:x:y
parser.add_argument('--type', default='send', type=str, help='main behavior')
args = parser.parse_args()


def run_experiments(experiments, init_command='python main_conv_prediction_cross_val.py with ',
                    run_string='sbatch run_tf.sh '):
    ds = dict2iter(experiments)
    print('Number jobs: {}'.format(len(ds)))
    for d in ds:
        config_update = ''.join(['{}={} '.format(k, v) for k, v in d.items()])
        command = init_command + config_update

        if not 'epochs' in command: command += ' epochs=60'
        if not 'batch_size' in command: command += ' batch_size=8'
        if not 'sound_len' in command: command += ' sound_len=87552'
        if not 'spike_len' in command: command += ' spike_len=256'
        command = run_string + "'{}'".format(command)
        print(command)
        os.system(command)
    print('Number jobs: {}'.format(len(ds)))


def timeStructured():
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S-", named_tuple)
    return time_string


def dict2iter(experiments):
    full_ds = []
    for experiment in experiments:
        c = list(itertools.product(*experiment.values()))
        ds = [{k: v for k, v in zip(experiment.keys(), i)} for i in c]
        full_ds.extend(ds)
    return full_ds


if args.type == 'send':
    experiments = []



    '''experiment = {#new
        'fusion_type': [

            'FiLM_v1_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_4th_fold',

        ],
        'cross_number' :['4th_fold'],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [28],
        'exp_folder':['2021-05-14--01-34-14--4614-mc-prediction_']
    }
    experiments.append(experiment)  

    experiment = {#new
        'fusion_type': [

            'FiLM_v1_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_5th_fold',

        ],
        'cross_number' :['5th_fold'],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [20],
        'exp_folder':['2021-05-14--14-23-52--6797-mc-prediction_']
    }
    experiments.append(experiment)   
    
    
    experiment = {#new
        'fusion_type': [

            'FiLM_v1_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_6th_fold',

        ],
        'cross_number' :['6th_fold'],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [20],
        'exp_folder':['2021-05-14--01-34-14--2194-mc-prediction_']
    }
    experiments.append(experiment) 


    experiment = {
        'fusion_type': [

            'FiLM_v1_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_3rd_fold',

        ],
        'cross_number' :['3rd_fold'],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'load_model':[False]
    }
    experiments.append(experiment)'''
      
    experiment = {
        'fusion_type': [

            'FiLM_v1_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_2nd_fold',

        ],
        'cross_number' :['2nd_fold'],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [32],
        'exp_folder':['2021-05-13--21-10-55--3208-mc-prediction_']
    }
    experiments.append(experiment)   
    
    
    '''experiment = {#new
        'fusion_type': [

            'FiLM_v1_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_1st_fold',

        ],
        'cross_number' :['1st_fold'],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [26],
        'exp_folder':['2021-05-13--21-40-53--8615-mc-prediction_']
    }
    experiments.append(experiment)'''   
 

    run_experiments(experiments)

elif args.type == 'common_voice':
    command = 'python data_processing/common_voice_to_h5.py --type=normalize --n_audios=10000  '
    run_string = 'sbatch run_tf2.sh '
    command = run_string + "'{}'".format(command)
    print(command)
    os.system(command)

else:
    raise NotImplementedError

