import os, itertools, argparse, time, json, datetime

# wmt14 has 281799 steps_per_epoch at a batch size of 16: 26h per epoch

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

parser = argparse.ArgumentParser(description='main')

# types: spiking_transformers, send, summary, scancel:x:y
parser.add_argument('--type', default='send', type=str, help='main behavior')
args = parser.parse_args()


def run_experiments(experiments, init_command='python main_conv_prediction.py with ',
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

           '_FiLM_v1_new_skip_unet_resnet_initializer:BiGamma_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
           '_FiLM_v1_new_skip_unet_resnet_initializer:BiGamma10_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
           '_FiLM_v1_new_skip_unet_resnet_initializer:BiGammaOrthogonal_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
           '_FiLM_v1_new_skip_unet_resnet_initializer:CauchyOrthogonal_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
           '_FiLM_v1_new_skip_unet_resnet_initializer:GlorotCauchyOrthogonal_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
           '_FiLM_v1_new_skip_unet_resnet_initializer:GlorotOrthogonal_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
           '_FiLM_v1_new_skip_unet_resnet_initializer:GlorotTanh_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
           '_FiLM_v1_new_skip_unet_resnet_initializer:MoreVarianceScalingAndOrthogonal_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
           '_FiLM_v1_new_skip_unet_resnet_initializer:TanhBiGamma10_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
           '_FiLM_v1_new_skip_unet_resnet_initializer:TanhBiGamma10Orthogonal_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
          
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'load_model':[False]

    }
    experiments.append(experiment)   '''
    

    '''experiment = {#new
        'fusion_type': [


            'FiLM_v1_new_skip_unet_resnet_mes_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_long',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'load_model':[False],
        'sound_len':[32000],
        'spike_len' :[256]

    }
    experiments.append(experiment)'''   
    
    
    
    experiment = {#new
        'fusion_type': [


            'FiLM_v1_new_skip_unet_resnet_mes_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_long',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-08-30--11-35-05--5219-mc-prediction_'],   
        'sound_len':[32000],
        'spike_len' :[256]

    }
    experiments.append(experiment)   

    ''' experiment = {#new
        'fusion_type': [

            '_add_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
            '_concatenate_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
            '_choice_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
            '_gating_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
            'FiLM_v1_new_skip_unet_resnet_resnet2_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
            'FiLM_v1_new_skip_unet_resnet_film_first_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
            'FiLM_v1_new_skip_unet_resnet_film_last_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4'

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'load_model':[False]

    }
    experiments.append(experiment) '''
    
    '''experiment = {#new
        'fusion_type': [

            'FiLM_v3_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [16],
        'exp_folder': ['2021-05-24--00-34-24--3446-mc-prediction_'],

    }
    experiments.append(experiment)  
    
    
    experiment = {#new
        'fusion_type': [

            'FiLM_v4_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [19],
        'exp_folder': ['2021-05-24--00-34-24--7223-mc-prediction_'],

    }
    experiments.append(experiment)'''
    
    '''experiment = {#new
        'fusion_type': [

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'load_model':[False]

    }
    experiments.append(experiment) 
    
    experiment = {#new
        'fusion_type': [

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['Adam'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'load_model':[False]

    }
    experiments.append(experiment) 
    
    experiment = {#new
        'fusion_type': [

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_noncausal',

        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'load_model':[False]

    }
    experiments.append(experiment) '''

    '''experiment = {#new
        'fusion_type': [

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:32:32_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'load_model':[False]

    }
    experiments.append(experiment)    

    experiment = {#new
        'fusion_type': [

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_orthogonal',

        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'load_model':[False]

    }
    experiments.append(experiment)    
    
    
    experiment = {#new
        'fusion_type': [

            'FiLM_v1_new_skip_unet_resnet_convblock:crnd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'load_model':[False]

    }
    experiments.append(experiment)'''        
    
    '''experiment = {#new
        'fusion_type': [

            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3',

        ],
        'input_type': ['denoising_eeg', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['Adam'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [9],
        'exp_folder': ['2021-05-22--07-47-01--6962-mc-prediction_'],

    }
    experiments.append(experiment)'''
    
    
    
    '''
    
   
    experiment = {#new
        'fusion_type': [

           'FiLM_v1_new_skip_unet_resnet_convblock:crnd_mmfilter:64:64_dilation:_nconvs:4',
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'epochs':[20],
        'exp_folder': ['2021-04-29--17-25-10--2869-mc-prediction_'],
    }
    experiments.append(experiment)   
   
    
    experiment = {  #new
        'fusion_type': [

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:32:32_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [10],
        'exp_folder': ['2021-05-02--19-16-09--6541-mc-prediction_']
    }
    experiments.append(experiment)  '''  




    run_experiments(experiments)

elif args.type == 'common_voice':
    command = 'python data_processing/common_voice_to_h5.py --type=normalize --n_audios=10000  '
    run_string = 'sbatch run_tf2.sh '
    command = run_string + "'{}'".format(command)
    print(command)
    os.system(command)

else:
    raise NotImplementedError

