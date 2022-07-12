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
        if not 'spike_len_test' in command: command += ' spike_len_test=2560'
        if not 'sound_len_test' in command: command += ' sound_len_test=875520'
        if not 'hours' in command: command += ' hours=33'
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
    
    experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
            'FiLM_v1_new_skip_unet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'hours':[33],
        'load_model': [False]

    }
    experiments.append(experiment)     

    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_kuleuven',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [90],
        'sound_len': [15872],
        'sound_len_test':[158720],
        'spike_len_test':[2560],
        'hours':[33],
         'exp_folder': ['2022-01-13--11-11-13--8925-mc-prediction_']
       
    }
    experiments.append(experiment)'''
   
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_filtered_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'load_model': [False]
      
    }
    experiments.append(experiment) '''
    
    

    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_kuleuven',
            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3_kuleuven'

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [90],
        'sound_len': [15872],
        'hours':[33],
        'load_model': [False]
        
    }
    experiments.append(experiment)'''

    
    '''
    experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_filtered_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'load_model': [False]
    }
    experiments.append(experiment) '''

    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_kuleuven',
           

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [90],
        'sound_len': [15872],
        'hours':[33],
        'exp_folder': ['2022-01-07--23-13-41--3399-mc-prediction_']
        
    }
    experiments.append(experiment) '''



    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC           
            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3_kuleuven',
            ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [90],
        'sound_len': [15872],
        'hours':[33],
        'load_model': [False]

        
    }
    experiments.append(experiment) '''

    
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_clean_comp'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'load_model': [False]
    }
    experiments.append(experiment) '''
    
    
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_RAW'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-12-01--23-05-47--3404-mc-prediction_']
        
    }
    experiments.append(experiment) 
    
    
    
    experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v5_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-12-01--13-36-56--6993-mc-prediction_']
        
    }
    experiments.append(experiment)     


    experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v6_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-12-01--13-36-57--7590-mc-prediction_']
        
    }
    experiments.append(experiment)''' 
    
    
    

    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC non causal

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
  

        ],
        'input_type': ['denoising_eeg_FBC'],
        'kernel_size': [3],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-10-20--10-51-18--3802-mc-prediction_']
      
    }
    experiments.append(experiment)



 


    experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC non causal

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_separable',
  

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-10-19--13-59-13--6779-mc-prediction_']
      
    }
    experiments.append(experiment)


    experiment = {#new
        'fusion_type': [


            'FiLM_v1_new_skip_unet_resnet_mes_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_long_separable',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'sound_len':[32000],
        'spike_len' :[256],  
        'exp_folder': ['2021-10-20--10-51-18--4128-mc-prediction_']

    }
    experiments.append(experiment)'''
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC non causal

            '_add_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-09-03--13-36-21--7342-mc-prediction_']
    }
    experiments.append(experiment)
  
    experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC non causal

            '_choice_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-09-01--17-02-02--1830-mc-prediction_']
    }
    experiments.append(experiment)    
  
    experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC non causal

            '_concatenate_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-09-01--17-02-02--4866-mc-prediction_']
    }
    experiments.append(experiment)     
  
    experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC non causal

            'FiLM_v1_new_skip_unet_resnet_film_first_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-09-01--17-02-02--8682-mc-prediction_']
    }
    experiments.append(experiment)  
    
    experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC non causal

            'FiLM_v1_new_skip_unet_resnet_film_last_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-09-01--17-02-02--9352-mc-prediction_']
    }
    experiments.append(experiment)

    experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC non causal

            '_gating_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-09-01--17-02-02--2243-mc-prediction_']
    }
    experiments.append(experiment)'''





    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC non causal

            'FiLM_v1_new_skip_unet_resnet_resnet2_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-11--21-38-57--6850-mc-prediction_']
    }
    experiments.append(experiment)'''
    
    
    
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC non causal

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_noncausal',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-09--11-17-41--7646-mc-prediction_']
    }
    experiments.append(experiment)
    
    

    experiment = {#new
        'fusion_type': [ #speaker independent filmv3 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:crnd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-09--11-17-41--0725-mc-prediction_']
    }
    experiments.append(experiment)'''   
    

    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC 32

            'FiLM_v4_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-09--13-29-31--6664-mc-prediction_']
    }
    experiments.append(experiment)



    
    experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC 32

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:32:32_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-09--11-17-41--9780-mc-prediction_']
    }
    experiments.append(experiment)


    experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_specific'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-10--01-58-35--5677-mc-prediction_']
    }
    experiments.append(experiment)

    experiment = {#new
        'fusion_type': [ #speaker independent filmv3 UBESD FBC

            'FiLM_v3_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-09--08-25-22--7979-mc-prediction_']
    }
    experiments.append(experiment)
 
    
    
    experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v2_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-09--08-19-53--3871-mc-prediction_']
    }
    experiments.append(experiment)''' 
    

    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-08--21-19-50--5754-mc-prediction_']
    }
    experiments.append(experiment)
    
    
    experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-08--21-18-22--4418-mc-prediction_']
    }
    experiments.append(experiment)    



    experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-08--21-16-11--4648-mc-prediction_']
    }
    experiments.append(experiment)   
    

    
    experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_specific'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-08--21-16-10--1893-mc-prediction_']
    }
    experiments.append(experiment)'''    
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC adam

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['Adam'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-07--19-37-32--1948-mc-prediction_']
    }
    experiments.append(experiment) 
    
    
    
    experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC no dilation

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-07--23-18-40--5265-mc-prediction_']
    }
    experiments.append(experiment) '''    
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC orthogonal

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_orthogonal',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [30],
        'exp_folder': ['2021-09-07--23-40-28--0428-mc-prediction_']
    }
    experiments.append(experiment)'''
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_specific'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [20],
        'exp_folder': ['2021-09-07--01-17-34--9496-mc-prediction_']
    }
    experiments.append(experiment)'''


    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

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
        'exp_folder': ['2021-09-01--12-20-24--6847-mc-prediction_']
    }
    experiments.append(experiment)'''     
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_specific'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [20],
        'exp_folder': ['2021-09-07--01-17-34--9271-mc-prediction_']
    }
    experiments.append(experiment) 
    
    experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_specific'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['noSpikes'],
        'epochs': [20],
        'exp_folder': ['2021-09-07--01-17-34--7663-mc-prediction_']
    }
    experiments.append(experiment)'''    
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v2_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-05-25--22-31-34--3672-mc-prediction_']
    }
    experiments.append(experiment) 
    

    experiment = {#new
        'fusion_type': [ #speaker independent filmv3 UBESD FBC

            'FiLM_v3_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-05-25--22-31-34--5283-mc-prediction_']
    }
    experiments.append(experiment)    


'''    
    '''experiment = {#new
        'fusion_type': [


            'FiLM_v1_new_skip_unet_resnet_mes_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_short',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-08-31--01-08-40--4296-mc-prediction_'],   
        'sound_len':[32000],
        'spike_len' :[256]

    }
    experiments.append(experiment)   
    
    
    
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
        'exp_folder': ['2021-08-31--10-20-13--8967-mc-prediction_'],   
        'sound_len':[32000],
        'spike_len' :[256]

    }
    experiments.append(experiment)''' 
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv1 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-05-21--01-06-14--9593-mc-prediction_']
    }
    experiments.append(experiment) 
    

    experiment = {#new
        'fusion_type': [#speaker independent filmv1 BESD FBC

            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-07-16--13-23-14--7765-mc-prediction_']
    }
    experiments.append(experiment)
    
    
    
    
    experiment = {#new
        'fusion_type': [ # speaker_specific filmv1 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_specific'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-05-19--16-49-25--1308-mc-prediction_']
    }
    experiments.append(experiment)   
    
    
    experiment = {#new
        'fusion_type': [#speaker_specific filmv1 BESD FBC

            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_specific'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-07-16--13-23-14--2018-mc-prediction_']
    }
    experiments.append(experiment)

    experiment = {#new
        'fusion_type': [ # speaker_specific filmv1 UBESD FBC no spikes

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_specific'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['noSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-05-20--01-32-01--0173-mc-prediction_']
    }
    experiments.append(experiment)

    experiment = {#new
        'fusion_type': [ # speaker_independent filmv1 UBESD EEG

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-05-21--09-07-00--4016-mc-prediction_']
    }
    experiments.append(experiment)

    experiment = {#new
        'fusion_type': [ # speaker_independent filmv1 BESD EEG

            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-07-16--13-23-16--9501-mc-prediction_']
    }
    experiments.append(experiment)



    experiment = {#new
        'fusion_type': [ # speaker_independent filmv1 UBESD EEG

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_specific'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-05-19--22-28-12--9897-mc-prediction_']
    }
    experiments.append(experiment)

    experiment = {#new
        'fusion_type': [ # speaker_independent filmv1 BESD EEG

            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_specific'],
        'testing': [False, ],
        'optimizer': ['cwAdaBelief'],
        'batch_size': [8, ],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'epochs': [60],
        'exp_folder': ['2021-07-16--13-23-14--8519-mc-prediction_']
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

