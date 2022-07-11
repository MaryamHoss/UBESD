
import os, itertools, argparse, time, json, datetime

# wmt14 has 281799 steps_per_epoch at a batch size of 16: 26h per epoch

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

parser = argparse.ArgumentParser(description='main')

# types: spiking_transformers, send, summary, scancel:x:y
parser.add_argument('--type', default='send', type=str, help='main behavior')
args = parser.parse_args()


def run_experiments(experiments, init_command='python main_conv_test.py with ',
                    run_string='sbatch run_test.sh '):
    ds = dict2iter(experiments)
    print('Number jobs: {}'.format(len(ds)))
    for d in ds:
        config_update = ''.join(['{}={} '.format(k, v) for k, v in d.items()])
        command = init_command + config_update

        if not 'sound_len' in command: command += ' sound_len=875520'#' sound_len=875520'
        if not 'spike_len' in command: command += ' spike_len=2560'#' spike_len=2560'
        if not 'sound_len_test' in command: command += ' sound_len_test=875520' #' sound_len_test=875520'
        if not 'spike_len_test' in command: command += ' spike_len_test=2560'#' spike_len_test=2560'
        command = run_string + "'{}'".format(command)
        print(command)
        os.system(command)
    print('Number jobs: {}'.format(len(ds)))


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

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_reduced_ch',

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'exp_folder': ['2022-01-28--06-38-03--5786-mc-prediction_']

    }
    
    
  
    
    
    experiments.append(experiment) 
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_filtered_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'exp_folder': ['2022-01-10--18-20-32--4872-mc-prediction_']

    }
    experiments.append(experiment)'''
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_kuleuven',
           

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'exp_folder': ['2022-01-17--08-02-03--0306-mc-prediction_'], 
        'sound_len': [15872],
        'sound_len_test':[158720],
        'spike_len_test':[2560],
        'testing': [True, ]
        
    }
    experiments.append(experiment) '''

    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
           

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'exp_folder': ['2021-09-10--01-58-35--8649-mc-prediction_'], 
        'testing': [True, ]
        
    }
    experiments.append(experiment) 
    
    experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',
           

        ],
        'input_type': ['denoising_eeg_FBC'],
        'test_type': ['speaker_independent'],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'exp_folder': ['2021-09-10--01-58-35--8649-mc-prediction_'], 
        'testing': [True, ],
        'sound_len': [87552],
        'spike_len': [256],
        'sound_len_test':[875520],
        'spike_len_test':[2560],
        
    }
    experiments.append(experiment) '''   


    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_filtered_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'exp_folder': ['2022-01-10--18-20-32--4872-mc-prediction_']

    }
    experiments.append(experiment)''' 
    
    
    
    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_kuleuven',
           

        ],
        'input_type': ['denoising_eeg'],
        'test_type': ['speaker_independent'],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'exp_folder': ['2022-01-10--09-44-03--8157-mc-prediction_'], 
        'sound_len': [158720],
        'sound_len_test':[158720],
        'spike_len_test':[2560],
        'testing': [True, ]
        
    }
    experiments.append(experiment) '''
    
    


    '''experiment = {#new
        'fusion_type': [ #speaker independent filmv2 UBESD FBC

            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4',

        ],
        'input_type': ['denoising_eeg_RAW'],
        'test_type': ['speaker_independent'],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type': ['WithSpikes'],
        'exp_folder': ['2021-12-03--14-50-19--1935-mc-prediction_'],
        'testing': [True, ]
        
    }
    experiments.append(experiment) 
    
    
    
    experiment = {  
                  
        'fusion_type': [
            'FiLM_v6_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-12-03--14-50-20--1152-mc-prediction_'], 

    }
    
    experiments.append(experiment)
    

    experiment = {  
                  
        'fusion_type': [
            'FiLM_v5_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-12-03--14-50-20--4788-mc-prediction_'], 

    }
    
    experiments.append(experiment)'''

    
    
    '''experiment = {  
                  
        'fusion_type': [
            'FiLM_v4_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-07--19-15-54--8771-mc-prediction_'], 

    }
    
    experiments.append(experiment)'''


    
    '''experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['Adam'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-09--11-17-41--7710-mc-prediction_'], 

    }
    
    experiments.append(experiment)


    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:crnd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-10--15-07-59--8434-mc-prediction_'], 

    }
    
    experiments.append(experiment)
    


    experiment = {  
                  
        'fusion_type': [
            'FiLM_v2_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-10--14-13-30--2662-mc-prediction_'], 

    }
    
    experiments.append(experiment)


    experiment = {  
                  
        'fusion_type': [
            'FiLM_v3_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-10--13-54-41--9730-mc-prediction_'], 

    }
    
    experiments.append(experiment)
    

    

    
    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:32:32_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-10--13-42-02--0761-mc-prediction_'], 

    }
    
    experiments.append(experiment)
    
    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_noncausal' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-10--15-07-59--2935-mc-prediction_'], 

    }
    
    experiments.append(experiment)

    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_specific' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-10--13-54-41--9991-mc-prediction_'], 

    }
    
    experiments.append(experiment)'''
    

    '''experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-10--01-58-35--8649-mc-prediction_'], 

    }
    
    experiments.append(experiment)

    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-09--11-17-41--1853-mc-prediction_'], 

    }
    
    experiments.append(experiment)


    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_orthogonal' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-09--11-55-21--0272-mc-prediction_'], 

    }
    
    experiments.append(experiment)
    
    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_specific' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-07--01-17-34--5863-mc-prediction_'], 

    }
    
    experiments.append(experiment)    
    
    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_specific' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['noSpikes'],
        'exp_folder': ['2021-09-08--12-53-42--0392-mc-prediction_'], 

    }
    
    experiments.append(experiment)

    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg', ],
        'test_type': ['speaker_specific' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-07--01-17-34--9514-mc-prediction_'], 

    }
    
    experiments.append(experiment)    


    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3' 
 
        ],
        'input_type': ['denoising_eeg', ],
        'test_type': ['speaker_specific' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-10--02-19-12--5745-mc-prediction_'], 

    }
    
    experiments.append(experiment) 

    
    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-10--01-37-21--9917-mc-prediction_'], 

    }
    
    experiments.append(experiment)    
    
    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-07--01-17-34--7662-mc-prediction_'], 

    }
    
    experiments.append(experiment)
    
    experiment = {  
                  
        'fusion_type': [
            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3' 
 
        ],
        'input_type': ['denoising_eeg', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-09-10--01-37-21--6152-mc-prediction_'], 

    }
    
    experiments.append(experiment)''' 
    
    
    
    
    

    
    '''experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_mes_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_long' 
        ],
        'input_type': ['denoising_eeg', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-21--01-06-14--9593-mc-prediction_'],
        'sound_len_test': [32000],
        'sound_len':[32000],
        'spike_len' :[256]
    }   
    experiments.append(experiment)'''
    
    '''experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v3_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-25--22-31-34--5283-mc-prediction_'], 
    }   
    experiments.append(experiment)'''

    
    run_experiments(experiments)
    
else:
    raise NotImplementedError

