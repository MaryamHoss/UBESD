
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

        if not 'sound_len' in command: command += ' sound_len=87552'#' sound_len=875520'
        if not 'spike_len' in command: command += ' spike_len=256'#' spike_len=2560'
        if not 'sound_len_test' in command: command += ' sound_len_test=87552' #' sound_len_test=875520'
        if not 'spike_len_test' in command: command += ' spike_len_test=256'#' spike_len_test=2560'
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
    
    
    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_mes_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_short' 
 
        ],
        'input_type': ['denoising_eeg', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-08-30--11-35-05--7552-mc-prediction_'], 
        'sound_len':[32000],
        'spike_len' :[256]
    }
    
    experiments.append(experiment)
    
    '''experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v2_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-25--22-31-34--3672-mc-prediction_'], 
    }   
    experiments.append(experiment)
    
    experiment = {  #speaker_specific BESD eeg
                  
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
    experiments.append(experiment)
    
    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v4_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-25--22-31-34--9660-mc-prediction_'], 
    }   
    experiments.append(experiment)    
    
    
    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['Adam'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-28--14-48-10--1786-mc-prediction_'], 
    }   
    experiments.append(experiment)    
    
    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:crnd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-26--23-32-48--4533-mc-prediction_'], 
    }   
    experiments.append(experiment)       
    
    
    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-28--14-48-10--5489-mc-prediction_'], 
    }   
    experiments.append(experiment)  
    
    
    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_noncausal' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-28--14-48-10--4018-mc-prediction_'], 
    }   
    experiments.append(experiment)   
    
    
    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4_orthogonal' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-26--23-32-48--2997-mc-prediction_'], 
    }   
    experiments.append(experiment)   
    
    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:32:32_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-26--23-32-48--5880-mc-prediction_'], 
    }   
    experiments.append(experiment) '''  
    
    '''experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-07-16--13-23-14--7765-mc-prediction_'], 
    }   
    experiments.append(experiment)
    
    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_specific' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-19--16-49-25--1308-mc-prediction_'], 
    }   
    experiments.append(experiment)


    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_specific' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['noSpikes'],
        'exp_folder': ['2021-05-20--01-32-01--0173-mc-prediction_'], 
    }   
    experiments.append(experiment)


    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3' 
 
        ],
        'input_type': ['denoising_eeg_FBC', ],
        'test_type': ['speaker_specific' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-07-16--13-23-14--2018-mc-prediction_'], 
    }   
    experiments.append(experiment)

    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3' 
 
        ],
        'input_type': ['denoising_eeg', ],
        'test_type': ['speaker_specific' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-07-16--13-23-14--8519-mc-prediction_'], 
    }   
    experiments.append(experiment)    

    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg', ],
        'test_type': ['speaker_specific' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-19--22-28-12--9897-mc-prediction_'], 
    }   
    experiments.append(experiment)

    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3' 
 
        ],
        'input_type': ['denoising_eeg', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-07-16--13-23-16--9501-mc-prediction_'], 
    }   
    experiments.append(experiment)

    experiment = {  #speaker_specific BESD eeg
                  
        'fusion_type': [
            'FiLM_v1_new_skip_unet_resnet_convblock:cnrd_mmfilter:64:64_dilation:_nconvs:4' 
 
        ],
        'input_type': ['denoising_eeg', ],
        'test_type': ['speaker_independent' ],
        'testing': [True, ],
        'optimizer': ['cwAdaBelief'],
        'activation': ['relu'],
        'exp_type' :['WithSpikes'],
        'exp_folder': ['2021-05-21--09-07-00--4016-mc-prediction_'], 
    }   
    experiments.append(experiment)'''

    
    run_experiments(experiments)
    
else:
    raise NotImplementedError

