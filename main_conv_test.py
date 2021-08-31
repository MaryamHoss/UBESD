import os, sys, shutil, json, time
from tqdm import tqdm

sys.path.append('../')

## For Luca: please put every thing you want to add after this line 
from GenericTools.KerasTools.esoteric_optimizers.AdaBelief import AdaBelief
from TrialsOfNeuralVocalRecon.data_processing.data_collection import getData,getData_mes

import pandas as pd
from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment, ChooseGPU
from GenericTools.StayOrganizedTools.utils import timeStructured
from TrialsOfNeuralVocalRecon.neural_models import build_model
from TrialsOfNeuralVocalRecon.tools.plotting import save_wav,evaluations_to_violins
from tensorflow.keras.optimizers import Adam
from TrialsOfNeuralVocalRecon.tools.calculate_intelligibility import find_intel
from TrialsOfNeuralVocalRecon.tools.utils.losses import *
from tensorflow.keras.models import load_model as lm
import pickle
from GenericTools.KerasTools.convenience_operations import snake
import numpy as np

tf.compat.v1.enable_eager_execution()

from GenericTools.StayOrganizedTools.utils import setReproducible

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
ex = CustomExperiment(random_string + '-mc-test', base_dir=CDIR, seed=14)


# CDIR='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon'
@ex.config
def cfg():
    GPU = 0
    learning_rate = 1e-05
    seed = 14



    fusion_type = 'denoising_eeg_FBCWithSpikes_FiLM_v1_new_convblock:cnrd_mmfilter:5:100_nconvs:3'
    exp_type = 'WithSpikes'

    input_type = 'denoising_eeg_FBC_'  # 'small_eeg_'  # choices: 1) denoising_eeg_ 2) denoising_eeg_FBC_ 3) real_prediction_ 4) random_eeg_
    # 5) real_reconstruction_ 6) denoising_ 7) cpc_prediction_ 8) real_prediction_eeg_ 9) denoising_eeg_RAW_
    # 10) kuleuven_denoising_eeg_ 11) small_eeg_
    data_type = input_type + exp_type + fusion_type
    test_type = 'speaker_independent'
    exp_folder = '2021-04-11--12-36-08--8227-mc-prediction_'
    if 'WithSpikes' in exp_type:
        load_model = os.path.abspath(os.path.join(*[CDIR, 'experiments', exp_folder, 'trained_models',
                                                'model_weights_WithSpikes_predict.h5']))  # wether we start from a previously trained model
    else:
        load_model = os.path.abspath(os.path.join(*[CDIR, 'experiments', exp_folder, 'trained_models',
                                                'model_weights_noSpikes_predict.h5']))  # wether we start from a previously trained model

               
    
    n_channels = 64 if 'mes' in data_type else 128
    testing = True
    optimizer = 'cwAdaBelief'  # adam #adablief
    activation = 'relu'  # sanke
    batch_size_test = 70  # 70 for speaker specific #118 speaker independent
    sound_len_test = 1313280
    spike_len_test = 3840

    downsample_sound_by = 1 if 'mes' in data_type else 3#3  # choices: 3 and 10
    sound_len = 32000 if 'mes' in data_type else 87552  # 87552  # 87040 for downsample by 10 #87552 for downsample sound by=3  # 87552  # insteead of88200  #2626560#2610860
    fs = 8000 if 'mes' in data_type else 44100/downsample_sound_by
    spike_len = 256  # 7680 # 7679


@ex.automain
def main(exp_type, data_type,
         learning_rate, sound_len, spike_len, load_model, n_channels,
         downsample_sound_by, GPU, fs, testing, optimizer, activation,
         test_type, sound_len_test, spike_len_test, seed):
    exp_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    images_dir = os.path.join(*[exp_dir, 'images'])
    text_dir = os.path.join(*[exp_dir, 'text'])
    models_dir = os.path.join(*[exp_dir, 'trained_models'])
    path_best_model = os.path.join(*[models_dir, 'model_weights_{}_predict.h5'.format(exp_type)])
    path_best_optimizer = os.path.join(*[models_dir, 'optimizer_{}_predict.pkl'.format(exp_type)])
    other_dir = os.path.join(*[exp_dir, 'other_outputs'])

    if load_model:
        load_model_folder = load_model.split('trained_models')[0]
        config_path = os.path.join(load_model_folder, '1', 'config.json')
        with open(config_path) as f:
            config = json.load(f)

        data_type = config['data_type']
        config['load_model'] = load_model

        for n in ['history', 'results']:
            old_path = os.path.join(load_model_folder, 'other_outputs', '{}.json'.format(n))

            with open(old_path) as f:
                json_dict = json.load(f)

            new_path = os.path.join(other_dir, '{}.json'.format(n))
            json.dump(json_dict, open(new_path, "w"))

    history_path = other_dir + '/log.csv'

    starts_at, starts_at_s = timeStructured(False, True)

    ChooseGPU(GPU)
    setReproducible(seed)

    model = build_model(learning_rate=learning_rate,
                        sound_shape=(None, 1),
                        spike_shape=(None, n_channels),
                        downsample_sound_by=downsample_sound_by,
                        data_type=data_type)

    if optimizer == 'cwAdaBelief':
        opt = 'cwAdaBelief'
        optimizer = AdaBelief(learning_rate=learning_rate, weight_decay=.1, clipnorm=1.)
    elif optimizer == 'AdaBelief':
        opt = 'AdaBelief'
        optimizer = AdaBelief(learning_rate=learning_rate)
    else:
        opt = 'adam'
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=si_sdr_loss, metrics=[si_sdr_loss, 'mse'])  # dummy_loss
    
    print('loading model')

    if not load_model is False:
        print('Loading weights from {}'.format(load_model))
        if "noiseinput" in data_type:
            model.load_weights(load_model)
            with open(path_best_optimizer, 'rb') as f:
                weight_values = pickle.load(f)
                model.optimizer.set_weights(weight_values)
        else:

            if opt == 'AdaBelief' or opt == 'cwAdaBelief':
                model = lm(load_model, custom_objects={'si_sdr_loss': si_sdr_loss, 'AdaBelief': AdaBelief})
            elif activation == 'snake':
                model = lm(load_model, custom_objects={'si_sdr_loss': si_sdr_loss, 'snake': snake})
            else:
                model = lm(load_model, custom_objects={'si_sdr_loss': si_sdr_loss})

    model.summary()

    ##############################################################
    #                    tests training
    ##############################################################

    if testing:
        print('testing the model')
        print('\n loading the test data')
        if 'mes' in data_type:
        
            generators = getData_mes(sound_shape=(sound_len, 1),
                         spike_shape=(spike_len, n_channels),
                         sound_shape_test=(sound_len_test, 1),
                         spike_shape_test=(spike_len_test, n_channels),
                         data_type=data_type,
                         batch_size=1,
                         downsample_sound_by=downsample_sound_by,
                         test_type=test_type)
        else:
            
            generators = getData(sound_shape=(sound_len_test, 1),
                                 spike_shape=(spike_len_test, n_channels),
                                 sound_shape_test=(sound_len_test, 1),
                                 spike_shape_test=(spike_len_test, n_channels),
                                 data_type=data_type,
                                 batch_size=1,
                                 downsample_sound_by=downsample_sound_by,
                                 test_type=test_type)
        del generators['train']#, generators['val']
        prediction_metrics = ['si-sdr', 'stoi', 'estoi' , 'pesq']
        noisy_metrics = [m + '_noisy' for m in prediction_metrics]

        inference_time = []
        if 'mes' in data_type:
            subjects = [None] + list(range(23,45))#list(range(34)) #[None] + list(range(34))  # None corresponds to all the subjects mixed
        else:
            subjects = [None] + list(range(34)) 
            
        

        for generator_type in ['test', 'test_unattended']:# ['test']:
            
            print(generator_type)
            gt = generators[generator_type]
            all_subjects_evaluations = {}
            print('going through subjects')
            for subject in subjects:

                prediction = []
                df = pd.DataFrame(columns=prediction_metrics + noisy_metrics)
                gt.select_subject(subject)
                print('Subject {}'.format(subject))
                try:
                    for batch_sample, b in tqdm(enumerate(gt)):
                        print('batch_sample {} for subject {}'.format(batch_sample, subject))
                        if 'WithSpikes' in data_type:
                            noisy_snd, clean = b[0][0], b[0][2]  
                        else:
                            noisy_snd, clean = b[0][0], b[0][1] 
                        intel_list, intel_list_noisy = [], []
                        inf_start_s = time.time()
                        print('predicting')
                        pred = model.predict(b[0])
                        print(pred.shape)
                        inf_t = time.time() - inf_start_s
                        if subject is None:
                            inference_time.append(inf_t)

                        prediction.append(pred)
                        prediction_concat = np.concatenate(prediction, axis=0)
                        # uncomment later
                        if not subject is None and not 'mes' in data_type:
                            print('saving sound')
                            save_wav(pred, noisy_snd, clean, exp_type, batch_sample, fs, images_dir, subject,generator_type)
                        # fig_path = os.path.join(
                        # images_dir,
                        # 'prediction_b{}_s{}_g{}.png'.format(batch_sample, subject, generator_type))
                        # print('saving plot')
                        # one_plot_test(pred, clean, noisy_snd, exp_type, '', fig_path)

                        print('finding metrics')
                        for m in prediction_metrics:
                            
                            print('     ', m)
                            pred_m = find_intel(clean, pred, metric=m)
                            intel_list.append(pred_m)
                            # print(pred_m)
                            # print(intel_list)

                            noisy_m = find_intel(clean, noisy_snd, metric=m)
                            intel_list_noisy.append(noisy_m)
                            # print(noisy_m)
                            # print(intel_list_noisy)

                        e_series = pd.Series(intel_list + intel_list_noisy, index=df.columns)
                        df = df.append(e_series, ignore_index=True)

                    if subject is None:
                        prediction_filename = os.path.join(
                            *[images_dir, 'prediction_{}_s{}_g{}.npy'.format(exp_type, subject, generator_type)])
                        print('saving predictions')
                        np.save(prediction_filename, prediction_concat)

                    del prediction, intel_list, intel_list_noisy, pred, prediction_concat, e_series
                    df.to_csv(os.path.join(*[other_dir, 'evaluation_s{}_g{}.csv'.format(subject, generator_type)]),
                              index=False)
                    if not subject is None:
                        print(df)
                        all_subjects_evaluations['Subject {}'.format(subject)] = df

                    '''fig, axs = plt.subplots(1, len(df.columns), figsize=(9, 4))

                    for ax, column in zip(axs, df.columns):
                        ax.set_title(column)
                        violin_handle = ax.violinplot(df[column])
                        violin_handle['bodies'][0].set_edgecolor('black')
                    fig.savefig(os.path.join(*[images_dir, 'metrics_s{}_g{}.png'.format(subject, generator_type)]),
                                bbox_inches='tight')
                    plt.close('all')'''

                    # if subject is None and generator_type == 'test':
                    #     results['inference_time'] = np.mean(inference_time)
                except Exception as e:
                    print(e)

            print('end of code, plotting violins')

            # this part is added to find out why "evaluations_to_violins"  doesnt work
            path_to_test = os.path.join(*[other_dir, 'all_subjects_evaluations_{}.pkl'.format(generator_type)])
            a_file = open(path_to_test, "wb")
            pickle.dump(all_subjects_evaluations, a_file)
            a_file.close()
            '''
            #for loading the pickles
            import pickle
            images_dir=''
            path=''
            a_file = open(path, "rb")
            all_subjects_evaluations=pickle.load(a_file)
            a_file.close()

            prediction_metrics = ['si-sdr', 'stoi','estoi','pesq']
            noisy_metrics = [m + '_noisy' for m in prediction_metrics]
            generator_type='test'
            '''
            #evaluations_to_violins({k: v[noisy_metrics] for k, v in all_subjects_evaluations.items()}, images_dir,
              #                     generator_type + 'noisy')
            #evaluations_to_violins({k: v[prediction_metrics] for k, v in all_subjects_evaluations.items()}, images_dir,
             #                      generator_type + '')

    shutil.make_archive(ex.observers[0].basedir, 'zip', exp_dir)

    return []
