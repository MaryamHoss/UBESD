import os, json, sys, argparse, time
import pandas as pd
import scipy
from sklearn.metrics import precision_recall_fscore_support as prfs
import h5py as hp
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.keras.backend as K

sys.path.append('../')
from GenericTools.StayOrganizedTools.utils import get_random_string

from TrialsOfNeuralVocalRecon.tools.plotting import evaluations_to_violins

from GenericTools.StayOrganizedTools.unzip import unzip_good_exps
from GenericTools.StayOrganizedTools.plot_tricks import large_num_to_reasonable_string

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 1)
pd.set_option('precision', 3)
pd.options.display.width = 500
# pd.options.display.max_colwidth = 16

#CDIR = os.path.dirname(os.path.realpath(__file__))
CDIR='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon'
EXPERIMENTS = os.path.join(*[CDIR, 'experiments'])
# GEXPERIMENTS = r'D:/work/stochastic_spiking/good_experiments/2021-01-05--ptb-small-noise-good'
GEXPERIMENTS = os.path.join(*[CDIR, 'good_experiments'])

DATA_path = os.path.join(*[CDIR, 'data', 'kuleuven'])
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--type', default='violins', type=str, help='main behavior', choices=['si-sdr-test','excel', 'violins','correlation_2'])
parser.add_argument('--mat', default='no', type=str, help='main behavior', choices=['yes', 'no'])
parser.add_argument('--noise', default='no', type=str, help='main behavior', choices=['yes', 'no'])
args = parser.parse_args()


def name_experiment(data_type, existing_names=[], causal=True):
    name = 'BESD'
    # if 'resnet' in data_type:
    # name = 'res' + name

    if 'unet' in data_type:
        name = 'U-' + name

    if 'FBC' in data_type and 'WithSpikes' in data_type:
        name += '\nMUA'
    else:
        name += '\nEEG'

    if not 'WithSpikes' in data_type:
        name = 'Autoencoder'

    if causal:
        if 'noncausal' in data_type:
            name += '\nnon-causal'
        else:
            name += '\ncausal'

    # if 'FiLM_v1' in data_type:
    # name += '\nFilm1'

    if name in existing_names:
        name += ' ' + get_random_string()

    return name


unzip_good_exps(
    GEXPERIMENTS, EXPERIMENTS,
    exp_identifiers=['mc-test'],
    unzip_what=['run.json', 'history', 'evaluation'])
ds = os.listdir(EXPERIMENTS)

time_start = time.perf_counter()


if args.type == 'excel':

    def preprocess_key(k):
        k = k.replace('n_dt_per_step', 'n_dt')
        return k


    def postprocess_results(k, v):
        if k == 'n_params':
            v = large_num_to_reasonable_string(v, 1)
        return v


    history_keys = ['val_mse', 'val_si_sdr_loss', ]

    config_keys = ['batch_size', 'data_type', 'epochs', 'exp_type', 'fusion_type', 'input_type', "test_type",
                   'optimizer', 'testing', ]
    hyperparams_keys = ['duration_experiment']
    extras = ['d_name', 'where', 'actual_epochs_ran', 'precision', 'recall', 'fscore']

    histories = []
    method_names = []
    df = pd.DataFrame(columns=history_keys + config_keys + hyperparams_keys + extras)
    for d in ds:
        d_path = os.path.join(EXPERIMENTS, d)
        history_path = os.path.join(*[d_path, 'other_outputs', 'history.json'])
        hyperparams_path = os.path.join(*[d_path, 'other_outputs', 'results.json'])
        config_path = os.path.join(*[d_path, '1', 'config.json'])
        run_path = os.path.join(*[d_path, '1', 'run.json'])

        with open(config_path) as f:
            config = json.load(f)

        with open(hyperparams_path) as f:
            hyperparams = json.load(f)

        with open(history_path) as f:
            history = json.load(f)

        with open(run_path) as f:
            run = json.load(f)

        results = {}

        if len(extras) > 0:
            results.update({'d_name': d})
            results.update({'where': run['host']['hostname'][:7]})
            results.update({'actual_epochs_ran': len(v) for k, v in history.items()})

            # TODO:
            # load 01s for recall/precision = y_pred
            # y_true = np.ones_like(y_pred)
            # precision, recall, fscore, support = prfs(y_true, y_pred, average='macro')
            # results.update({'precision': precision, 'recall': recall, 'fscore': fscore})

        results.update({k: v for k, v in config.items() if k in config_keys})
        what = lambda k, v: max(v) if 'cat' in k else min(v)
        results.update(
            {k.replace('output_net_', '').replace('categorical_', ''): what(k, v) for k, v in history.items() if
             k in history_keys})
        results.update({k: postprocess_results(k, v) for k, v in hyperparams.items() if k in hyperparams_keys})

        small_df = pd.DataFrame([results])

        df = df.append(small_df)
        # method_names.append(config['net_name'] + '_' + config['task_name'])
        # khistory = lambda x: None
        # khistory.history = history
        # histories.append(khistory)

    # val_categorical_accuracy val_bpc
    df = df.sort_values(by=['val_si_sdr_loss'], ascending=True)

    print()
    print(df.to_string(index=False))
    save_path = d_path = os.path.join(CDIR, 'All_results.xlsx')
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
    df.to_excel(writer, na_rep='NaN', sheet_name='Sheet1', index=False)
    worksheet = writer.sheets['Sheet1']  # pull worksheet object
    for idx, col in enumerate(df.columns):  # loop through all columns
        series = df[col]
        max_len = max((series.astype(str).map(len).max(),  # len of largest item
                       len(str(series.name))  # len of column name/header
                       )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width

    writer.save()
    # plot_filename = os.path.join(*['experiments', 'transformer_histories.png'])
    # plot_history(histories=histories, plot_filename=plot_filename, epochs=results['final_epochs'],
    #              method_names=method_names)

elif args.type == 'violins':
    print(ds)
    for unattended_or_not in ['test']:  # , 'test_unattended']:
        evaluation_file = 'evaluation_sNone_g{}.csv'.format(unattended_or_not)
        metrics_of_interest =['pesq','pesq_noisy'] #['pesq','pesq_noisy'] # ['stoi','stoi_noisy']#['pesq','pesq_noisy'] #['si-sdr', 'si-sdr_noisy']#['stoi','stoi_noisy']#, 'si-sdr', 'pesq', 'stoi_noisy', 'si-sdr_noisy', 'pesq_noisy']
        non_noisy_metrics = [m for m in metrics_of_interest if not 'noisy' in m]
        noisy_metrics = [m for m in metrics_of_interest if 'noisy' in m]
        if args.mat == 'yes':
            # matlab_metrics = ['OPS', 'segSNR', 'fwsegSNR']
            matlab_metrics = ['OPS']
            for m in matlab_metrics:
                non_noisy_metrics.append(m)
                noisy_metrics.append(m + '_noisy')

        new_ds = []
        for d in ds:
            evaluation_path = os.path.join(*[EXPERIMENTS, d, 'other_outputs', evaluation_file])
            if os.path.isfile(evaluation_path):
                df = pd.read_csv(evaluation_path)
                c_names = list([n for n in df.columns])  # if not 'noisy' in n])
                if all([c in c_names for c in metrics_of_interest]):
                    new_ds.append(d)

        ds = new_ds

        all_exps = {}
        for i, d in enumerate(ds):
            d_path = os.path.join(EXPERIMENTS, d)
            evaluation_path = os.path.join(*[d_path, 'other_outputs', evaluation_file])
            df = pd.read_csv(evaluation_path)

            '''if args.mat == 'yes':
                PEASS_metrics_path = os.path.join(*[EXPERIMENTS, d, 'OPS_results.mat'])
                if os.path.isfile(PEASS_metrics_path):
                    for m in matlab_metrics:
                        non_noisy_path = os.path.join(*[EXPERIMENTS, d, '{}_results.mat'.format(m)])
                        noisy_path = os.path.join(*[EXPERIMENTS, d, '{}_results_mixture.mat'.format(m)])
                        non_noisy = scipy.io.loadmat(non_noisy_path)[m][:]
                        df[m] = non_noisy
                        if args.noise == 'yes':
                            noisy = scipy.io.loadmat(noisy_path)[m][:]
                            df[m + '_noisy'] = noisy'''

            config_path = os.path.join(*[d_path, '1', 'config.json'])
            with open(config_path) as f:
                config = json.load(f)

            name_exp = name_experiment(str(i) + '_' + config['data_type'], existing_names=list(all_exps.keys()))

            all_exps[name_exp] = df[non_noisy_metrics]

        if args.noise == 'yes':
            noisy_df = df[noisy_metrics].rename(columns={n: n.replace('_noisy', '') for n in noisy_metrics})
            all_exps['Mixture'] = noisy_df
            noise = 'yes'
        else:
            noise = 'no'

        #all_exps = dict(sorted(all_exps.items(), reverse=True))
        evaluations_to_violins(all_exps, EXPERIMENTS, unattended_or_not, 'no', noise)
        




elif args.type == 'correlation':
    unattended_or_not = 'test'

    # definition of pearson_r that gives back a (batch_size, 1) metric
    def pearson_r(y_true, y_pred):
        # original: https://github.com/WenYanger/Keras_Metrics/
        x = y_true.astype('float32')
        y = y_pred.astype('float32')
        mx = tf.reduce_mean(x, axis=1, keepdims=True)
        my = tf.reduce_mean(y, axis=1, keepdims=True)
        xm, ym = x - mx, y - my
        r_num = tf.reduce_sum(xm * ym, axis=1)
        x_square_sum = tf.reduce_sum(xm * xm, axis=1)
        y_square_sum = tf.reduce_sum(ym * ym, axis=1)
        r_den = tf.sqrt(x_square_sum * y_square_sum)
        r = r_num / r_den
        return np.array(r)

    d = '2022-01-13--11-52-02--4350-mc-test_' # '2022-01-03--13-49-44--8570-mc-test_' #'2022-01-10--23-30-56--6889-mc-test_' 

    #for d in d:
    reconstructed_data_file = 'prediction_WithSpikes_sNone_g{}.npy'.format(unattended_or_not)
    evaluation_path = os.path.join(*[EXPERIMENTS, d, 'images', reconstructed_data_file])
    if os.path.isfile(evaluation_path):

        reconstructed_data = np.load(evaluation_path)  # shape is (64,158720,1)

        clean_data_file = 'clean_{}.h5'.format(unattended_or_not)
        clean_data_path = os.path.join(*[DATA_path, clean_data_file])
        clean_data_f = hp.File(clean_data_path, 'r')
        clean_data = clean_data_f['clean_test'][:]
        clean_data_f.close()
        clean_data = clean_data[:, 0:np.shape(reconstructed_data)[1]]  # shape is (64,158720,1)
        unattended_data_file = 'unattended_{}.h5'.format(unattended_or_not)
        unattended_data_path = os.path.join(*[DATA_path, unattended_data_file])
        unattended_data_f = hp.File(unattended_data_path, 'r')
        unattended_data = unattended_data_f['unattended_test'][:]
        unattended_data_f.close()
        unattended_data = unattended_data[:, 0:np.shape(reconstructed_data)[1]]  # shape is (64,158720,1)

      
        

        clean = clean_data[..., 0]
        unattended = unattended_data[..., 0]
        reconstructed = reconstructed_data[..., 0]

        speaker1_pt1 = clean[:, 0:80000]
        speaker1_pt2 = unattended[:, 80001:]
        speaker1 = np.concatenate((speaker1_pt1, speaker1_pt2), axis=1)

        speaker2_pt2 = clean[:, 80001:]
        speaker2_pt1 = unattended[:, 0:80000]
        speaker2 = np.concatenate((speaker2_pt1, speaker2_pt2), axis=1)

        #del unattended, clean_data, unattended_data, reconstructed_data
        #del speaker2_pt2, speaker2_pt1, speaker1_pt1, speaker1_pt2

        win = np.int(9 * 8000)  # 5 second window
        hop = np.int(500)
        
        col_to_add = (np.floor((win - np.shape(reconstructed)[1]) % hop + hop + win -1)/2).astype(int)

        speaker1 = np.pad(speaker1, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)

        speaker2 = np.pad(speaker2, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)

        reconstructed = np.pad(reconstructed, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)
        
        clean=np.pad(clean, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)

        no_of_windows = int(1 + np.floor((np.shape(reconstructed)[1] - win) / hop))

        r_s1 = []
        r_s2 = []

        # find the correlation coefficient for each segment
        for j in range(no_of_windows):
            print(j)
            r_win_s1= pearson_r(speaker1[:,j * hop:j*hop+ win - 1], reconstructed[:,j * hop:j*hop+ win - 1])
            #r_win_s1= scipy.stats.spearmanr(speaker1[:,j * hop:j*hop+ win - 1], reconstructed[:,j * hop:j*hop+ win - 1],axis=1)
            r_s1.append(r_win_s1)

            r_win_s2= pearson_r(speaker2[:,j * hop:j*hop+ win - 1], reconstructed[:,j * hop:j*hop+ win - 1])
            #r_win_s2= scipy.stats.spearmanr(speaker2[:,j * hop:j*hop+ win - 1], reconstructed[:,j * hop:j*hop+ win - 1],axis=1)

            r_s2.append(r_win_s2)


        # put all the correlation coeffcients together to make an array with the same length as the reconstructed array
        r_s1 = np.array(r_s1)
        r_s2 = np.array(r_s2)

        # now put them all together for the 64 samples
        r_all_subject_s1 = r_s1
        r_all_subject_s2 = r_s2

elif args.type == 'si-sdr-test':
    unattended_or_not = 'test'

    # definition of pearson_r that gives back a (batch_size, 1) metric
    def log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator


    def si_sdr(y_true, y_pred):
        # print("######## SI-SDR LOSS ########")
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
    
        #x = tf.squeeze(y_true, axis=-1)
        x=y_true
        #y = tf.squeeze(y_pred, axis=-1)
        y=y_pred
        smallVal = 1e-9  # To avoid divide by zero
        a = K.sum(y * x, axis=-1, keepdims=True) / (K.sum(x * x, axis=-1, keepdims=True) + smallVal)
    
        xa = a * x
        xay = xa - y
        d = K.sum(xa * xa, axis=-1, keepdims=True) / (K.sum(xay * xay, axis=-1, keepdims=True) + smallVal)
        # d1=tf.zeros(d.shape)
        d1 = d == 0
        d1 = 1 - tf.cast(d1, tf.float32)
    
        d = K.mean(10 * d1 * log10(d + smallVal),axis=-1)
        return  np.array(d)




    d = '2022-01-13--11-52-02--4350-mc-test_'#'2022-01-17--15-07-23--9702-mc-test_'#'2022-01-13--11-52-02--4350-mc-test_' # '2022-01-03--13-49-44--8570-mc-test_' #'2022-01-10--23-30-56--6889-mc-test_' 

    #for d in d:
    reconstructed_data_file = 'prediction_WithSpikes_sNone_g{}.npy'.format(unattended_or_not)
    evaluation_path = os.path.join(*[EXPERIMENTS, d, 'images', reconstructed_data_file])
    if os.path.isfile(evaluation_path):

        reconstructed_data = np.load(evaluation_path)  # shape is (64,158720,1)

        clean_data_file = 'clean_{}.h5'.format(unattended_or_not)
        clean_data_path = os.path.join(*[DATA_path, clean_data_file])
        clean_data_f = hp.File(clean_data_path, 'r')
        clean_data = clean_data_f['clean_test'][:]
        clean_data_f.close()
        clean_data = clean_data[:, 0:np.shape(reconstructed_data)[1]]  # shape is (64,158720,1)
        unattended_data_file = 'unattended_{}.h5'.format(unattended_or_not)
        unattended_data_path = os.path.join(*[DATA_path, unattended_data_file])
        unattended_data_f = hp.File(unattended_data_path, 'r')
        unattended_data = unattended_data_f['unattended_test'][:]
        unattended_data_f.close()
        unattended_data = unattended_data[:, 0:np.shape(reconstructed_data)[1]]  # shape is (64,158720,1)

      
        

        clean = clean_data[..., 0]
        unattended = unattended_data[..., 0]
        reconstructed = reconstructed_data[..., 0]

        speaker1_pt1 = clean[:, 0:80000]
        speaker1_pt2 = unattended[:, 80001:]
        speaker1 = np.concatenate((speaker1_pt1, speaker1_pt2), axis=1)

        speaker2_pt2 = clean[:, 80001:]
        speaker2_pt1 = unattended[:, 0:80000]
        speaker2 = np.concatenate((speaker2_pt1, speaker2_pt2), axis=1)

        #del unattended, clean_data, unattended_data, reconstructed_data
        #del speaker2_pt2, speaker2_pt1, speaker1_pt1, speaker1_pt2

        win = np.int(0.25*8000)  # 5 second window
        hop = np.int(1)
        
        speaker1 = np.pad(speaker1, ((0,0),(win-1, 0)), 'constant', constant_values=0)

        speaker2 = np.pad(speaker2, ((0,0),(win-1, 0)), 'constant', constant_values=0)

        reconstructed = np.pad(reconstructed, ((0,0),(win-1, 0)), 'constant', constant_values=0)        
        
        
        '''col_to_add = (np.floor((win - np.shape(reconstructed)[1]) % hop + hop + win -1)/2).astype(int)
        
        
        speaker1 = np.pad(speaker1, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)

        speaker2 = np.pad(speaker2, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)

        reconstructed = np.pad(reconstructed, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)
        
        clean=np.pad(clean, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)'''
        
        

        '''speaker1 = np.pad(speaker1, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)

        speaker2 = np.pad(speaker2, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)

        reconstructed = np.pad(reconstructed, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)
        
        clean=np.pad(clean, ((0,0),(col_to_add, col_to_add)), 'constant', constant_values=0)'''

        no_of_windows = int(1 + np.floor((np.shape(reconstructed)[1] - win) / hop))

        r_s1 = []
        r_s2 = []

        # find the correlation coefficient for each segment
        for j in range(no_of_windows):
            print(j)
            r_win_s1= si_sdr(speaker1[:,j * hop:j*hop+ win - 1], reconstructed[:,j * hop:j*hop+ win - 1])
            #r_win_s1= scipy.stats.spearmanr(speaker1[:,j * hop:j*hop+ win - 1], reconstructed[:,j * hop:j*hop+ win - 1],axis=1)
            r_s1.append(r_win_s1)

            r_win_s2= si_sdr(speaker2[:,j * hop:j*hop+ win - 1], reconstructed[:,j * hop:j*hop+ win - 1])
            #r_win_s2= scipy.stats.spearmanr(speaker2[:,j * hop:j*hop+ win - 1], reconstructed[:,j * hop:j*hop+ win - 1],axis=1)

            r_s2.append(r_win_s2)


        # put all the correlation coeffcients together to make an array with the same length as the reconstructed array
        r_s1 = np.array(r_s1)
        r_s2 = np.array(r_s2)

        # now put them all together for the 64 samples
        r_all_subject_s1 = r_s1
        r_all_subject_s2 = r_s2


else:
    raise NotImplementedError

time_elapsed = (time.perf_counter() - time_start)
print('All done, in ' + str(time_elapsed) + 's')
