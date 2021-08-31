import os, json, sys, argparse
import pandas as pd
import scipy
from sklearn.metrics import precision_recall_fscore_support as prfs

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

CDIR = os.path.dirname(os.path.realpath(__file__))
# CDIR='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon'
EXPERIMENTS = os.path.join(*[CDIR, 'experiments'])
# GEXPERIMENTS = r'D:/work/stochastic_spiking/good_experiments/2021-01-05--ptb-small-noise-good'
GEXPERIMENTS = os.path.join(*[CDIR, 'good_experiments'])

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--type', default='violins', type=str, help='main behavior', choices=['excel', 'violins'])
parser.add_argument('--mat', default='no', type=str, help='main behavior', choices=['yes', 'no'])
parser.add_argument('--noise', default='yes', type=str, help='main behavior', choices=['yes', 'no'])
args = parser.parse_args()


def name_experiment(data_type, existing_names=[], causal=False):
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
        metrics_of_interest = ['stoi', 'si-sdr', 'pesq', 'stoi_noisy', 'si-sdr_noisy', 'pesq_noisy']
        non_noisy_metrics = [m for m in metrics_of_interest if not 'noisy' in m]
        noisy_metrics = [m for m in metrics_of_interest if 'noisy' in m]
        if args.mat=='yes':
            #matlab_metrics = ['OPS', 'segSNR', 'fwsegSNR']
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

            if args.mat=='yes':
                PEASS_metrics_path = os.path.join(*[EXPERIMENTS, d, 'OPS_results.mat'])
                if os.path.isfile(PEASS_metrics_path):
                    for m in matlab_metrics:
                        non_noisy_path = os.path.join(*[EXPERIMENTS, d, '{}_results.mat'.format(m)])
                        noisy_path = os.path.join(*[EXPERIMENTS, d, '{}_results_mixture.mat'.format(m)])
                        non_noisy = scipy.io.loadmat(non_noisy_path)[m][:]
                        df[m] = non_noisy
                        if args.noise=='yes':
                            noisy = scipy.io.loadmat(noisy_path)[m][:]
                            df[m + '_noisy'] = noisy

            config_path = os.path.join(*[d_path, '1', 'config.json'])
            with open(config_path) as f:
                config = json.load(f)

            name_exp = name_experiment(str(i) + '_' + config['data_type'], existing_names=list(all_exps.keys()))
            
            all_exps[name_exp] = df[non_noisy_metrics]

        if args.noise=='yes':
            noisy_df = df[noisy_metrics].rename(columns={n: n.replace('_noisy', '') for n in noisy_metrics})
            all_exps['Mixture'] = noisy_df
            noise='yes'
        else:
            noise='no'

        # all_exps = dict(sorted(all_exps.items(), reverse=True))
        evaluations_to_violins(all_exps, EXPERIMENTS, unattended_or_not,'no',noise)

else:
    raise NotImplementedError
