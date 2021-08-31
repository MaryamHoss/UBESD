import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

models_names = ['linear',
                'convolutional',
                'spectrogram',
                'CPC']
n_models = len(models_names)

# with spikes as input
metrics_w = ['ns_5_55_w', 'w_5_55_w', 'na_5_55_w',
           'ns_15_55_w', 'w_15_55_w', 'na_15_55_w',
           'ns_5_65_w', 'w_5_65_w', 'na_5_65_w',
           'ns_15_65_w', 'w_15_65_w', 'na_15_65_w',
           ]

# without spikes as input
metrics_wo = ['ns_5_55_wo', 'w_5_55_wo', 'na_5_55_wo',
           'ns_15_55_wo', 'w_15_55_wo', 'na_15_55_wo',
           'ns_5_65_wo', 'w_5_65_wo', 'na_5_65_wo',
           'ns_15_65_wo', 'w_15_65_wo', 'na_15_65_wo',
           ]
metrics = metrics_wo + metrics_w
n_metrics = len(metrics)
df = pd.DataFrame(np.random.rand(n_models, n_metrics), columns=metrics, index=models_names)


metrics_names = [m[:-2] for m in metrics_w]


fig, axs = plt.subplots(4, 3,
                        figsize=(8, 8), sharex='all', sharey='all',
                        gridspec_kw={'hspace': 0, 'wspace': 0})





for m in metrics_names:
    noise_type = m.split('_')[0]
    snr = m.split('_')[1]
    level = m.split('_')[2]

    if noise_type == 'ns':
        column = 0
    elif noise_type == 'w':
        column = 1
    elif noise_type == 'na':
        column = 2

    if '_5_55' in m:
        row = 0
    elif '_15_55' in m:
        row = 1
    elif '_5_65' in m:
        row = 2
    elif '_15_65' in m:
        row = 3

    df[[m + '_w', m + '_wo']].plot(ax=axs[row, column], kind='bar', rot=16, legend=False)


fig.suptitle("Title for whole figure", fontsize=16)

cols = ['ns', 'w', 'na']
rows = ['5 SNR 55 level', '15 SNR 55 level', '5 SNR 65 level', '15 SNR 65 level']

for ax, col in zip(axs[0], cols):
    ax.set_title(col)

for ax, row in zip(axs[:,0], rows):
    ax.set_ylabel(row, rotation=90, size='large')




#fig.savefig('data/plot.pdf')
