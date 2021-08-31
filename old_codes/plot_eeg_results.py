import os
import os, sys, shutil


sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from TrialsOfNeuralVocalRecon.data_processing.convenience_tools import timeStructured

#CDIR = os.path.dirname(os.path.realpath(__file__))
CDIR = 'C:/Users\hoss3301\work\TrialsOfNeuralVocalRecon'
EXPERIMENTS = os.path.join(CDIR, 'experiments')
#if not os.path.isdir(EXPERIMENTS): os.mkdir(EXPERIMENTS)

time_string = timeStructured()
plot_one_path = os.path.join(EXPERIMENTS, "plot_one-{}________".format(time_string))
if not os.path.isdir(plot_one_path): os.mkdir(plot_one_path)

list_experiments = [d for d in os.listdir(EXPERIMENTS) if 'experiment' in d]
list_experiments = [d for d in list_experiments if not '.zip' in d]

fusions = []
for exp in list_experiments:
    exp_path = os.path.join(*[EXPERIMENTS, exp])
    with open(exp_path + r'/1/run.json', 'r') as inf:
        dict_string = inf.read().replace('false', 'False').replace('null', 'None').replace('true', 'True')
        run_dict = eval(dict_string)

    with open(exp_path + r'/1/config.json', 'r') as inf:
        dict_string = inf.read().replace('false', 'False').replace('null', 'None').replace('true', 'True')
        config_dict = eval(dict_string)

    print()
    print(config_dict['fusion_type'])
    print(run_dict['result'])
    fusions.append(config_dict['fusion_type'])

metrics = list(run_dict['result'].keys())
metrics.remove('loss')
fusions = sorted(np.unique(fusions).tolist())

print(metrics)
print(fusions)

data = np.zeros((len(metrics), len(fusions)))
for exp in list_experiments:
    try:
        exp_path = os.path.join(*[EXPERIMENTS, exp])
        with open(exp_path + r'/1/run.json', 'r') as inf:
            dict_string = inf.read().replace('false', 'False').replace('null', 'None').replace('true', 'True')
            run_dict = eval(dict_string)

        with open(exp_path + r'/1/config.json', 'r') as inf:
            dict_string = inf.read().replace('false', 'False').replace('null', 'None').replace('true', 'True')
            config_dict = eval(dict_string)

        fusion = config_dict['fusion_type']

        f_i = fusions.index(fusion)
        for metric in metrics:
            m_i = metrics.index(metric)
            data[m_i, f_i] = run_dict['result'][metric]
    except Exception as e:
        print(e)

ld = len(metrics)
lm = len(fusions)
width = 1 / lm - .05
X = np.arange(ld)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])

for i in range(lm):
    ax.bar(X + i * width, data.T[i], width=width)

ax.set_ylabel('accuracy')
plt.xticks(X + lm * width / 2, metrics)

fusions = [f.replace('_', '') for f in fusions]
ax.legend(labels=fusions)
plt.savefig(os.path.join(plot_one_path, 'plot_bars_accs.png'), bbox_inches="tight")
