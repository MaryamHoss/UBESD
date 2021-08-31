import os
import numpy as np
import matplotlib.pyplot as plt

from TrialsOfNeuralVocalRecon.data_processing.convenience_tools import getData
from innvestigate_guinea import innvestigate

def plot_attention(guinea_model, data_type, save_dir, plot_name='attention_visualization', n_samples=3):

    _, generator_test = getData(data_type=data_type, batch_size=n_samples)
    x = generator_test.__getitem__()[0]

    # Create analyzer
    analyzer = innvestigate.create_analyzer("gradient",
                                            guinea_model)

    for i in range(n_samples):
        sample = [x[0][i][np.newaxis, ...], x[1][i][np.newaxis, ...]]

        # Apply analyzer w.r.t. maximum activated output-neuron
        all_a = analyzer.analyze(sample)

        # Aggregate along color channels and normalize to [-1, 1]
        fig, axs = plt.subplots(len(all_a), figsize=(8, 8))

        fig.suptitle('attention')
        for ax, a, s in zip(axs, all_a, sample):
            # plot training and validation losses
            ax.plot(s[0], label='input')
            ax.plot(a[0], label='attention')
            ax.set_xlabel('time')
            ax.legend()

        plotpath = os.path.join(*[save_dir, plot_name + '_s{}.pdf'.format(i)])
        plt.savefig(plotpath)