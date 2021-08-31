import os


CDIR = os.path.dirname(os.path.realpath(__file__))
MDIR = os.path.abspath(os.path.join(CDIR, '..'))
EXPERIMENTS = os.path.join(MDIR, 'experiments')

def create_random_experiments(n):
    epochs = 0
    sound_len = 256 * 3
    fusion_types = ['_gating', '_FiLM_v1', '_FiLM_v2', '_FiLM_v3', '_FiLM_v4', '_choice']
    input_type = 'random_'

    for i in range(n):
        fusion_type = fusion_types[i%len(fusion_types)]
        os.system('python {}/main_conv_prediction.py with '.format(MDIR) +
                  'epochs={} sound_len={} fusion_type={} '.format(epochs, sound_len, fusion_type) +
                  'input_type={}'.format(input_type))

if __name__ == '__main__':
    create_random_experiments(n=6)