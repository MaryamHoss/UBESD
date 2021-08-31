import tarfile, os, argparse, sys, h5py, aubio
import tables

aubio.quiet = True
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from datetime import timedelta

sys.path.append('../')
sys.path.append('../..')

from GenericTools.StayOrganizedTools.download_utils import download_url
from GenericTools.StayOrganizedTools.utils import timeStructured

CDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data', 'common_voice']))
CLIPS = os.path.abspath(os.path.join(*[CDIR, '..', 'data', 'common_voice', 'extracted', 'clips']))
TARPATH = os.path.join(DATADIR, 'en.tar')
H5S = os.path.join(DATADIR, 'voices_h5s')
if not os.path.isdir(DATADIR): os.mkdir(DATADIR)
if not os.path.isdir(H5S): os.mkdir(H5S)

url = 'https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-1/en.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3BAMIG5W4%2F20210416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210416T084800Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzENL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDDbhkel0oQEaa9p9%2BiKSBHg9Tcs3jIdgDnoR%2Fyvxs0p38iGkT08egrdSszcFmeyg2zevinG5wwerdMCsT%2FrviWxJjASVdvcALMFE%2FOG9rVSpyDYYT%2BfCBiqO5bNrUpJKHZoPW0tw6oXmxzYYO6s%2B7HpaQOnfVO7mZ7SD8xOLn8PaZ%2BFQneJlkQZzOF3zj0LWHM%2BthQsgW1%2F3xBQpYnOxs6WsCXOB19rUqst55ORGLvjlxGYRaf2k8MnI0zKNKRUbeBFkZ2sSeuNVHKswgbOW3r5xipeEeuBrD%2FMZuXTN7rItaiXgeAliKEiFxnvOn74LoDpbpNwKswfyRMW0Wc7wfIeZO0Gqz3lFHli%2FejThUFj2%2FmLTPwkqchRRN4J4C5AIDpdM%2FYxxgqZntp%2FePZp79mwNTYP223SDjlwdhqzedEla93D%2B0CFT52PO9SXttyA9lJdsGiyxZvJx7r%2Bt2%2ByJYmA8mAMVYaetWvv51F7c%2B4NSJrUMU%2BRP5QpLTAq8hLuJ5rNMM%2F0yAcDVDJQRRFR0jWTu2cVp8mIaEtQGrmKTtc8zgtpp5vr%2FkJ1JLU4vJQlJgQP%2FWhUuBIN7qFWVUEQOhT4AS7COeIjFzZTSZG5H%2Fi4ZEsCae4mAOdCF%2BM%2BXumZ7PGcISxLJK%2B%2FcrYH9Fujo6RO%2BR7RbFe4HpTG1ACz73R7ALLs2t6d2sHXzVDb3hBe5V%2Fqa9ruEvX1YRBzr1Ch1FLB3KKed5YMGMioQnUF0AVFtKwSGvotv%2FtlhHlL3XfByLJ3bJLbvgoq70Wdyiu3AXffGxfk%3D&X-Amz-Signature=3a942192c50eb5cfaa5e6a519ecba04aec6644e0c602e033784edc1cbd4c253f&X-Amz-SignedHeaders=host'


def trim_enum_nb2(A):
    idx = (A != 0.0)
    return A[idx]


def untar_in_for(tar, t):
    tar.extract(t, path=DATADIR + '/extracted')


def load_loop(args, listdirs):
    if args.multiprocess:
        pool = mp.Pool(processes=args.processes)
        manager = mp.Manager()

        long_audio = manager.list()  # []
        mean_len = manager.list()  # []

        _, starts_at_s = timeStructured(False, True)
        [pool.apply_async(load_in_for, args=[d, long_audio, mean_len]) for d in listdirs]
        pool.close()
        pool.join()

        _, ends_at_s = timeStructured(False, True)
        duration_experiment = timedelta(seconds=ends_at_s - starts_at_s)
        print('Multiprocess took {}s'.format(duration_experiment))
    else:

        _, starts_at_s = timeStructured(False, True)
        long_audio = []
        pbar = tqdm(total=len(listdirs), desc='Load Numpys')
        mean_len = []
        for i, d in enumerate(listdirs):
            if i > args.n_audios: break
            datapath = os.path.join(CLIPS, d)

            try:
                src = aubio.source(datapath, hop_size=int(3e5))
                npaudio = src.do()
                npaudio = trim_enum_nb2(npaudio[0][None])

                # to make it 44100 Hz, since the dataset is saved as 22050 Hz, the signal is upsampled twice
                upsampled = npaudio.repeat(2, axis=0)
                long_audio.append(upsampled)

                mean_len.append(len(upsampled) / 22050)
            except Exception as e:
                print(e)

            if i % args.update_every == 0:
                pbar.update(args.update_every)
        pbar.close()
        _, ends_at_s = timeStructured(False, True)
        duration_experiment_2 = timedelta(seconds=ends_at_s - starts_at_s)
        print('Non-Multiprocess took {}s'.format(duration_experiment_2))

    return long_audio, mean_len


def load_in_for(d, long_audio, mean_len):
    datapath = os.path.join(CLIPS, d)

    try:
        src = aubio.source(datapath, hop_size=int(3e5))
        npaudio = src.do()
        npaudio = trim_enum_nb2(npaudio[0][None])

        # to make it 44100 Hz, since the dataset is saved as 22050 Hz, the signal is upsampled twice
        upsampled = npaudio.repeat(2, axis=0)
        long_audio.append(upsampled)

        mean_len.append(len(upsampled) / 22050)
    except Exception as e:
        print(e)


def main(args):
    if args.type == 'download':
        download_url(url, DATADIR + '/en.tar')

    if args.type in ['untar', 'untar_normalize']:
        pool = mp.Pool(processes=args.processes)

        if args.multiprocess:
            _, starts_at_s = timeStructured(False, True)
            with tarfile.open(TARPATH) as tar:
                [pool.apply_async(untar_in_for, args=[tar, t]) for t in tar]
            pool.close()
            pool.join()
            _, ends_at_s = timeStructured(False, True)
            duration_experiment = timedelta(seconds=ends_at_s - starts_at_s)
            print('Multiprocess took {}s'.format(duration_experiment))
        else:
            _, starts_at_s = timeStructured(False, True)
            with tarfile.open(TARPATH) as tar:
                pbar = tqdm(total=args.n_audios + 6, desc='Untar')
                for i, t in tqdm(enumerate(tar)):
                    if i > args.n_audios + 6: break
                    tar.extract(t, path=DATADIR + '/extracted')
                    if i % args.update_every == 0:
                        pbar.update(args.update_every)
                pbar.close()
            _, ends_at_s = timeStructured(False, True)
            duration_experiment_2 = timedelta(seconds=ends_at_s - starts_at_s)
            print('Non-Multiprocess took {}s'.format(duration_experiment_2))

    if args.type in ['normalize', 'untar_normalize']:

        max_samples_in_memory = 5000
        target_sample_rate = 44100
        target_n_samples = target_sample_rate * args.target_seconds

        sets = ['train', 'test', 'val']
        h5_files = {}
        fs = []
        for s in sets:
            filename = H5S + '/voices_{}.h5'.format(s)
            f = tables.open_file(filename, mode='w')
            array_c = f.create_earray(f.root, 'data', tables.Float64Atom(), (0, target_n_samples))
            h5_files[s] = array_c
            fs.append(f)

        n_loops = int(args.n_audios / max_samples_in_memory)

        pbar = tqdm(total=n_loops, desc='Load Numpys and h5 append')
        for idx in range(n_loops):
            try:
                print('H5 saving loop iteration: ', idx)
                listdirs = os.listdir(CLIPS)[idx * max_samples_in_memory:(idx + 1) * max_samples_in_memory]
                long_audio, mean_len = load_loop(args, listdirs)

                print('  concatenating...')
                # print('Mean audio length of {} audios: {}s'.format(len(long_audio), np.round(np.mean(mean_len)), 1))
                long_audio = np.concatenate(long_audio)

                print('  resizing...')
                # turn concatenated audios into a batch
                total_length = len(long_audio)
                final_length = total_length - total_length % target_n_samples
                long_audio = long_audio[:final_length]
                long_audio = long_audio.reshape((-1, target_n_samples))

                print('  splitting...')
                X_train, X_test = train_test_split(long_audio, test_size=0.10, random_state=42)
                X_train, X_val, = train_test_split(X_train, test_size=0.1111, random_state=1)  # 0.25 x 0.8 = 0.2
                X_train, X_test, X_val = X_train / np.max(X_train), X_test / np.max(X_train), X_val / np.max(X_train)
                print(X_train.shape, X_test.shape, X_val.shape)

                print('  to h5...')
                for k, v in zip(['train', 'test', 'val'], [X_train, X_test, X_val]):
                    # print('{}.shape:  '.format(k), v.shape)
                    h5_files[k].append(v)

                del long_audio, mean_len, X_train, X_test, X_val
            except Exception as e:
                print(e)

            pbar.update(1)
        pbar.close()

        for f in fs:
            f.close()

        for s in sets:
            filename = H5S + '/voices_{}.h5'.format(s)
            with h5py.File(filename, "r") as f:
                # List all groups
                a_group_key = list(f.keys())[0]

                # Get the data
                data = f[a_group_key]
                print('{}.shape:  '.format(s), data.shape)

    if args.type == 'test_h5':

        sets = ['train', 'test', 'val']
        for s in sets:
            filename = H5S + '/voices_{}.h5'.format(s)
            with h5py.File(filename, "r") as f:
                # List all groups
                a_group_key = list(f.keys())[0]

                # Get the data
                data = f[a_group_key]
                print('{}.shape:  '.format(s), data.shape)
                print(data[2:])


def parse_arguments():
    parser = argparse.ArgumentParser(description='voices pretraining')
    parser.add_argument('--type', default='normalize', type=str, help='main behavior',
                        choices=['download', 'normalize', 'untar', 'untar_normalize'])
    parser.add_argument('--target_seconds', default=2, type=int, help='time_steps in the batch')
    parser.add_argument('--n_audios', default=100, type=int, help='number of audios to unzip')
    parser.add_argument('--update_every', default=100, type=int, help='number of audios to unzip')
    parser.add_argument('--processes', default=8, type=int, help='number of audios to unzip')
    parser.add_argument('--multiprocess', action='store_true')
    args = parser.parse_args()

    print(args)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    print('DONE!')
