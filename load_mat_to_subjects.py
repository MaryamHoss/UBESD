


import numpy as np
import scipy.io as sio
import pickle
import pandas as pd
import os, sys, shutil, json, time, copy
sys.path.append('../')

from TrialsOfNeuralVocalRecon.tools.plotting import save_wav, one_plot_test, evaluations_to_violins



CDIR = os.path.dirname(os.path.realpath(__file__))
# CDIR='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon'



exp_folder='2'
EXPERIMENTS = os.path.join(*[CDIR, 'experiments'])
Other=os.path.join(*[EXPERIMENTS, exp_folder,'other_outputs','all_subjects_evaluations_test.pkl'])

a_file = open(Other, "rb")
all_subjects_evaluations=pickle.load(a_file)
a_file.close()

path_OPS=os.path.join(*[EXPERIMENTS,exp_folder,'OPS_Results.mat'])
path_subjects= os.path.join(*[EXPERIMENTS,exp_folder,'subjects.mat'])
OPS_matlab=sio.loadmat(path_OPS)
subjects_matlab=sio.loadmat(path_subjects)
OPS_vec=OPS_matlab['OPS'][:]
subjects_vec=subjects_matlab['subjects'][:]
subjects_vec=np.transpose(subjects_vec)
subjects_vec=subjects_vec.astype('int')

OPS_subjects={}
for i in range(np.size(OPS_vec)):    
    key='Subject {}'.format(subjects_vec[i][0])
    OPS_subjects.setdefault(key,[]).append(OPS_vec[i][0])
    

for key in OPS_subjects:
    all_subjects_evaluations[key]['OPS']=OPS_subjects[key]
    

path=os.path.join(*[CDIR, 'other_outputs','all_subjects_evaluations_test_OPS.pkl'])
os.makedirs(os.path.dirname(path), exist_ok=True)
a_file = open(path, "wb")
pickle.dump(all_subjects_evaluations, a_file)
a_file.close()

images_dir=os.path.join(*[EXPERIMENTS, exp_folder])


prediction_metrics = ['OPS']
generator_type='test'

prediction_metrics = ['si-sdr', 'stoi', 'estoi' , 'pesq']
noisy_metrics = [m + '_noisy' for m in prediction_metrics]
generator_type='test'


evaluations_to_violins({k: v[prediction_metrics] for k, v in all_subjects_evaluations.items()}, images_dir,
                       generator_type + '')

 
    