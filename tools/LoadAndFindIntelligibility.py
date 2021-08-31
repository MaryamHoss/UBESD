# this will load noisy, clean and different predictions

import os, sys
import h5py as hp
import matplotlib.pyplot as plt

sys.path.append('../')

import tensorflow as tf
import numpy as np

sound_len = 87552  # 218880
from TrialsOfNeuralVocalRecon.tools.calculate_intelligibility import find_intel
from TrialsOfNeuralVocalRecon.tools.utils.losses import *

#load noisy file
#test_path= 'C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/'
exp_folder='D:/data/1/images/prediction_31.npy'
#exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/3-both speakers/60-epochs/'#'Linear/'
#exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/3-both speakers/equal speakers/'#'Linear/'
exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/original_EEG/3-speaker specific/'
'''
prediction_31=np.load(exp_folder)
'''

exp_folder='D:/data/results_22-12/'
date='2020-12-22--16-29-40--mcp_'#'2020-12-21--13-48-14--mcp_' #2020-12-21--12-39-48--mcp_

#test_path='D:/data/EEG processed data/luca way'
#test_path='D:/data/EEG processed data_FBC/2_sec/equal'
test_path='D:/data/EEG processed data_FBC/2_sec'
test_path='D:/data/EEG_original data/2_sec'
test_path='D:/data/EEG processed data_FBC/2_sec/journey'
test_path='D:/data/EEG_original data/2_sec/journey'


#test_path='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/data/Cocktail_Party/Normalized/2s/fbc'
#exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/6/'
#exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/9-luca data/2020-09-15/'
file=hp.File(test_path+'/noisy_test.h5','r')
snd=file['noisy_test'][:]
file.close()

snd=snd[:,0:sound_len,:]
snd=snd[:,::3,:]

file=hp.File(test_path+'/eegs_test.h5','r')
eeg=file['eegs_test'][:]
file.close()
#load clean sound
file=hp.File(test_path+'/clean_test.h5','r')
clean=file['clean_test'][:]
file.close()
clean=clean[:,0:sound_len,:]
clean=clean[:,::3,:]
file.close()
#### do the prediction:
With_spike=exp_folder+'film1/trained_models/model_weights_WithSpikes_predict.h5'                #model_weights_WithSpikes_predict.h5'
model=tf.keras.models.load_model(With_spike, custom_objects={'si_sdr_loss': si_sdr_loss})
prediction_withSpikes_film1=model.predict([snd,eeg])

np.save(exp_folder+'film1/prediction',prediction_withSpikes_film1)


With_spike=exp_folder+date+'/trained_models/model_weights_WithSpikes_predict.h5' 
               #model_weights_WithSpikes_predict.h5'
from GenericTools.TFTools.AdaBelief import AdaBelief
from GenericTools.TFTools.convenience_operations import snake
model=tf.keras.models.load_model(With_spike, custom_objects={'si_sdr_loss': si_sdr_loss, 'AdaBelief':AdaBelief, 'snake':snake})
model=tf.keras.models.load_model(With_spike, custom_objects={'si_sdr_loss': si_sdr_loss})

#intel_matrix=np.zeros(shape=(833,4)) #0:pesq 1:stoi 2:estoi 3:si-sdr
#load prediction with spikes
exp_type='film1'
size=prediction_withSpikes_film1.shape[0]
intel_matrix_film1=np.zeros(shape=(size,5)) #0:pesq 1:stoi 2:estoi 3:si-sdr 4:seg_SNR
prediction_withSpikes_film1=np.load(exp_folder+exp_type+'/prediction.npy')
intel_matrix_film1=np.load(exp_folder+exp_type+'/loss_matrix.npy')
for i in range(size):
    print(i)
    intel_matrix_film1[i,0]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film1[i:i+1,:,:])

for i in range(size):
    print(i)
    intel_matrix_film1[i,1]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film1[i:i+1,:,:],metric='stoi')
    
for i in range(size):
    print(i)
    intel_matrix_film1[i,2]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film1[i:i+1,:,:],metric='estoi')
    
for i in range(size):
    print(i)
    intel_matrix_film1[i,3]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film1[i:i+1,:,:],metric='si-sdr')

for i in range(size):
    print(i)
    intel_matrix_film1[i,4]=find_intel(clean[i:i+1,:,:]/(np.max(np.abs(clean[i:i+1,:,:]))),prediction_withSpikes_film1[i:i+1,:,:](np.max(np.abs(prediction[i:i+1,:,:]))),metric='seg_SNR')

# for i in range(size):
#     print(i)
#     c[i,0]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film1[i:i+1,:,:],metric='stsa-mse')
     
np.save(exp_folder+'film1/loss_matrix',intel_matrix_film1)

#load prediction with spikes
exp_type='film2'
size=prediction_withSpikes_film2.shape[0]

intel_matrix_film2=np.zeros(shape=(size,4)) #0:pesq 1:stoi 2:estoi 3:si-sdr

prediction_withSpikes_film2=np.load(exp_folder+exp_type+'/prediction.npy')
intel_matrix_film2=np.load(exp_folder+exp_type+'/loss_matrix.npy')

for i in range(size):
    print(i)
    intel_matrix_film2[i,0]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film2[i:i+1,:,:])
    
for i in range(size):
    print(i)
    intel_matrix_film2[i,1]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film2[i:i+1,:,:],metric='stoi')
    
for i in range(size):
    print(i)
    intel_matrix_film2[i,2]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film2[i:i+1,:,:],metric='estoi')
    
for i in range(size):
    print(i)
    intel_matrix_film2[i,3]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film2[i:i+1,:,:],metric='si-sdr')
    
# for i in range(size):
#     print(i)
#     c[i,0]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film2[i:i+1,:,:],metric='stsa-mse')
 
    
np.save(exp_folder+'film2/loss_matrix',intel_matrix_film2)

#load prediction with spikes
exp_type='film3'
size=prediction_withSpikes_film3.shape[0]

intel_matrix_film3=np.zeros(shape=(size,4)) #0:pesq 1:stoi 2:estoi 3:si-sdr

prediction_withSpikes_film3=np.load(exp_folder+exp_type+'/prediction.npy')
intel_matrix_film3=np.load(exp_folder+exp_type+'/loss_matrix.npy')

for i in range(size):
    print(i)
    intel_matrix_film3[i,0]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film3[i:i+1,:,:])
    
for i in range(size):
    print(i)
    intel_matrix_film3[i,1]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film3[i:i+1,:,:],metric='stoi')
    
for i in range(size):
    print(i)
    intel_matrix_film3[i,2]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film3[i:i+1,:,:],metric='estoi')
    
for i in range(size):
    print(i)
    intel_matrix_film3[i,3]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film3[i:i+1,:,:],metric='si-sdr')
np.save(exp_folder+'film3/loss_matrix',intel_matrix_film3)
#load prediction with spikes
exp_type='film4'
size=prediction_withSpikes_film4.shape[0]

intel_matrix_film4=np.zeros(shape=(size,4)) #0:pesq 1:stoi 2:estoi 3:si-sdr

prediction_withSpikes_film4=np.load(exp_folder+exp_type+'/prediction.npy')
intel_matrix_film4=np.load(exp_folder+exp_type+'/loss_matrix.npy')

for i in range(size):
    print(i)
    intel_matrix_film4[i,0]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film4[i:i+1,:,:])
    
for i in range(size):
    print(i)
    intel_matrix_film4[i,1]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film4[i:i+1,:,:],metric='stoi')
    
for i in range(size):
    print(i)
    intel_matrix_film4[i,2]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film4[i:i+1,:,:],metric='estoi')
    
for i in range(size):
    print(i)
    intel_matrix_film4[i,3]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film4[i:i+1,:,:],metric='si-sdr')
np.save(exp_folder+'film4/loss_matrix',intel_matrix_film4)
#load prediction with spikes
exp_type='add'
intel_matrix_add=np.zeros(shape=(833,4)) #0:pesq 1:stoi 2:estoi 3:si-sdr

prediction_withSpikes_add=np.load(exp_folder+exp_type+'/prediction.npy')
intel_matrix_add=np.load(exp_folder+exp_type+'/loss_matrix.npy')

for i in range(833):
    print(i)
    intel_matrix_add[i,0]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_add[i:i+1,:,:])
    
for i in range(833):
   print(i)
   intel_matrix_add[i,1]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_add[i:i+1,:,:],metric='stoi')
    
for i in range(833):
    print(i)
    intel_matrix_add[i,2]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_add[i:i+1,:,:],metric='estoi')
    
for i in range(833):
    print(i)
    intel_matrix_add[i,3]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_add[i:i+1,:,:],metric='si-sdr')
#load prediction with spikes
exp_type='choice'
intel_matrix_choice=np.zeros(shape=(833,4)) #0:pesq 1:stoi 2:estoi 3:si-sdr

prediction_withSpikes_choice=np.load(exp_folder+exp_type+'/prediction.npy')
intel_matrix_choice=np.load(exp_folder+exp_type+'/loss_matrix.npy')

for i in range(833):
    print(i)
    intel_matrix_choice[i,0]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_choice[i:i+1,:,:])
    
for i in range(833):
    print(i)
    intel_matrix_choice[i,1]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_choice[i:i+1,:,:],metric='stoi')
    
for i in range(833):
    print(i)
    intel_matrix_choice[i,2]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_choice[i:i+1,:,:],metric='estoi')
    
for i in range(833):
    print(i)
    intel_matrix_choice[i,3]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_choice[i:i+1,:,:],metric='si-sdr')

#load prediction with spikes
exp_type='concatenate'
intel_matrix_concatenate=np.zeros(shape=(833,4)) #0:pesq 1:stoi 2:estoi 3:si-sdr

prediction_withSpikes_concatenate=np.load(exp_folder+exp_type+'/prediction.npy')
intel_matrix_concatenate=np.load(exp_folder+exp_type+'/loss_matrix.npy')

for i in range(833):
    print(i)
    intel_matrix_concatenate[i,0]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_concatenate[i:i+1,:,:])
    
for i in range(833):
    print(i)
    intel_matrix_concatenate[i,1]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_concatenate[i:i+1,:,:],metric='stoi')
    
for i in range(833):
    print(i)
    intel_matrix_concatenate[i,2]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_concatenate[i:i+1,:,:],metric='estoi')
    
for i in range(833):
    print(i)
    intel_matrix_concatenate[i,3]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_concatenate[i:i+1,:,:],metric='si-sdr')

#no spikes
exp_type='noSpike'
size=prediction_noSpike.shape[0]

intel_matrix_noSpike=np.zeros(shape=(size,4)) #0:pesq 1:stoi 2:estoi 3:si-sdr

prediction_noSpike=np.load(exp_folder+exp_type+'/prediction.npy')
intel_matrix_noSpike=np.load(exp_folder+exp_type+'/loss_matrix.npy')
    
for i in range(size):
    print(i)
    intel_matrix_noSpike[i,0]=find_intel(clean[i:i+1,:,:],prediction_noSpike[i:i+1,:,:])
    
for i in range(size):
    print(i)
    intel_matrix_noSpike[i,1]=find_intel(clean[i:i+1,:,:],prediction_noSpike[i:i+1,:,:],metric='stoi')
    
for i in range(size):
    print(i)
    intel_matrix_noSpike[i,2]=find_intel(clean[i:i+1,:,:],prediction_noSpike[i:i+1,:,:],metric='estoi')
    
for i in range(size):
    print(i)
    intel_matrix_noSpike[i,3]=find_intel(clean[i:i+1,:,:],prediction_noSpike[i:i+1,:,:],metric='si-sdr')
np.save(exp_folder+exp_type+'loss_matrix',intel_matrix_noSpike)
    
import h5py as hp    
    
exp_type='noisy'
intel_matrix_noisy=np.zeros(shape=(size,5)) #0:pesq 1:stoi 2:estoi 3:si-sdr
file=hp.File(test_path+'/noisy_test.h5','r')
snd=file['noisy_test'][:]
file.close()

snd=snd[:,0:sound_len,:]
snd=snd[:,::3,:]

size=snd.shape[0]
intel_matrix_noisy=np.zeros(shape=(size,4)) #0:pesq 1:stoi 2:estoi 3:si-sdr

intel_matrix_noisy=np.load(exp_folder+'loss_matrix_noisy.npy')
for i in range(size):
    print(i)
    intel_matrix_noisy[i,0]=find_intel(clean[i:i+1,:,:],snd[i:i+1,:,:])
    
for i in range(size):
    print(i)
    intel_matrix_noisy[i,1]=find_intel(clean[i:i+1,:,:],snd[i:i+1,:,:],metric='stoi')
    
for i in range(size):
    print(i)
    intel_matrix_noisy[i,2]=find_intel(clean[i:i+1,:,:],snd[i:i+1,:,:],metric='estoi')
    
for i in range(size):
    print(i)
    intel_matrix_noisy[i,3]=find_intel(clean[i:i+1,:,:],snd[i:i+1,:,:],metric='si-sdr')

for i in range(size):
    print(i)
    intel_matrix_noisy[i,4]=find_intel(clean[i:i+1,:,:],snd[i:i+1,:,:],metric='seg_SNR')

    
np.save(exp_folder+'loss_matrix_noisy',intel_matrix_noisy)


exp_folder = '../data'
intel_matrix_noisy = np.load(exp_folder + '/loss_matrix_wo_eeg.npy')
intel_matrix = np.load(exp_folder + '/loss_matrix_with_eeg.npy')

fusions = ['film1', 'film2', 'film3', 'film4', 'add', 'concat', 'choice', 'noSp', 'noisy']
metrics = ['pesq', 'stoi', 'estoi', 'si-sdr']

fusions = ['mixture', 'auto', 'BESD']
metrics = ['PESQ', 'STOI', 'ESTOI', 'SI-SDR']

data = []

z =4
# data.append(intel_matrix_film1[:,z])
# data.append(intel_matrix_film2[:,z])
# data.append(intel_matrix_film3[:,z])
# data.append(intel_matrix_film4[:,z])
# data.append(intel_matrix_add[:,z])
# data.append(intel_matrix_concatenate[:,z])
# data.append(intel_matrix_choice[:,z])
# data.append(intel_matrix_noSpike[:,z])
data.append(intel_matrix_noisy[:, z])
data.append(intel_matrix_noSpike[:, z])

data.append(intel_matrix_film1[:, z])

ld = len(metrics)
lm = len(fusions)
width = 1 / lm - .05
X = np.arange(ld)

"""
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])

for i in range(lm):
    ax.bar(X + i * width, data[i], width=width)

ax.set_ylabel('intelligibility')
plt.xticks(X + lm * width / 2, metrics)

fusions = [f.replace('_', '') for f in fusions]
ax.legend(labels=fusions)
#plt.savefig(os.path.join(plot_one_path, 'plot_bars_accs.png'), bbox_inches="tight")
"""

num_boxes = len(data)
pos = np.arange(num_boxes) + 1
medians = np.zeros(shape=len(data))
for i in range(num_boxes):
    medians[i] = np.median(data[i])
upper_labels = [str(np.round(s, 2)) for s in medians]
weights = ['bold', 'semibold']

"""
fig1, ax1 = plt.subplots()
ax1.boxplot(data, labels=fusions)
ax1.set_title(metrics[z])
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
    k = tick % 2
    ax1.text(pos[tick], .95, upper_labels[tick],
             transform=ax1.get_xaxis_transform(),
             horizontalalignment='center', size='x-small',
             weight=weights[k])

plt.show()

"""

fig, ax = plt.subplots()


violin_handle = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, widths=.7)

colors = plt.cm.gist_ncar(np.linspace(0.2, .8, len(violin_handle['bodies'])))
np.random.seed(0)
colors = plt.cm.twilight(np.random.rand(len(violin_handle['bodies'])))
np.random.shuffle(colors)
for pc, c in zip(violin_handle['bodies'], colors):
    pc.set_facecolor(c)
    pc.set_edgecolor('black')
    pc.set_alpha(1.)

for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
    k = tick % 2
    ax.text(pos[tick], .95, upper_labels[tick],
             transform=ax.get_xaxis_transform(),
             horizontalalignment='center', size='x-small',
             weight=weights[k])

ax.set_ylabel(metrics[z])
ax.set_ylim(-20,20)
plt.xticks(np.arange(len(fusions)) + 1, fusions)

plt.show()

fig.savefig('cocktail_sisdr_violins.pdf', bbox_inches='tight')





from scipy.io import savemat
mdic = {"clean": clean}
savemat('clean.mat',mdic)
mdic = {"noisy": snd}
savemat('noisy.mat',mdic)
mdic = {"prediction": prediction_withSpikes_film1}
savemat('prediction.mat',mdic)

#### to load from matlab and make the violin plots
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/3-both speakers/60-epochs/'
exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/3-both speakers/equal speakers/'
exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/original_EEG/1-both_speaker/'
exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/original_EEG/3-speaker specific/'

exp_folder='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon/experiments/CC/sisdr/with my changes/4-speaker specific/'


exp_type='film1_correct'
exp_type='film1'
exp_type='noSpike'


fusions = ['Mixture', 'BESD \nEEG','BESD \nMUA']


fusions = ['mixture', 'BESD']
fusions = ['Mixture', 'Autoencoder', 'BESD \nEEG','BESD \nMUA']

metric=['stoi','ssnr','SDR']
file=loadmat(exp_folder+exp_type+'/'+ metric[0]+ '.mat')

file=loadmat(exp_folder+exp_type+'/'+ metric[1]+ '.mat')
file=loadmat(exp_folder+exp_type+'/'+ metric[2]+ '.mat')
file=loadmat(exp_folder+exp_type+'/SDR_noisy.mat')
SDR_pred_fbc
SDR_noisy
stoi_noisy=file['stoi_noisy'][:]

stoi_pred_fbc=file['stoi_pred'][:]
stoi_pred_ori=file['stoi_pred'][:]

stoi_pred_auto=file['stoi_pred'][:]


ssnr_noisy=file['ssnr_noisy'][:]
ssnr_pred_auto=file['ssnr_pred'][:]
ssnr_pred_ori=file['ssnr_pred'][:]
ssnr_pred_fbc=file['ssnr_pred'][:]

SDR_noisy=file['SDR_noisy'][:]
SDR_pred_auto=file['SDR_auto'][:]
SDR_pred_ori=file['SDR_pred_ori'][:]
SDR_pred_fbc=file['SDR_pred_fbc'][:]



metrics = ['SI-SDR','STOI',  'segSNR']

data = []
z=2

data.append(sorted(stoi_noisy[:,0]))
data.append(sorted(stoi_pred_auto[:,0]))
data.append(sorted(stoi_pred_ori[:,0]))
data.append(sorted(stoi_pred_fbc[:,0]))



data.append(sorted(SDR_noisy[0]))
data.append(sorted(SDR_pred_auto[0]))
data.append(sorted(SDR_pred_ori[0]))
data.append(sorted(SDR_pred_fbc[0]))

data.append(sorted(SDR_noisy[0]))
data.append(sorted(SDR_pred_ori[0]))
data.append(sorted(SDR_pred_fbc[0]))



data.append(sorted(ssnr_noisy[:,0]))
data.append(sorted(ssnr_pred_auto[:,0]))

data.append(sorted(ssnr_pred_ori[:,0]))

data.append(sorted(ssnr_pred_fbc[:,0]))


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value



fig, ax = plt.subplots()
violin_handle = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, widths=.7)


colors = plt.cm.gist_ncar(np.linspace(0.2, .8, len(violin_handle['bodies'])))
np.random.seed(0)
colors = plt.cm.twilight(np.random.rand(len(violin_handle['bodies'])))
np.random.shuffle(colors)
for pc, c in zip(violin_handle['bodies'], colors):
    pc.set_facecolor(c)
    pc.set_edgecolor('black')
    pc.set_alpha(1.)
    
num_boxes = len(data)
pos = np.arange(num_boxes) + 1
medians = np.zeros(shape=len(data))
quartile1=np.zeros(shape=len(data))
quartile3=np.zeros(shape=len(data))
for i in range(num_boxes):
    quartile1[i], medians[i], quartile3[i] = np.percentile(data[i], [25, 50, 75])
    #medians[i] = np.median(data[i])
"""    
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

"""

    
    
    
upper_labels = [str(np.round(s, 2)) for s in medians]
weights = ['bold', 'semibold']



"""
inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

"""


for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
    k = tick % 2
    ax.text(pos[tick], .9, upper_labels[tick],
             transform=ax.get_xaxis_transform(),
             horizontalalignment='center', fontsize=12,
             weight=weights[k])

ax.set_ylabel(metrics[z],fontsize=15)
ax.set_ylim(-10,16)
plt.yticks([-10,0,10,15],fontsize=15)
#ax.set_xticks()
plt.xticks(np.arange(len(fusions)) + 1, fusions,fontsize=12)

plt.show()

fig.savefig('speaker_specific_stoi.eps', format='eps', bbox_inches='tight')
fig.savefig('speaker_specific_segSNR.eps', format='eps', bbox_inches='tight')
fig.savefig('speaker_specific_SDR.eps', format='eps', bbox_inches='tight')

fig.savefig('speaker_independent_SDR.eps', format='eps', bbox_inches='tight')
fig.savefig('speaker_independent_stoi.eps', format='eps', bbox_inches='tight')
fig.savefig('speaker_independent_segSNR.eps', format='eps', bbox_inches='tight')

from scipy.stats import ranksums

ranksums(stoi_noisy[:,0],stoi_pred[:,0])

ranksums(ssnr_noisy[:,0],ssnr_pred[:,0])

ranksums(stoi_noisy[:,0],stoi_pred_ori[:,0])
ranksums(stoi_noisy[:,0],stoi_pred_fbc[:,0])
ranksums(stoi_pred_ori[:,0],stoi_pred_fbc[:,0])

ranksums(stoi_pred_auto[:,0],stoi_pred_fbc[:,0])





##################################################

import os, sys
import h5py as hp
import matplotlib.pyplot as plt

sys.path.append('../')

import tensorflow as tf
import numpy as np

sound_len = 87552  # 218880
from TrialsOfNeuralVocalRecon.tools.calculate_intelligibility import find_intel
from TrialsOfNeuralVocalRecon.tools.utils.losses import *


exp_folder='D:/data/experiments/13-1/'
date='2021-01-13--11-50-56--mcp_/'


test_path=exp_folder


file=hp.File(test_path+'/noisy_test.h5','r')
snd=file['noisy_test'][:]
file.close()

snd=snd[:,0:sound_len,:]
snd=snd[:,::3,:]

file=hp.File(test_path+'/eegs_test.h5','r')
eeg=file['eegs_test'][:]
file.close()
#load clean sound
file=hp.File(test_path+'/clean_test.h5','r')
clean=file['clean_test'][:]
file.close()
clean=clean[:,0:sound_len,:]
clean=clean[:,::3,:]
file.close()

from scipy.io import savemat
mdic = {"clean": clean}
savemat('clean.mat',mdic)
mdic = {"noisy": snd}
savemat('noisy.mat',mdic)
mdic = {"prediction": prediction_withSpikes_film1}
savemat('prediction.mat',mdic)


#### do the prediction:
With_spike=exp_folder+date+'trained_models/model_weights_WithSpikes_predict.h5' 

model=tf.keras.models.load_model(With_spike, custom_objects={'si_sdr_loss': si_sdr_loss})

from GenericTools.TFTools.AdaBelief import AdaBelief
from GenericTools.TFTools.convenience_operations import snake
model=tf.keras.models.load_model(With_spike, custom_objects={'si_sdr_loss': si_sdr_loss,'snake':snake})
model=tf.keras.models.load_model(With_spike, custom_objects={'si_sdr_loss': si_sdr_loss,'AdaBelief':AdaBelief})

eeg_part=eeg[0:1000,:,:]
eeg_part=np.repeat(eeg_part,114,1)
snd_part=snd[0:1000,:,:]
prediction_withSpikes_film1_1=model.predict([snd_part,eeg_part])


prediction_withSpikes_film1=np.concatenate((prediction_withSpikes_film1_1,prediction_withSpikes_film1_2,prediction_withSpikes_film1_3,prediction_withSpikes_film1_4),axis=0)

np.save(exp_folder+date+'prediction',prediction_withSpikes_film1)


size=prediction_withSpikes_film1.shape[0]
intel_matrix_film1=np.zeros(shape=(size,1)) #0:pesq 1:stoi 2:estoi 3:si-sdr 4:seg_SNR
for i in range(size):
    print(i)
    intel_matrix_film1[i,0]=find_intel(clean[i:i+1,:,:],prediction_withSpikes_film1[i:i+1,:,:],metric='si-sdr')
    
np.median(intel_matrix_film1) 