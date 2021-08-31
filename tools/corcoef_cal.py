import numpy as np
import os
import sys
sys.path.append('../')
sys.path.append('../..')


#for this to work, cd in the folder containing this code
DATADIR = os.path.curdir + r'/../data'
Noise_type='NS_'
Level='55'
SNR='5_'
experiment='NoSpike'



def test():

    #batch_size = 32
    #time_steps = 30000 #2100
    #features = 1
    corcoef=np.zeros((1,768)) #the matrix at the end, having the three losses in it
    #np_true = np.random.rand(batch_size, time_steps, features).astype(np.float32)
    #np_pred = np.random.rand(batch_size, time_steps, features).astype(np.float32)
    content = [d for d in os.listdir(DATADIR) if 'npy' in d]
    print(content)
    path_true=DATADIR +'/4layers/'+experiment+ '/clean_'+Level+'.npy'
    np_true_1 = np.load(path_true)[:].astype(np.float32)
    np_true=np.zeros(shape=(np_true_1.shape[0]*np_true_1.shape[1],np_true_1.shape[2],1))
    np_true=np.reshape(np_true_1,(np_true_1.shape[0]*np_true_1.shape[1],np_true_1.shape[2],1))
    np_true=np_true[:,1:,:]
    #np_pred = np.load(DATADIR + '/prediction_NoSpike.npy')[36+36*z:37+36*z, :, np.newaxis].astype(np.float32)
    path_pred=DATADIR + '/4layers/'+experiment+ '/prediction_'+Noise_type+SNR+Level+'.npy'
    np_pred = np.load(path_pred)[:, :,:].astype(np.float32)
    print('real data shapes: ', np_true.shape, np_pred.shape)
    
    for batch in range(np_true.shape[0]):
        print(batch)
        np_true_batch=np_true[batch,:,:] #for the code to run for each example we have separately
                #so at the end we can take a mean over all the estoi metrics
        np_pred_batch=np_pred[batch,:,:]
        
        cormatrix=np.corrcoef(np_true_batch,np_pred_batch,rowvar=False)
        corcoef[:,batch]=cormatrix[0,1]
        if np.isnan(corcoef[:,batch])==True:
            corcoef[:,batch]=0
        

    sum_corcoef=np.sum(corcoef)
    corcoef_score= sum_corcoef/((corcoef.shape[1]*2)/3)
    corcoef_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
    np.save(corcoef_save_path,corcoef_score)

    
if __name__ == '__main__':
    test()
    
    
    
    
    
    
    
    
##no spike
import pandas as pd
import os
import numpy as np

DATADIR = os.path.curdir + r'/../data'
Noise_type='wh_'
Level='65'
SNR='5_'
experiment='NoSpike'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_wh_5_65_NoSpike=np.load(Estoi_save_path)

Noise_type='wh_'
Level='65'
SNR='15_'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_wh_15_65_NoSpike=np.load(Estoi_save_path)

Noise_type='wh_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_wh_15_55_NoSpike=np.load(Estoi_save_path)


Noise_type='wh_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_wh_5_55_NoSpike=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_Na_5_55_NoSpike=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_Na_15_55_NoSpike=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_Na_15_65_NoSpike=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_Na_5_65_NoSpike=np.load(Estoi_save_path)



Noise_type='NS_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_NS_5_55_NoSpike=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_NS_15_55_NoSpike=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_NS_15_65_NoSpike=np.load(Estoi_save_path)


Noise_type='NS_'
SNR='5_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_NS_5_65_NoSpike=np.load(Estoi_save_path)





##with spikes

DATADIR = os.path.curdir + r'/../data'
Noise_type='wh_'
Level='65'
SNR='5_'
experiment='WithSpike'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_wh_5_65=np.load(Estoi_save_path)

Noise_type='wh_'
Level='65'
SNR='15_'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_wh_15_65=np.load(Estoi_save_path)

Noise_type='wh_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_wh_15_55=np.load(Estoi_save_path)


Noise_type='wh_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_wh_5_55=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_Na_5_55=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_Na_15_55=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_Na_15_65=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_Na_5_65=np.load(Estoi_save_path)



Noise_type='NS_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_NS_5_55=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_NS_15_55=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_NS_15_65=np.load(Estoi_save_path)


Noise_type='NS_'
SNR='5_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'corcoef_'+Noise_type+SNR+Level+'.npy'
estoi_NS_5_65=np.load(Estoi_save_path)



models_names = ['convolutional']
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

data={metrics_w[0]:estoi_NS_5_55,metrics_w[1]:estoi_wh_5_55,metrics_w[2]:estoi_Na_5_55,
      metrics_w[3]:estoi_NS_15_55,metrics_w[4]:estoi_wh_15_55,metrics_w[5]:estoi_Na_15_55,
      metrics_w[6]:estoi_NS_5_65,metrics_w[7]:estoi_wh_5_65,metrics_w[8]:estoi_Na_5_65,
      metrics_w[9]:estoi_NS_15_65,metrics_w[10]:estoi_wh_15_65,metrics_w[11]:estoi_Na_15_65,
      metrics_wo[0]:estoi_NS_5_55_NoSpike, metrics_wo[1]:estoi_wh_5_55_NoSpike,  
      metrics_wo[2]:estoi_Na_5_55_NoSpike,
      metrics_wo[3]:estoi_NS_15_55_NoSpike,metrics_wo[4]:estoi_wh_15_55_NoSpike, 
      metrics_wo[5]:estoi_Na_15_55_NoSpike,
      metrics_wo[6]:estoi_NS_5_65_NoSpike, metrics_wo[7]:estoi_wh_5_65_NoSpike,  
      metrics_wo[8]:estoi_Na_5_65_NoSpike,
      metrics_wo[9]:estoi_NS_15_65_NoSpike,metrics_wo[10]:estoi_wh_15_65_NoSpike,
      metrics_wo[11]:estoi_Na_15_65_NoSpike}
df=pd.DataFrame(data,columns=metrics,index=models_names)
import matplotlib.pyplot as plt

plot_bar(metrics_w,metrics_wo,df)

def plot_bar(metrics_w,metrics_wo,df):
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
    