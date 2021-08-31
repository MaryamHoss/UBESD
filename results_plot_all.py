

import pandas as pd
import numpy as np

def get_metric_matrix(paths, metric='stoi',get_noisy='stoi_noisy'):
    metric_matrix=[]
    for path_number,path_item in enumerate(paths):
        df=pd.read_csv(path_item)
        metric_matrix.append(df[[metric]])
    if not get_noisy is False:
        metric_matrix.append(df[[get_noisy]])

        
    metric_matrix_concat=np.concatenate(metric_matrix,axis=1)
    return metric_matrix_concat
def plot_intel_comp(paths,metric='stoi',get_noisy='stoi_noisy'):
    
    metric_matrix_concat=
            
        
    
    
        