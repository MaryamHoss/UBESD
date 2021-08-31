import os
import requests
path_s='C:/Users/hoss3301/work/snhl/data/stimuli/'
path_dl='https://resources.drcmr.dk/uheal/data_without_audio/stimuli/'

def exists(path,auth=('paper','paper')):
    r = requests.head(path,auth=auth)
    return r.status_code == requests.codes.ok




for s in ['masker']:
    for subjects in range(33,34):
        print('Downloading data from subject {}'.format(subjects))
        for i in range(1,49):
            print('Audio number: {}'.format(i))
            if len(str(i))==1:
                sub=str(subjects)
                song=str(i)
                path_download=path_dl+'sub0'+sub+'/'+s+'/m00'+song+'.wav'
                path_save=path_s+'sub0'+sub+'/'+s+'/m00'+song+'.wav'
                save_folder=path_s+'sub0'+sub+'/'+s
            else:
                sub=str(subjects)
                song=str(i)
                path_download=path_dl+'sub0'+sub+'/'+s+'/m0'+song+'.wav'
                path_save=path_s+'sub0'+sub+'/'+s+'/m0'+song+'.wav'
                save_folder=path_s+'sub0'+sub+'/'+s
                
            try:
                if exists(path_download,auth=('paper', 'paper')):
                    doc = requests.get(path_download,auth=('paper', 'paper'))
                                                
                    if not os.path.isdir(save_folder):
                        os.mkdir(save_folder)
                    
                    f = open(path_save,"wb")
                    f.write(doc.content)
                    f.close()
                else:
                    print('This file does not exist: audio {} of the {}'.format(i, subjects))
                    
            except:
                print('This file does not exist: audio {} of the {}'.format(i, subjects))