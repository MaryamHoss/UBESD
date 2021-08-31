import os

import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from GenericTools.KerasTools.convenience_tools import plot_history
from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment
from GenericTools.StayOrganizedTools.utils import email_results
#from TrialsOfNeuralVocalRecon.data_processing.visualization_tools import plot_attention


from TrialsOfNeuralVocalRecon.neural_models.model_helpers import timeStructured, \
    build_autoencoder_convolutional_voltage_gated_spectrogram

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from TrialsOfNeuralVocalRecon.data_processing.convenience_tools import getData
#from TrialsOfNeuralVocalRecon.neural_tools.
# from preprocessing.tools import TensorBoardWrapper
#from keras.backend.tensorflow_backend import set_session
#GPU = 0
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
#config = tf.compat.v1.ConfigProto()  # tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
#config.gpu_options.allow_growth = True
#set_session(tf.compat.v1.Session(config=config))
#sess = tf.compat.v1.Session(config=config)  # tf.Session(config=config)


CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment('guinea_denoiser_spectrogram', base_dir=CDIR, GPU=1, seed=14)


@ex.config
def cfg():
    activation_encode = 'relu'
    activation_spikes = 'relu'
    activation_decode = 'relu'
    activation_all = 'tanh'
    learning_rate = 1e-5
    n_convolutions = 3


    epochs = 1#3
    steps = 1#3900
    data_type =  'spectrogram_prediction'#,denoising_spectrogram,
    #spectrogram_prediction_NoSpike,spectrogram_denoising_NoSpike,
    
    batch_size = 32
    exp_type="noSpikes" #WithSpikes
    sound_len=95 
    spike_len=95
    n_channels=128

@ex.automain
def main(activation_encode, activation_spikes, activation_decode, activation_all, learning_rate, n_convolutions, 
         epochs, steps, data_type, batch_size,exp_type,sound_len, spike_len,n_channels):
    
    print('model_type='+data_type+'_'+exp_type)
    
    model = build_autoencoder_convolutional_voltage_gated_spectrogram(activation_encode=activation_encode,
                                                          activation_spikes=activation_spikes,
                                                          activation_decode=activation_decode,
                                                          activation_all=activation_all,
                                                          learning_rate=learning_rate,
                                                          n_convolutions=n_convolutions)
    
     #model = build_autoencoder_convolutional_voltage_gated_noSpike_spectrogram(activation_encode=activation_encode,
      #                                                    activation_spikes=activation_spikes,
     #                                                     activation_decode=activation_decode,
      #                                                    activation_all=activation_all,
      #                                                    learning_rate=learning_rate,
       #                                                   n_convolutions=n_convolutions)
     
     
    sacred_dir = os.path.join(*[CDIR, ex.observers[0].basedir, '1'])
    #plot_attention(model=model, data_type=data_type, plot_name='attention_bt', save_dir=sacred_dir)
    model.summary()

    callbacks = []  # [tensorboard]

    ##############################################################
    #                    train
    ##############################################################
    if exp_type == "WithSpikes":


        generator_train, generator_test = getData(
                                      sound_shape=(sound_len, n_channels,1),
                                      spike_shape=(spike_len, n_channels,1),
                                      data_type=data_type, 
                                      batch_size=batch_size)
        history = model.fit_generator(generator_train,
                                      epochs=epochs,
                                      validation_data=generator_test,
                                      use_multiprocessing=False,
                                      shuffle=False,
                                      verbose=1,
                                      workers=1,
                                      callbacks=callbacks)
        plot_filename = 'model/train_history_spectrogram.pdf'
        plot_history(history, plot_filename, epochs)
        ex.add_artifact(plot_filename)
        #plot_attention(model=model, data_type=data_type, plot_name='attention_at', save_dir=sacred_dir)
    
        # list_keys=[key for key in loss_hystory]
        # print(list_keys)
    
        print('fitting done,saving model')
        time_string = timeStructured()
        path_best_model = 'model/{}_model_spectrogram_withSpike_predict.h5'.format(time_string)
        model.save(path_best_model)
    
        ##############################################################
        #                    tests training
        ##############################################################
    
        [batch_snd_in_test, batch_spk_test], batch_snd_out_test = generator_test.__getitem__()
        prediction = model.predict([batch_snd_in_test, batch_spk_test])
        one_sound, one_sound_out, one_predicted_sound = batch_snd_in_test[0], batch_snd_out_test[0], prediction[0]
        fig, axs = plt.subplots(3)
        fig.suptitle('prediction plot vs inputs')

        axs[0].plot(one_sound[:,:,0])
        axs[0].set_title('input sound')
        axs[1].plot(one_predicted_sound[:,:,0])
        axs[1].set_title('predicted sound')
        axs[2].plot(one_sound_out[:,:,0])
        axs[2].set_title('output sound')   
        fig_path = './model/prediction_spectrogram.pdf'
        fig.savefig(fig_path, bbox_inches='tight')
        ex.add_artifact(fig_path)
    
        ########################################################################
        #                              finetune
        #########################################################################
    
        #tensorboard = TensorBoard(log_dir='./model/logs/{}_predict_finetuned'.format(time_string))
    
        callbacks = []  # [tensorboard]
        generator_train_noisy, generator_test_noisy = getData(
                                                        sound_shape=(sound_len, n_channels,1),
                                                        spike_shape=(spike_len, n_channels,1),
                                                        data_type='denoising_spectrogram', 
                                                        batch_size=batch_size)
        noisy_history = model.fit_generator(generator_train_noisy,
                                            epochs=epochs,
                                            validation_data=generator_test_noisy,
                                            use_multiprocessing=False,
                                            shuffle=False,
                                            verbose=1,
                                            workers=1,
                                            callbacks=callbacks)
        plot_filename = 'model/denoising_history_spectrogram.pdf'
        plot_history(noisy_history, plot_filename, epochs)
        ex.add_artifact(plot_filename)
        #plot_attention(model=model, data_type=data_type, plot_name='attention_af', save_dir=sacred_dir)
    
        print('finetuning done,saving model')
        path_best_model_finetune = 'model/{}_model_spectrogram_withSpike_finetuned.h5'.format(time_string)
        model.save(path_best_model_finetune)
    
        ##############################################################
        #                    tests finetuning
        ##############################################################
    
        [batch_snd_in_test_denoise, batch_spk_test_denoise], batch_snd_out_test_denoise = generator_test_noisy.__getitem__()
        prediction_denoise = model.predict([batch_snd_in_test_denoise, batch_spk_test_denoise])
        one_sound_noisy, one_sound_denoised, one_predicted_sound_denoised = batch_snd_in_test_denoise[0], \
                                                                         batch_snd_out_test_denoise[0],\
                                                                             prediction_denoise[0]
        axs[0].plot(one_sound_noisy[:,:,0])
        axs[0].set_title('input sound')
        axs[1].plot(one_predicted_sound_denoised[:,:,0])
        axs[1].set_title('predicted sound')
        axs[2].plot(one_sound_denoised[:,:,0])
        axs[2].set_title('output sound')   
    
        fig_path = './model/prediction_finetuned_spectrogram.pdf'
        fig.savefig(fig_path, bbox_inches='tight')
        ex.add_artifact(fig_path)
    
        sacred_dir = os.path.join(*[CDIR, ex.observers[0].basedir, '1'])
        email_results(
            folders_list=[sacred_dir],
            name_experiment=' guinea, multiplied attention ',
            receiver_emails=['manucelotti@gmail.com', 'm.hosseinite@gmail.com'])
        
        
        
        

    #################################################################
    #######################if no spikes
    ###########################################################        
        
        
        
    if exp_type == "noSpikes":


        generator_train, generator_test = getData(
                                      sound_shape=(sound_len, n_channels,1),
                                      spike_shape=(spike_len, n_channels,1),
                                      data_type=data_type, 
                                      batch_size=batch_size)
        history = model.fit_generator(generator_train,
                                      epochs=epochs,
                                      validation_data=generator_test,
                                      use_multiprocessing=False,
                                      shuffle=False,
                                      verbose=1,
                                      workers=1,
                                      callbacks=callbacks)
        plot_filename = 'model/train_history_spectrogram_NoSpike.pdf'
        plot_history(history, plot_filename, epochs)
        ex.add_artifact(plot_filename)
        #plot_attention(model=model, data_type=data_type, plot_name='attention_at', save_dir=sacred_dir)
    
        # list_keys=[key for key in loss_hystory]
        # print(list_keys)
    
        print('fitting done,saving model')
        time_string = timeStructured()
        path_best_model = 'model/{}_model_NoSpike_predict_spectrogram.h5'.format(time_string)
        model.save(path_best_model)
    
        ##############################################################
        #                    tests training
        ##############################################################
    
        batch_snd_in_test, batch_snd_out_test = generator_test.__getitem__()
        prediction = model.predict(batch_snd_in_test)
        one_sound,  one_predicted_sound,one_sound_out = batch_snd_in_test[0],  prediction[0],\
            batch_snd_out_test[0]
        fig, axs = plt.subplots(3)
        fig.suptitle('prediction plot vs inputs')
        axs[0].plot(one_sound)
        axs[0].set_title('input sound')
        axs[1].plot(one_sound_out)
        axs[1].set_title('output sound')
        axs[2].plot(one_predicted_sound)
        axs[2].set_title('predicted sound')
    
               

        fig_path = './model/prediction_NoSpike_spectrogram.pdf'
        fig.savefig(fig_path, bbox_inches='tight')
        ex.add_artifact(fig_path)
        ########################################################################
        #                              finetune
        #########################################################################
    
        #tensorboard = TensorBoard(log_dir='./model/logs/{}_predict_finetuned'.format(time_string))
    
        callbacks = []  # [tensorboard]
        generator_train_noisy, generator_test_noisy = getData(
                                                        sound_shape=(sound_len, n_channels,1),
                                                        spike_shape=(spike_len, n_channels,1),
                                                        data_type='denoising_spectrogram', 
                                                        batch_size=batch_size)
        noisy_history = model.fit_generator(generator_train_noisy,
                                            epochs=epochs,
                                            validation_data=generator_test_noisy,
                                            use_multiprocessing=False,
                                            shuffle=False,
                                            verbose=1,
                                            workers=1,
                                            callbacks=callbacks)
        plot_filename = 'model/denoising_history_NoSpike_spectrogram.pdf'
        plot_history(noisy_history, plot_filename, epochs)
        ex.add_artifact(plot_filename)
        #plot_attention(model=model, data_type=data_type, plot_name='attention_af', save_dir=sacred_dir)
    
        print('finetuning done,saving model')
        path_best_model_finetune = 'model/{}_model_spectrogram_NoSpike_finetuned.h5'.format(time_string)
        model.save(path_best_model_finetune)
    
        ##############################################################
        #                    tests finetuning
        ##############################################################
    
        batch_snd_in_test_denoise, batch_snd_out_test_denoise = generator_test_noisy.__getitem__()
        prediction_denoise = model.predict(batch_snd_in_test_denoise)
        np.save('./model/prediction_finetune_NoSpike_spectrogram', prediction_denoise)

        one_sound_noisy, one_sound_denoised, one_predicted_sound_denoised = batch_snd_in_test_denoise[0], \
                                                                         batch_snd_out_test_denoise[0], \
                                                                             prediction_denoise[0]
    
        fig, axs = plt.subplots(3)
        fig.suptitle('denoising prediction plot vs inputs')
        
        axs[0].plot(one_sound_noisy)
        axs[0].set_title('input sound')
        axs[1].plot(one_sound_denoised)
        axs[1].set_title('output sound')
        axs[2].plot(one_predicted_sound_denoised)
        axs[2].set_title('predicted sound')    
        fig_path = './model/prediction_finetuned_spectrogram.pdf'
        fig.savefig(fig_path, bbox_inches='tight')
        ex.add_artifact(fig_path)
    
        sacred_dir = os.path.join(*[CDIR, ex.observers[0].basedir, '1'])
        email_results(
            folders_list=[sacred_dir],
            name_experiment=' guinea, multiplied attention ',
            receiver_emails=['manucelotti@gmail.com', 'm.hosseinite@gmail.com'])

