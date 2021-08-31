

from TrialsOfNeuralVocalRecon.neural_models.model_helpers_Maryam import build_autoencoder_convolutional_voltage_gated, \
            build_autoencoder_convolutional_voltage_gated_noSpike


def create_guinea_model(exp_type,activation_encode,activation_spikes,activation_decode,activation_all,learning_rate,n_convolutions):

    if exp_type=="withSpikes":
        
        data_type = 'real_prediction'
        
        model_guinea=build_autoencoder_convolutional_voltage_gated(activation_encode=activation_encode,
                                                          activation_spikes=activation_spikes,
                                                          activation_decode=activation_decode,
                                                          activation_all=activation_all,
                                                          learning_rate=learning_rate,
                                                          n_convolutions=n_convolutions)
    elif exp_type=="noSpikes":
        
        data_type = 'real_prediction_NoSpike'
        model_guinea = build_autoencoder_convolutional_voltage_gated_noSpike(activation_encode=activation_encode,
                                                                      activation_decode=activation_decode,
                                                                      activation_all=activation_all, 
                                                                      learning_rate=learning_rate,
                                                                      n_convolutions=n_convolutions)
    return model_guinea,data_type

