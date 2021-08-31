from TrialsOfNeuralVocalRecon.neural_models.models_convolutional import build_conv_with_fusion, \
    build_conv_with_fusion_skip,build_conv_with_fusion_yamnet, build_conv_with_fusion_skip_tests
from TrialsOfNeuralVocalRecon.neural_models.models_transformer import build_transformer
from TrialsOfNeuralVocalRecon.neural_models.multirresolution import MakeMultirresolution


def build_model(*args, **kwargs):
    if 'transformer' in kwargs['data_type']:
        model = build_transformer(*args, **kwargs)
    # elif 'separable' in kwargs['data_type']:
    #     model = build_conv_with_fusion_separable(*args, **kwargs)
    elif 'skip' in kwargs['data_type'] and not'film_first' in kwargs['data_type'] and not 'film_last' in kwargs['data_type']:
        model = build_conv_with_fusion_skip(*args, **kwargs)
        
    elif 'yamnet' in kwargs['data_type']:
        model = build_conv_with_fusion_yamnet(n_convolutions=5, min_filters=16, max_filters=512)
        
    elif 'film_first' in kwargs['data_type'] or 'film_last' in kwargs['data_type'] :
        model = build_conv_with_fusion_skip_tests(*args, **kwargs)

        
    else:

        # model = build_conv_with_fusion_big(*args, **kwargs)
        model = build_conv_with_fusion(*args, **kwargs)

    model.summary()

    if 'multirresolution' in kwargs['data_type']:
        model = MakeMultirresolution(model, kwargs['data_type'])

    return model
