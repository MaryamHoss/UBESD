from UBESD.neural_models.models_convolutional import build_conv_with_fusion, \
    build_conv_with_fusion_skip



def build_model(*args, **kwargs):

    if 'skip' in kwargs['data_type']:
        model = build_conv_with_fusion_skip(*args, **kwargs)

    else:

        # model = build_conv_with_fusion_big(*args, **kwargs)
        model = build_conv_with_fusion(*args, **kwargs)

    model.summary()



    return model
