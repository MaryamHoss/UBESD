from tensorflow.keras.models import Model

from GenericTools.KerasTools.noise_curriculum import InputSensitiveGaussianNoise


def MakeNoisyInput(model):
    isgn = InputSensitiveGaussianNoise(0.)
    inputs = model.inputs
    if len(inputs) == 3:
        input_sound, input_spike, output_sound = inputs
    else:
        input_sound, output_sound = inputs

    sound = isgn(input_sound)
    spike = isgn(input_spike)
    inputs_model = [sound, spike, output_sound]
    output = model(inputs_model)
    model = Model(inputs, output)

    curriculum_noise = isgn.stddev
    model.curriculum_noise = curriculum_noise

    return model

