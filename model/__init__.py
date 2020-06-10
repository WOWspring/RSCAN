from model import srfeat, srgan, esrgan, edsr, rcan, erca, rcsan


def get_generator(model_arc, is_train=True):
    if model_arc == 'srfeat':
        model = srfeat.generator(is_train=is_train)
    elif model_arc == 'srgan':
        model = srgan.generator(is_train=is_train)
    elif model_arc == 'esrgan':
        model = esrgan.generator()
    elif model_arc == 'edsr':
        model = edsr.generator()
    elif model_arc == 'rcan':
        model = rcan.generator()
    #我的修改，加了rcsan
    elif model_arc == 'rcsan':
        model = rcsan.generator()
    #
    elif model_arc == 'erca':
        model = erca.generator()
    elif model_arc == 'gan':
        model = srfeat.generator(is_train=is_train, use_bn=False)
    else:
        #我的修改，加了rcsan
        raise Exception('Wrong model architecture! It should be srfeat, argan, esrgan, edsr, rcan or erca or rcsan.')
        #
    return model
