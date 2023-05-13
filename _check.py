def check_parameters(pre):
    if pre and (not callable(pre)):
        raise RuntimeError('Input argument "pre" should be callable.')