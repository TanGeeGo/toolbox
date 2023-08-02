
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M
    elif model == 'multiout':
        from models.model_multiout import ModelMultiout as M
    elif model == 'multiin':
        from models.model_multiin import ModelMultiin as M
    elif model == 'progressive':
        from models.model_progressive import ModelProgressive as M
    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    return m