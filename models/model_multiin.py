import torch.nn.functional as F
from collections import OrderedDict

from models.model_plain import ModelPlain

from utils.utils_regularizers import regularizer_orth, regularizer_clip

class ModelMultiin(ModelPlain):
    """Train with pixel loss and has multiple outputs"""
    def __init__(self, opt):
        super(ModelMultiin, self).__init__(opt)

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        self.Kernels = data['Kernels'].to(self.device)
        self.Eptional = data['Eptional'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        # multiple input to the models
        self.E = self.netG(self.L, self.Kernels, self.Eptional)
    