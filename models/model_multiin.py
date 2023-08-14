import torch
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
        if self.opt['netG']['net_type'] == "fsanet":
            self.Kernels = data['Kernels'].to(self.device)
            self.Eptional = data['Eptional'].to(self.device)
        elif self.opt['netG']['net_type'] == 'painter':
            self.Mask = data['Mask'].to(self.device)
            self.Valid = data['Valid'].to(self.device)

        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        # multiple input to the models
        if self.opt['netG']['net_type'] == "fsanet":
            self.E = self.netG(self.L, self.Kernels, self.Eptional)
        elif self.opt['netG']['net_type'] == 'painter':
            loss, self.E, self.Mask = self.netG(self.L, self.H, self.Mask, self.Valid)
            return loss
            
    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        
        if self.opt['netG']['net_type'] == 'painter':
            G_loss = self.netG_forward()
        else:
            self.netG_forward()
            if len(self.G_lossfn_weight) == 1:
                G_loss = self.G_lossfn_weight[0] * self.G_lossfn(self.E, self.H)
            elif len(self.G_lossfn_weight) == 2:
                G_loss = self.G_lossfn_weight[0] * self.G_lossfn(self.E, self.H) + \
                    self.G_lossfn_weight[1] * self.G_lossfn_aux(self.E, self.H)
        
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])
