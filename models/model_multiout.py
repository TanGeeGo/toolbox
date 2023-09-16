import torch
import torch.nn.functional as F
from collections import OrderedDict

from models.model_plain import ModelPlain

from utils.utils_regularizers import regularizer_orth, regularizer_clip

class ModelMultiout(ModelPlain):
    """Train with pixel loss and has multiple outputs"""
    def __init__(self, opt):
        super(ModelMultiout, self).__init__(opt)

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
        if need_H:
            # multiple supervision 
            self.H = list()
            H_1 = data['H'].to(self.device)
            self.H.append(F.interpolate(H_1, scale_factor=0.25))
            self.H.append(F.interpolate(H_1, scale_factor=0.5))
            self.H.append(H_1)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        # ensure the length of E equals to H
        assert len(self.E) == len(self.H), ValueError('Output amount is not right')
        # G_loss_list = []
        if len(self.G_lossfn_weight) == 1:
            G_loss = self.G_lossfn(self.E[0], self.H[0]) + self.G_lossfn(self.E[1], self.H[1]) + \
                     self.G_lossfn(self.E[2], self.H[2])
        elif len(self.G_lossfn_weight) == 2:
            G_loss_main = self.G_lossfn(self.E[0], self.H[0]) + self.G_lossfn(self.E[1], self.H[1]) + \
                          self.G_lossfn(self.E[2], self.H[2])
            G_loss_aux = self.G_lossfn_aux(self.E[0], self.H[0]) + self.G_lossfn_aux(self.E[1], self.H[1]) + \
                         self.G_lossfn_aux(self.E[2], self.H[2])
            G_loss = self.G_lossfn_weight[0] * G_loss_main + \
                     self.G_lossfn_weight[1] * G_loss_aux
            # if self.opt_train['G_lossfn_type'] == 'l1+fft':
            #     loss_content = self.G_lossfn(self.E[0], self.H[0]) + self.G_lossfn(self.E[1], self.H[1]) + \
            #                 self.G_lossfn(self.E[2], self.H[2])
            #     loss_fft = self.G_lossfn(torch.fft.rfft2(self.E[0]), torch.fft.rfft2(self.H[0])) + \
            #             self.G_lossfn(torch.fft.rfft2(self.E[1]), torch.fft.rfft2(self.H[1])) + \
            #             self.G_lossfn(torch.fft.rfft2(self.E[2]), torch.fft.rfft2(self.H[2]))
            #     G_loss = self.G_lossfn_weight[0] * loss_content + self.G_lossfn_weight[1] * loss_fft
            # else:
            #     raise NotImplementedError('Loss type [{:s}] is not found.'.format(self.opt_train['G_lossfn_type']))
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
        if len(self.G_lossfn_weight) == 1:
            self.log_dict['G_loss'] = G_loss.item()
        elif len(self.G_lossfn_weight) == 2:
            self.log_dict[self.G_lossfn_type_[0]+'_loss'] = G_loss_main.item()
            self.log_dict[self.G_lossfn_type_[1]+'_loss'] = G_loss_aux.item()
            self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        # multiscale outputs, the highest resolution is in the last
        out_dict['E'] = self.E[-1].detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H[-1].detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        # multiscale outputs, the highest resolution is in the last
        out_dict['E'] = self.E[-1].detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict
    