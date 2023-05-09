import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_plain import ModelPlain
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss
from utils.utils_regularizers import regularizer_orth, regularizer_clip

class ModelPlain_1in3out(ModelPlain):
    """
    Train with one input and with three outputs (list)
    The outputs may have different resolution, please check the network
    """
    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1' or G_lossfn_type == 'msl1fft':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        
        if self.opt_train['G_lossfn_type'] == 'msl1fft':
            # multiscale spatial domain
            H_2 = F.interpolate(self.H, scale_factor=0.5, mode='bilinear')
            H_4 = F.interpolate(self.H, scale_factor=0.25,mode='bilinear')
            l1 = self.G_lossfn(self.E[0], H_4)
            l2 = self.G_lossfn(self.E[1], H_2)
            l3 = self.G_lossfn(self.E[2], self.H)

            # multiscale fourier domain
            H_fft4 = torch.fft.rfft(H_4, signal_ndim=2, normalized=False, onesided=False)
            E_fft4 = torch.fft.rfft(self.E[0], signal_ndim=2, normalized=False, onesided=False)
            H_fft2 = torch.fft.rfft(H_2, signal_ndim=2, normalized=False, onesided=False)
            E_fft2 = torch.fft.rfft(self.E[1], signal_ndim=2, normalized=False, onesided=False)
            H_fft1 = torch.fft.rfft(self.H, signal_ndim=2, normalized=False, onesided=False)
            E_fft1 = torch.fft.rfft(self.E[2], signal_ndim=2, normalized=False, onesided=False)
            f1 = self.G_lossfn(H_fft4, E_fft4)
            f2 = self.G_lossfn(H_fft2, E_fft2)
            f3 = self.G_lossfn(H_fft1, E_fft1)
            
            # accumulate loss
            assert len(self.G_lossfn_weight) == 2, 'two weights for the loss terms to trade-off'
            G_loss = self.G_lossfn_weight[0] * (l1+l2+l3) + self.G_lossfn_weight[1] * (f1+f2+f3)
        else:
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)

        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

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