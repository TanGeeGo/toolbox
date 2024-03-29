from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam, AdamW

from models.select_network import define_G
from models.model_base import ModelBase
from models._loss import CharbonnierLoss, SSIMLoss, FFTLoss, PSNRLoss, PerceptualLoss
from models._lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR, CosineAnnealingRestartCyclicLR, GradualWarmupScheduler

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # initialize evaluation
    # ----------------------------------------
    def init_test(self):
        self.load()
        self.netG.eval()
        self.log_dict = OrderedDict()         # log
        self.log_dict['psnr'] = []
        self.log_dict['ssim'] = []
        self.log_dict['psnr_y'] = []
        self.log_dict['ssim_y'] = []

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        # check the cases of one loss or multiple losses
        self.G_lossfn_type_ = G_lossfn_type.split('+')
        if len(self.G_lossfn_type_) == 1:
            if G_lossfn_type == 'l1':
                self.G_lossfn = nn.L1Loss().to(self.device)
            elif G_lossfn_type == 'l2':
                self.G_lossfn = nn.MSELoss().to(self.device)
            elif G_lossfn_type == 'l2sum':
                self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
            elif G_lossfn_type == 'ssim':
                self.G_lossfn = SSIMLoss().to(self.device)
            elif G_lossfn_type == 'psnr':
                self.G_lossfn = PSNRLoss().to(self.device)
            elif G_lossfn_type == 'charbonnier':
                self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
            elif G_lossfn_type == 'smoothl1':
                self.G_lossfn = nn.SmoothL1Loss(reduction='none', beta=0.01)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        elif len(self.G_lossfn_type_) == 2:
            if G_lossfn_type == 'l1+ssim':
                self.G_lossfn = nn.L1Loss().to(self.device)
                self.G_lossfn_aux = SSIMLoss().to(self.device)
            elif G_lossfn_type == 'l1+fft':
                self.G_lossfn = nn.L1Loss().to(self.device)
                self.G_lossfn_aux = FFTLoss().to(self.device)
            elif G_lossfn_type == 'l2+perc':
                self.G_lossfn = nn.MSELoss().to(self.device)
                self.G_lossfn_aux = PerceptualLoss().to(self.device)

        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        if len(self.G_lossfn_type_) > 1:
            assert len(self.G_lossfn_type_) == len(self.G_lossfn_weight), ValueError(\
                'Loss type not equals to Loss weight.')
        else:
            pass

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(self.netG.parameters(), lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        elif self.opt_train['G_optimizer_type'] == 'adamw':
            self.G_optimizer = AdamW(self.netG.parameters(), lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer, 
                                                            self.opt_train['G_scheduler_period'],
                                                            eta_min=self.opt_train['G_scheduler_eta_min']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingRestartCyclicLR':
            self.schedulers.append(CosineAnnealingRestartCyclicLR(self.G_optimizer,
                                                            periods=self.opt_train['G_scheduler_periods'],
                                                            restart_weights=self.opt_train['G_scheduler_restart_weights'],
                                                            eta_mins=self.opt_train['G_scheduler_eta_mins']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'GradualWarmupScheduler':
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.G_optimizer, 
                                                                          self.opt_train['total_epoch']-self.opt_train['G_scheduler_warmup_epochs'], 
                                                                          eta_min=self.opt_train['G_scheduler_eta_min']
                                                                          )
            self.schedulers.append(GradualWarmupScheduler(self.G_optimizer, 
                                                          multiplier=self.opt_train['G_scheduler_multiplier'],
                                                          total_epoch=self.opt_train['G_scheduler_warmup_epochs'], 
                                                          after_scheduler=scheduler_cosine
                                                          ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingLR':
            self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(self.G_optimizer,
                                                                              T_max=self.opt_train['total_epoch'],
                                                                              eta_min=self.opt_train['G_scheduler_eta_min']
                                                                              ))
        else:
            raise NotImplementedError

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
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        if len(self.G_lossfn_weight) == 1:
            G_loss = self.G_lossfn_weight[0] * self.G_lossfn(self.E, self.H)
        elif len(self.G_lossfn_weight) == 2:
            G_loss_main = self.G_lossfn(self.E, self.H)
            G_loss_aux = self.G_lossfn_aux(self.E, self.H)
            G_loss = self.G_lossfn_weight[0] * G_loss_main + \
                self.G_lossfn_weight[1] * G_loss_aux

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
    # valid / inference
    # ----------------------------------------
    def valid(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # valid / inference x8
    # ----------------------------------------
    def validx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
    