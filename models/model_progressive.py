import random
import numpy as np
from models.model_plain import ModelPlain

class ModelProgressive(ModelPlain):
    """Train with pixel loss and has multiple outputs"""
    def __init__(self, opt):
        super(ModelProgressive, self).__init__(opt)
        # ------------------------------------
        # check the configuration of progressive training
        # ------------------------------------
        self.opt = opt
        self.dist = self.opt['dist']
        self.num_gpu = self.opt['num_gpu']
        self.scale = self.opt['scale']
        self.batch_size = self.opt['datasets']['train'].get('dataloader_batch_size')//self.num_gpu if self.dist else \
            self.opt['datasets']['train'].get('dataloader_batch_size')
        self.H_size = self.opt['datasets']['train'].get('H_size')
        mini_batch_sizes = self.opt['datasets']['train'].get('mini_batch_sizes')
        self.mini_batch_sizes = [bs // self.num_gpu for bs in mini_batch_sizes] if self.dist else mini_batch_sizes
        iters_milestones = self.opt['datasets']['train'].get('iters_milestones')
        self.mini_H_sizes = self.opt['datasets']['train'].get('mini_H_sizes')
        assert self.mini_batch_sizes and iters_milestones and self.mini_H_sizes, 'Error: Key progressive is empty.'
        assert len(self.mini_batch_sizes) == len(iters_milestones) and \
            len(self.mini_H_sizes) == len(iters_milestones), 'Error: List mismatch - batch:{}, milestone:{}, H_size{}'.format(\
            len(self.mini_batch_sizes), len(iters_milestones), len(self.mini_H_sizes))

        self.iters_milestones = np.array([sum(iters_milestones[0:i + 1]) for i in range(0, len(iters_milestones))])

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, iter, need_H=True):
        L = data['L']
        if need_H:
            H = data['H']

        # progressive changing the batch size and patch size during the training
        progressive_seq = ((iter > self.iters_milestones) != True).nonzero()[0]
        progressive_idx = len(self.iters_milestones) - 1 if len(progressive_seq) == 0 else progressive_seq[0]

        mini_batch_size = self.mini_batch_sizes[progressive_idx]
        mini_H_size = self.mini_H_sizes[progressive_idx]

        assert mini_batch_size < self.batch_size, ValueError('Mini Batch Size {} must smaller than Batch Size {}'.format(mini_batch_size, self.batch_size))
        assert mini_H_size < self.H_size, ValueError('Mini H Size {} must smaller than H Size {}'.format(mini_H_size, self.H_size))

        if mini_batch_size < self.batch_size:
            batch_pick = random.sample(range(0, self.batch_size), k=mini_batch_size)
            L = L[batch_pick]
            if need_H:
                H = H[batch_pick]

        if mini_H_size < self.H_size:
            x0 = int((self.H_size - mini_H_size) * random.random())
            y0 = int((self.H_size - mini_H_size) * random.random())
            L = L[..., x0 : x0 + mini_H_size, y0 : y0 + mini_H_size]
            if need_H:
                H = H[..., x0*self.scale : (x0+mini_H_size)*self.scale, y0*self.scale : (y0+mini_H_size)*self.scale]
            
        self.L = L.to(self.device)
        if need_H:
            self.H = H.to(self.device)
 