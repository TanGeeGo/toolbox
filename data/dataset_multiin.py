import glob
import random
import torch
import numpy as np
import scipy.io as scio
import torch.utils.data as data
import utils.utils_image as util
import utils.utils_filter as filter

class DatasetMultiin(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with H and L
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetMultiin, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 64

        # ------------------------------------
        # get the path of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: High path is empty.'
        assert self.paths_L, 'Error: L path is empty. Plain dataset assumes both L and H are given!'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(\
                len(self.paths_L), len(self.paths_H))
        
        # ------------------------------------
        # get the kernel and sigma (for fsanet)
        # ------------------------------------
        self.kernel_set = self.load_kernel()
        self.eptional = self.load_eptional()
        
    def load_kernel(self):
        kernel_file = sorted(glob.glob(self.opt['dataroot_K'] + "/*.mat"))

        kernel_set = list()
        for i in range(self.opt['kr_num']):
            kernel = scio.loadmat(kernel_file[i])
            kernel = kernel['PSF']
            # turning into gray scale (rgb make few differences, not a lot)
            kernel = kernel[:,:,0] + kernel[:,:,1] + kernel[:,:,2] 
            kernel = kernel / np.sum(kernel)
            kernel = torch.FloatTensor(kernel).unsqueeze(0)
            # form a batch
            kernel_set.append(kernel)
        
        kernel_set = torch.cat(kernel_set, dim=0)

        return kernel_set
    
    def load_eptional(self):
        # different eptional fitting type
        if self.opt["eptional_name"] == 'sinefit':
            return torch.FloatTensor(filter.sinefit(self.opt['en_size'], self.opt['en_size'], self.opt['omega'], 
                                                    self.opt['theta_num'], self.opt['eptional_sigma']))
        elif self.opt["eptional_name"] == 'gauss_fit':
            return torch.FloatTensor(filter.gauss_fit(self.opt['en_size'], self.opt['en_size'], self.opt['sig_low'], 
                                                      self.opt['sig_high'], self.opt['sig_num']))
        elif self.opt["eptional_name"] == 'dir_map':
            return torch.ones((1, self.opt['en_size'], self.opt['en_size']))

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_mat(H_path, 'hr', self.n_channels)

        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        img_L = util.imread_mat(L_path, 'lr', self.n_channels+2) # with fov info

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_L, patch_H = util.augment_img(patch_L, mode=mode), util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(patch_L), util.uint2tensor3(patch_H)

        else:

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path, 
                'Kernels': self.kernel_set, 'Eptional': self.eptional}

    def __len__(self):
        return len(self.paths_H)
