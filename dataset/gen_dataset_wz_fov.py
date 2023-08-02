"""
The frequency self-adaptive network for optical degradation correction
paper link:
@article{Lin_2022_OE,
  author = {Ting Lin and ShiQi Chen and Huajun Feng and Zhihai Xu and Qi Li and Yueting Chen},
  journal = {Opt. Express},
  keywords = {All optical devices; Blind deconvolution; Image processing; Image quality; Optical design; Ray tracing},
  number = {13},
  pages = {23485--23498},
  publisher = {Optica Publishing Group},
  title = {Non-blind optical degradation correction via frequency self-adaptive and finetune tactics},
  volume = {30},
  month = {Jun},
  year = {2022},
  url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-30-13-23485},
  doi = {10.1364/OE.458530},
  abstract = {In mobile photography applications, limited volume constraints the diversity of optical design. In addition to the narrow space, the deviations introduced in mass production cause random bias to the real camera. In consequence, these factors introduce spatially varying aberration and stochastic degradation into the physical formation of an image. Many existing methods obtain excellent performance on one specific device but are not able to quickly adapt to mass production. To address this issue, we propose a frequency self-adaptive model to restore realistic features of the latent image. The restoration is mainly performed in the Fourier domain and two attention mechanisms are introduced to match the feature between Fourier and spatial domain. Our method applies a lightweight network, without requiring modification when the fields of view (FoV) changes. Considering the manufacturing deviations of a specific camera, we first pre-train a simulation-based model, then finetune it with additional manufacturing error, which greatly decreases the time and computational overhead consumption in implementation. Extensive results verify the promising applications of our technique for being integrated with the existing post-processing systems.},
}
"""
##### Data preparation file for training Model on the Dataset with field of view information ########

import tifffile
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx
from joblib import Parallel, delayed
import scipy.io as scio

def crop_files(file_):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = tifffile.imread(lr_file) 
    hr_img = tifffile.imread(hr_file)
    # normalize to 0~1 float
    lr_img = lr_img.astype(np.float32) / 65535.
    hr_img = hr_img.astype(np.float32) / 65535.
    num_patch = 0
    
    # field of view calculation
    h, w = lr_img.shape[:2]
    h_range = np.arange(0, h, 1)
    w_range = np.arange(0, w, 1)
    fov_w, fov_h = np.meshgrid(w_range, h_range)
    fov_h = ((fov_h - (h-1)/2) / ((h-1)/2)).astype(np.float32) # normalization
    fov_w = ((fov_w - (w-1)/2) / ((w-1)/2)).astype(np.float32) # normalization
    fov_h = np.expand_dims(fov_h, -1)
    fov_w = np.expand_dims(fov_w, -1)
    lr_wz_fov = np.concatenate([lr_img, fov_h, fov_w], 2)

    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=np.int32))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=np.int32))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in h1:
            for j in w1:
                num_patch += 1
                
                lr_patch = lr_wz_fov[i:i+patch_size, j:j+patch_size,:]
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size,:]
                
                lr_savename = os.path.join(lr_tar, filename + '-' + str(num_patch) + '.mat')
                hr_savename = os.path.join(hr_tar, filename + '-' + str(num_patch) + '.mat')
                
                scio.savemat(lr_savename, {'lr': lr_patch})
                scio.savemat(hr_savename, {'hr': hr_patch})

    else:
        lr_savename = os.path.join(lr_tar, filename + '.mat')
        hr_savename = os.path.join(hr_tar, filename + '.mat')
        
        scio.savemat(lr_savename, {'lr': lr_patch})
        scio.savemat(hr_savename, {'hr': hr_patch})

############ Prepare Training data ####################
num_cores = 10
patch_size = 512
overlap = 256
p_max = 0

src = '/data1/Aberration_Correction/synthetic_datasets/train'
tar = '/data1/Aberration_Correction/train'

os.makedirs(tar, exist_ok=True)

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

lr_files = natsorted(glob(os.path.join(src, 'input', '*.tiff')) + glob(os.path.join(src, 'input', '*.png')))
hr_files = natsorted(glob(os.path.join(src, 'target', '*.tiff')) + glob(os.path.join(src, 'target', '*.png')))

files = [(i, j) for i, j in zip(lr_files, hr_files)]

Parallel(n_jobs=num_cores)(delayed(crop_files)(file_) for file_ in tqdm(files))

############ Prepare Validating data ####################
num_cores = 10
patch_size = 512
overlap = 256
p_max = 0
src = '/data1/Aberration_Correction/synthetic_datasets/val'
tar = '/data1/Aberration_Correction/val'

os.makedirs(tar, exist_ok=True)

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

lr_files = natsorted(glob(os.path.join(src, 'input', '*.tiff')) + glob(os.path.join(src, 'input', '*.png')))
hr_files = natsorted(glob(os.path.join(src, 'target', '*.tiff')) + glob(os.path.join(src, 'target', '*.png')))

files = [(i, j) for i, j in zip(lr_files, hr_files)]

Parallel(n_jobs=num_cores)(delayed(crop_files)(file_) for file_ in tqdm(files))
