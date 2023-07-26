import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import logging
import torch
from torch.utils.data import DataLoader

from utils import utils_image as util
from utils import utils_option as option

from utils import utils_logger
from data.select_dataset import define_Dataset
from models.select_model import define_Model

def main(json_path='options/option.json'):
    
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=False)
    
    iters, path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = path_G

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'test'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and valid
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, 
                                     batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False,
                                     num_workers=dataset_opt['dataloader_num_workers'],
                                     drop_last=False,
                                     pin_memory=True)
        else:
            # leave the phase of train and valid into the training
            pass

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_test()

    '''
    # ----------------------------------------
    # Step--4 (main testing)
    # ----------------------------------------
    '''

    for i, test_data in enumerate(test_loader):

        # -------------------------------
        # 1) feed patch pairs
        # -------------------------------
        model.feed_data(test_data)

        # -------------------------------
        # 2) evaluate data
        # -------------------------------
        model.netG_forward()

        visuals = model.current_visuals()
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])

        # -------------------------------
        # 3) save tested image E
        # -------------------------------
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        save_E_img_path = os.path.join(opt['path']['images'], '{:s}_pred.png'.format(img_name))
        save_H_img_path = os.path.join(opt['path']['images'], '{:s}_grdt.png'.format(img_name))
        util.imsave(E_img, save_E_img_path)
        util.imsave(H_img, save_H_img_path)

        # -----------------------
        # 4) calculate indicators
        # -----------------------
        psnr = util.calculate_psnr(E_img, H_img)
        ssim = util.calculate_ssim(E_img, H_img)
        model.log_dict['psnr'].append(psnr)
        model.log_dict['ssim'].append(ssim)
        if H_img.ndim == 3:
            E_img_y = util.bgr2ycbcr(E_img.astype(np.float32) / 255.) * 255.
            H_img_y = util.bgr2ycbcr(H_img.astype(np.float32) / 255.) * 255.
            psnr_y = util.calculate_psnr(E_img_y, H_img_y)
            ssim_y = util.calculate_ssim(E_img_y, H_img_y)
            model.log_dict['psnr_y'].append(psnr_y)
            model.log_dict['ssim_y'].append(ssim_y)

        logger.info('Image:{}, PSNR: {:<.4f}dB, SSIM: {:<.4f}, PSNR_Y: {:<.4f}dB, SSIM_Y: {:<.4f}\n'.format(\
            img_name, psnr, ssim, psnr_y, ssim_y))
        
    logger.info('PSNR_Average: {:<.4f}dB, SSIM_Average: {:<.4f}, PSNR_Y_Average: {:<.4f}dB, SSIM_Y_Average: {:<.4f}\n'.format(\
        sum(model.log_dict['psnr'])/(i+1), sum(model.log_dict['ssim'])/(i+1), sum(model.log_dict['psnr_y'])/(i+1), sum(model.log_dict['ssim_y'])/(i+1)))

if __name__ == '__main__':
    main()