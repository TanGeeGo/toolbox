import os
from pathlib import Path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import init_dist, get_dist_info
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--dist', default=False)
    
    args = parser.parse_args()
    
    opt = option.parse(args.opt, is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    log_dir = Path(opt['path']['log'])
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    # ----------------------------------------
    # distributed settings of training 
    # ----------------------------------------
    opt['dist'] = args.dist
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['word_size'] = get_dist_info()

    if opt['rank'] == 0:
        print('export CUDA_VISIBLE_DEVICES=' + ','.join(str(x) for x in opt['gpu_ids']))
        print('number of GPUs is: ' + str(opt['num_gpu']))
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
    
    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)
    
    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            if opt['rank'] == 0:
                print('Dataset [{:s} - {:s}] is created.'.format(train_set.__class__.__name__, dataset_opt['name']))
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'valid':
            valid_set = define_Dataset(dataset_opt)
            if opt['rank'] == 0:
                print('Dataset [{:s} - {:s}] is created.'.format(valid_set.__class__.__name__, dataset_opt['name']))
            valid_loader = DataLoader(valid_set, 
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=False,
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=False,
                                      pin_memory=True)
        else:
            # leave the phase of test into the evaluation
            pass

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    if opt['rank'] == 0:
        print('Training model [{:s}] is created.'.format(model.__class__.__name__))
        if opt['netG']['init_type'] not in ['default', 'none']:
            print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(\
                opt['netG']['init_type'],  opt['netG']['init_bn_type'],  opt['netG']['init_gain']))
        else:
            print('Pass this initialization! Initialization was done during network definition!')
            
    model.init_train()
    # if you want to see the detail of the model
    # if opt['rank'] == 0:
    #     logger.info(model.info_network())
    #     logger.info(model.info_params())
    #     pass

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    for epoch in range(opt['train']['total_epoch']): # TODO: the terminate condition
        if opt['dist']:
            train_sampler.set_epoch(epoch) # set the sampler in data distribution

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) feed patch pairs
            # -------------------------------
            model.feed_data(train_data, epoch) if opt['model'] == 'progressive' else model.feed_data(train_data)
            
            # -------------------------------
            # 2) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 3) training information
            # -------------------------------
            logs = model.current_log()
            writer.add_scalar("lr", model.current_learning_rate(), epoch + 1)
            for k, v in logs.items():  # merge log information into message
                writer.add_scalar(k, v, epoch + 1)
            
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 4) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 5) update learning rate
            # -------------------------------
            model.update_learning_rate()

            # -------------------------------
            # 6) validating
            # -------------------------------
            if current_step % opt['train']['checkpoint_valid'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                idx = 0

                for valid_data in valid_loader:
                    idx += 1
                    image_name_ext = os.path.basename(valid_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(valid_data)
                    model.valid()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                # validating log
                writer.add_scalar('Average PSNR', avg_psnr, epoch + 1)
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))
    writer.close()

if __name__ == '__main__':
    main()
