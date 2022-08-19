#!/usr/bin/env python3
import os.path
import logging
import numpy as np
from collections import OrderedDict

from options import *
from data_loader import testDataset
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_model
from utils import utils_image as util
from utils.crop_ import *


def test_denoise(use_cuda=True, load_pt=220):
    # ----------------- Preparation -----------------------
    border = 0 # shave border to calculate PSNR and SSIM
    opt = Option(use_cuda=use_cuda)
    device = opt.device      
    results = 'results'
    task_current = 'denoise'
    result_name = task_current + '_result'
    # ----------------------------------------
    E_path = os.path.join(results, result_name)  
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------- model -----------------------
    from models.network_unet import UNetRes as net
    from models.AttGenerator import AttenGenerator
    netG = net(in_nc=opt.channel+1, out_nc=opt.channel, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                downsample_mode="strideconv", upsample_mode="convtranspose").to(device)
    netA = AttenGenerator().to(device)

    model_path = './denoise_models/'
    modelG_path = model_path + 'drunet_color.pth'
    modelA_path = model_path + str(load_pt) + '-A.pth'
    netG.load_state_dict(torch.load(modelG_path), strict=True)
    netA.load_state_dict(torch.load(modelA_path), strict=True)

    netG.eval()
    netA.eval()

    for k, v in netG.named_parameters():
        v.requires_grad = False
        
    for k, v in netA.named_parameters():
        v.requires_grad = False

    # test result dict
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    
    testset = testDataset(root=opt.test_root,sigma=50)
    loader = DataLoader(testset, batch_size=1, shuffle=False,
                        num_workers=opt.workers, pin_memory=True)


    for i, data_dict in enumerate(loader, 0):
        L = data_dict['L'].to(device)
        H = data_dict['H'].to(device)
        N = data_dict['noise'].to(device)
        img_name = data_dict['name'][0]

        x = torch.cat((L, N), dim=1)
        y = utils_model.test_mode(netG, x, mode=5, refield=64)
       
        # ------------- PSNR and SSIM -------------------
        e = util.tensor2uint(y)
        h = util.tensor2uint(H)
        # util.imsave(y, os.path.join(E_path, 'combine_'+img_name))
        psnr = util.calculate_psnr(e, h, border=border)
        ssim = util.calculate_ssim(e, h, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name, psnr, ssim))
    
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'
                .format(result_name, ave_psnr, ave_ssim))
    

if __name__ == '__main__':
    test_denoise()