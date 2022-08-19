#!/usr/bin/env python3
import os.path
import cv2
import logging

import numpy as np
from collections import OrderedDict
import hdf5storage
from scipy import ndimage

from utils import utils_logger
from utils import utils_model
from data_loader import testDataset
from utils import utils_pnp as pnp
from utils import utils_sisr as sisr
from utils import utils_image as util
from options import *
import warnings
warnings.filterwarnings("ignore")

def main(use_cuda=True):
    # Preparation
    start = 45
    opt = Option(use_cuda=use_cuda)
    device = opt.device
    noise_level_img = 7.65 / 255.0           # noise level for LR image
    noise_level_model = noise_level_img      # noise level of model
    iter_num = 24                             # number of iterations
    modelSigma1 = 49                         # sigma_1
    modelSigma2 = noise_level_model * 255.   # sigma_2

    show_img = False      # default: False
    save_L = False         # save LR image
    save_E = False         # save estimated image
    save_LEH = True      # save zoomed LR, E and H images
    border = 0

    # --------------------------------
    # load kernel
    # --------------------------------
    sf = 1      # scale factor
    task_current = 'deblur'
    n_channels = 3
    testsets = 'testsets'
    results = 'results'
    testset_name = 'set3c'
    result_name = testset_name + '_' + task_current
    kernels = hdf5storage.loadmat(os.path.join('kernels', 'Levin09.mat'))['kernels']
    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name)  # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)  # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    # load model
    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                    downsample_mode="strideconv", upsample_mode="convtranspose")

    load_model_ckpt(model, start, opt, opt.name_A)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # logger
    logger.info('image sigma:{:.3f}, model sigma:{:.3f}'
                .format(noise_level_img, noise_level_model))
    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []  # record average PSNR for each kernel

    set = testDataset(opt.test_root)
    for k_index in range(kernels.shape[1]):
        logger.info('-------k:{:>2d} ---------'.format(k_index))
        test_results = OrderedDict()
        test_results['psnr'] = []
        k = kernels[0, k_index].astype(np.float64)
        logger.info("kernel size: ({}, {})".format(k.shape[0], k.shape[1]))
       
        for i in range(len(set)):
            img_H, img_name = set[i]
            # img_H = util.modcrop(img_H, 8)  # set image size
            img_L = ndimage.filters.convolve(img_H, np.expand_dims(k, axis=2), mode='wrap')
            util.imshow(img_L) if show_img else None
            
            img_L = util.uint2single(img_L)  # img_L: convolved image
            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img, img_L.shape)  # add AWGN
            # (2) ------------ get rhos and sigmas ------------
            rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255 / 255., noise_level_model), iter_num=iter_num,
                                             modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
            rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

            # (3) -------------- initialize x, and pre-calculation --------------
            x = util.single2tensor4(img_L).to(device)
            img_L_tensor, k_tensor = util.single2tensor4(img_L), util.single2tensor4(np.expand_dims(k, 2))
            [k_tensor, img_L_tensor] = util.todevice([k_tensor, img_L_tensor], device)
            FB, FBC, F2B, FBFy = sisr.pre_calculate(img_L_tensor, k_tensor, sf)

            # (4) -------------- main iterations --------------
            for j in range(iter_num):
                # -------------- step 1, FFT --------------
                tau = rhos[j].float().repeat(1, 1, 1, 1)
                x = sisr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf)

                # -------------- step 2, denoiser --------------
                x = util.augment_img_tensor4(x, j % 8)
                x = torch.cat((x, sigmas[j].float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
                x = utils_model.test_mode(model, x, mode=2, refield=32, min_size=256, modulo=16)

                if j % 8 == 3 or j % 8 == 5:
                    x = util.augment_img_tensor4(x, 8 - j % 8)
                else:
                    x = util.augment_img_tensor4(x, j % 8)

            # (3) img_E
            img_E = util.tensor2uint(x)
            if save_E:
                util.imsave(img_E, os.path.join(E_path, img_name + '_k' + str(k_index) + '.png'))

            # (4) img_LEH
            if save_LEH:
                img_L = util.single2uint(img_L)
                k_v = k / np.max(k) * 1.0
                k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                k_v = cv2.resize(k_v, (3 * k_v.shape[1], 3 * k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I = cv2.resize(img_L, (sf * img_L.shape[1], sf * img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
                img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
                # util.imshow(np.concatenate([img_I, img_E, img_H], axis=1),
                #             title='LR / Recovered / Ground-truth') if show_img else None
                util.imsave(np.concatenate([img_I, img_E, img_H], axis=1),
                            os.path.join(E_path, img_name + '_k' + str(k_index) + '_LEH.png'))

            if save_L:
                util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name + '_k' + str(k_index) + '_LR.png'))

            psnr = util.calculate_psnr(img_E, img_H, border=border)  # change with your own border
            test_results['psnr'].append(psnr)
            logger.info('{:->4d}--> {:>10s} --k:{:>2d} PSNR: {:.2f}dB'.format(i + 1, img_name, k_index, psnr))

        # --------------- Average PSNR -----------------
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR of ({}), kernel: ({}) sigma: ({:.2f}): {:.2f} dB'.
                    format(testset_name, k_index, noise_level_model, ave_psnr))
        test_results_ave['psnr'].append(ave_psnr)


if __name__ == '__main__':
    main()
