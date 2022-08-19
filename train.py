#!/usr/bin/env python3
import os.path
import logging
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from options import *
from data_loader import trainDataset
from utils import utils_logger
from utils import utils_image as util
from torch.utils.data import DataLoader


def train_denoise(use_cuda=True, load_model=True, start=0, num_epochs=10):
    # ----------------- Preparation -----------------------
    opt = Option(use_cuda=use_cuda)
    device = opt.device
    results = 'results'
    task_current = 'denoise' 
    result_name = task_current + '_'
    E_path = os.path.join(results, result_name)  
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------- model -----------------------
    from models.network_unet import UNetRes
    from models.DelasD import NLayerDiscriminator
    from models.AttGenerator import AttenGenerator
    from models.Loss import GANLoss

    netG = UNetRes(in_nc=opt.channel+1, out_nc=opt.channel, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                   downsample_mode="strideconv", upsample_mode="convtranspose").to(device)
    netA = AttenGenerator().to(device)
    netD = NLayerDiscriminator(in_nc=opt.channel*2).to(device)

    model_path_G = opt.save_models_dir + 'drunet_color.pth'
    load_net(netG, model_path_G)

    if load_model: 
        s = str(start)
        model_path_D = opt.save_models_dir + s + '-D.pth'
        load_net(netD, model_path_D)
        model_path_A = opt.save_models_dir + s + '-A.pth'
        load_net(netA, model_path_A)

    # ----------------- optimizer -----------------------
    optimA = torch.optim.Adam(netA.parameters(), lr=opt.lr, weight_decay=0)
    optimD = torch.optim.Adam(netD.parameters(), lr=opt.lr, weight_decay=0)

    # ----------------- define loss -----------------------
    L1_criterion = nn.L1Loss()
    GAN_criterion = GANLoss(use_cuda=use_cuda)

    # test result dict
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    trainset = trainDataset(root=opt.root)
    loader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                        num_workers=opt.workers, pin_memory=True)
    length = len(loader)

    for k, v in netG.named_parameters():
        v.requires_grad = False

    netG.eval()

    for epoch in range(start+1, num_epochs+start+1):
        test_results['psnr'] = []
        for i, data_dict in enumerate(loader, 0):
            L = data_dict['L'].to(device)
            H = data_dict['H'].to(device)
            N = data_dict['noise'].to(device)
            level = data_dict['level'].to(device)

            # forward
            y1, P = netG.forward(torch.cat((L, N), dim=1))
            _, y2 = netA.forward(y1)

            # backward D
            optimD.zero_grad()
            fake2 = netD.forward(torch.cat((L, y2), dim=1))
            D_loss_fake =  GAN_criterion(fake2, False)
            real = netD.forward(torch.cat((L, H), dim=1))
            D_loss_real = GAN_criterion(real, True)
            D_loss = (D_loss_fake + D_loss_real) * 0.5
            D_loss.backward()
            optimD.step()

            # backward A
            optimA.zero_grad()
            y1, P = netG.forward(torch.cat((L, N), dim=1))
            u, y2 = netA.forward(y1)
            fake2 = netD.forward(torch.cat((L, y2), dim=1))
            G_loss_GAN = GAN_criterion(fake2, True)
            G_L1_loss = opt.lambda_ * L1_criterion(y2, H)
            # G_ucty_loss = torch.mean(torch.div(torch.abs(y2 - H), u) + torch.log(u)) * opt.lambda_
            G_loss = G_loss_GAN + G_L1_loss #+  G_ucty_loss
            G_loss.backward()
            optimA.step()

            if i % 10 == 0:
                e1 = util.tensor2uint(y2)
                h = util.tensor2uint(H)
                psnr1 = util.calculate_psnr(e1, h, border=0)
                test_results['psnr'].append(psnr1)
                logger.info("Epoch[{}/{}], Step[{}/{}], G_loss: {:.3f}, D_loss: {:.3f}, PSNR:{:.2f}"
                            .format(epoch, num_epochs+start, i, length, G_loss.item(), D_loss.item(), psnr1))

        logger.info("-------Average PSNR--------Epoch[{}], PSNR[{:.2f}]---------------"
                    .format(epoch, np.mean(test_results['psnr'])))

        if epoch % 20 == 0:
            save_path = opt.save_models_dir + str(epoch) + '-D.pth'
            save_net(netD, save_path)
            save_path = opt.save_models_dir + str(epoch) + '-A.pth'
            save_net(netA, save_path)


if __name__ == '__main__':
    train_denoise(use_cuda=True, load_model=True, start=220, num_epochs=20)