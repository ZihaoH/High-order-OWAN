#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable
from skimage.measure import compare_psnr as ski_psnr
from skimage.measure import compare_ssim as ski_ssim
import os
import logging

import torch.nn.functional as F
from data_load_own import get_training_set, get_test_set
from data_load_mix import get_dataset_deform
import utils
import argparse
import csv



parser = argparse.ArgumentParser(description='Operation-wise Attention Network')
parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
parser.add_argument('--mode', '-m', default='mix', help='Mode (mix / yourdata)')
parser.add_argument('--model_path', '-m_path', default='./trained_model/best_model.pth.tar', help='model file')
parser.add_argument('--model_id', '-m_id', type=int, default=2, help='model id')
parser.add_argument('--layer_num', '-l', type=int, default=10, help='Num. of Block')
parser.add_argument('--rank', '-r', type=int, default=16, help='Num. of rank')
parser.add_argument('--class_name', '-cn', default='moderate', help='class name of test set')

args = parser.parse_args()

def load_checkpoint(model, checkpoint_PATH, optimizer, last_epoch=0):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_CKPT['state_dict'].items():
            name = k[7:]  # remove the "module." when the Model file was saved in multiple GPU
            new_state_dict[name] = v
        try:
            model.load_state_dict(new_state_dict)
        except:
            model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!' + checkpoint_PATH)
        optimizer.load_state_dict(model_CKPT['optimizer'])
        last_epoch = model_CKPT['epoch']
    return model, optimizer, last_epoch


# load dataset
if args.mode == 'mix' or args.mode == 'yourdata':
    if args.mode == 'mix':
        num_work = 8

        train_dir = '../'
        val_dir = './dataset/valid/' 
        test_dir = './dataset/test/' 

        test_set = get_dataset_deform(train_dir, val_dir, test_dir, 2,class_name=args.class_name)
        test_dataloader = DataLoader(dataset=test_set, num_workers=num_work, batch_size=1, shuffle=False, pin_memory=False)
    elif args.mode == 'yourdata':
        num_work = 8
        test_input_dir = '/dataset/yourdata_test/input/'
        test_target_dir = '/dataset/yourdata_test/target/'
        test_set = get_training_set(test_input_dir, test_target_dir, False)
        test_dataloader = DataLoader(dataset=test_set, num_workers=num_work, batch_size=1, shuffle=False, pin_memory=False)
else:
    print('\tInvalid input dataset name at CNN_train()')
    exit(1)


# model

gpuID = 0
torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
L1_loss = nn.L1Loss()
L1_loss = L1_loss.cuda(gpuID)
steps=4
rank=args.rank
layer_num=args.layer_num
model_id=args.model_id
if model_id==1:
    from model.OWAN import Network as Network1
    model = Network1(16, layer_num, L1_loss, gpuID=gpuID, steps=steps)
elif model_id==2:
    from model.model_2order_ming import Network as Network2
    model = Network2(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==3:
    from model.model_3order_ming import Network as Network3
    model = Network3(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==4:
    from model.model_4order_ming import Network as Network4
    model = Network4(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==5:
    from model.model_2order_ming_unshare import Network as Network5
    model = Network5(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==6:
    from model.model_3order_ming_unshare import Network as Network6
    model = Network6(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==7:
    from model.model_4order_ming_unshare import Network as Network7
    model = Network7(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==8:
    from model.model_2order_tt import Network as Network8
    model = Network8(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==9:
    from model.model_3order_tt import Network as Network9
    model = Network9(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==10:
    from model.model_4order_tt import Network as Network10
    model = Network10(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==11:
    from model.model_2order_tr_share import Network as Network11
    model = Network11(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==12:
    from model.model_3order_tr_share import Network as Network12
    model = Network12(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==13:
    from model.model_4order_tr_share import Network as Network13
    model = Network13(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==14:
    from model.model_2order_tr_unshare import Network as Network14
    model = Network14(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)
elif model_id==15:
    from model.model_3order_tr_unshare import Network as Network15
    model = Network15(16, layer_num, L1_loss, gpuID=gpuID, steps=steps,rank=rank)


optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

model, _, _ = load_checkpoint(model, args.model_path, optimizer)


model = model.cuda(gpuID)
logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
print('Param:', utils.count_parameters_in_MB(model))

start = time.time()
with torch.no_grad():
    print('------------------------')
    for module in model.children():
        module.train(False)

    test_psnr = 0
    test_ssim = 0
    eps = 1e-10
    test_ite = 0
    for i, (input, target) in enumerate(test_dataloader):
        lr_patch = Variable(input.float(), requires_grad=False).cuda(gpuID)
        hr_patch = Variable(target.float(), requires_grad=False).cuda(gpuID)
        output = model(lr_patch)
        # Calculation of SSIM and PSNR values
        output = output.data.cpu().numpy()
        output[output > 1] = 1
        output[output < 0] = 0
        output = output.transpose((0,2, 3, 1))
        hr_patch = hr_patch.data.cpu().numpy()
        hr_patch[hr_patch > 1] = 1
        hr_patch[hr_patch < 0] = 0
        hr_patch = hr_patch.transpose((0, 2, 3, 1))
        # SSIM
        for index in range(output.shape[0]):
            test_ssim += ski_ssim(output[index], hr_patch[index], data_range=1, multichannel=True)
        # PSNR
        for index in range(output.shape[0]):
            imdf = (output[index] - hr_patch[index]) ** 2
            mse = np.mean(imdf) + eps
            test_psnr += 10 * math.log10(1.0 / mse)
            test_ite += 1
    test_psnr /= (test_ite)
    test_ssim /= (test_ite)
    print('Test PSNR: {:.4f}'.format(test_psnr))
    print('Test SSIM: {:.4f}'.format(test_ssim))
    print('------------------------')

    
