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
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from skimage.measure import compare_psnr as ski_psnr
from skimage.measure import compare_ssim as ski_ssim
import os
import csv
import logging

import torch.nn.functional as F
from data_load_own import get_training_set, get_test_set
from data_load_mix import get_dataset_deform
import utils



def load_checkpoint(model, checkpoint_PATH, optimizer, last_epoch=0):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!' + checkpoint_PATH)
        optimizer.load_state_dict(model_CKPT['optimizer'])
        last_epoch = model_CKPT['epoch']
    return model, optimizer, last_epoch


class CNN_train():
    def __init__(self, dataset_name, imgSize=63, batchsize=32):
        self.imgSize = imgSize
        self.batchsize = batchsize
        self.dataset_name = dataset_name

        # load dataset
        if dataset_name == 'mix' or dataset_name == 'yourdata':
            if dataset_name == 'mix':
                self.num_work = 8
                train_dir = '../'
                val_dir = '../test_images/'
                test_dir = '../test_images/'
    
                train_set = get_dataset_deform(train_dir, val_dir, test_dir, 0)

                val_set = get_dataset_deform(train_dir, val_dir, test_dir, 2)

                #test_set = get_dataset_deform(train_dir, val_dir, test_dir, 2)
                

                self.dataloader = DataLoader(dataset=train_set, num_workers=self.num_work, batch_size=self.batchsize,
                                             shuffle=True, pin_memory=True)

                self.val_loader = DataLoader(dataset=val_set, num_workers=self.num_work, batch_size=1, shuffle=False, pin_memory=False)

                #self.test_set_loader = DataLoader(dataset=test_set, num_workers=self.num_work,batch_size=1, shuffle=False, pin_memory=False)
                
            elif dataset_name == 'yourdata':
                self.num_work = 8
                # Specify the path of your dataset
                train_input_dir = '/dataset/yourdata_train/input/'
                train_target_dir = '/dataset/yourdata_train/target/'
                test_input_dir = '/dataset/yourdata_test/input/'
                test_target_dir = '/dataset/yourdata_test/target/'
                train_set = get_training_set(train_input_dir, train_target_dir, True)
                test_set = get_training_set(test_input_dir, test_target_dir, False)
                self.dataloader = DataLoader(dataset=train_set, num_workers=self.num_work, batch_size=self.batchsize,
                                             shuffle=True, drop_last=True)
                self.test_dataloader = DataLoader(dataset=test_set, num_workers=self.num_work, batch_size=1,
                                                  shuffle=False)
        else:
            print('\tInvalid input dataset name at CNN_train()')
            exit(1)

    def __call__(self, cgp, gpuID, epoch_num=150, gpu_num=1, result_file='results', rank=4, layer_num=1, steps=4,model_id=1):
        print('GPUID    :', gpuID)
        print('epoch_num:', epoch_num)

        # define model
        torch.manual_seed(2018)
        torch.cuda.manual_seed(2018)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        L1_loss = nn.L1Loss()
        L1_loss = L1_loss.cuda(gpuID)
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



        if gpu_num > 1:
            device_ids = [i for i in range(gpu_num)]
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda(gpuID)
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        print('Param:', utils.count_parameters_in_MB(model))
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

        # for output images
        if not os.path.exists(result_file):
            os.makedirs('./'+result_file+'/Inputs')
            os.makedirs('./'+result_file+'/Outputs')
            os.makedirs('./'+result_file+'/Targets')
            os.makedirs(result_file + '/Models')

        checkpoint_path = None
        model_dir = './'+ result_file + '/Models/'
        model_list = os.listdir(model_dir)
        flag = 0
        for num in range(100, 0, -1):
            for model_name in model_list:
                if str(num) in model_name:
                    checkpoint_path = model_dir + model_name
                    flag = 1
                    break
            if flag == 1:
                break

        model, optimizer, last_epoch = load_checkpoint(model, checkpoint_path, optimizer)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
        test_interval = 5

        # Train loop
        for epoch in range(1, epoch_num + 1):
            scheduler.step()

            if epoch <= last_epoch:
                continue

            start_time = time.time()
            print('epoch', epoch)
            train_loss = 0
            for module in model.children():
                module.train(True)
            for ite, (input, target) in enumerate(self.dataloader):
                lr_patch = Variable(input, requires_grad=False).cuda(gpuID)
                hr_patch = Variable(target, requires_grad=False).cuda(gpuID)
                optimizer.zero_grad()
                output = model(lr_patch)
                l1_loss = L1_loss(output, hr_patch)
                l1_loss.backward()
                optimizer.step()
                train_loss += l1_loss.item()

                if ite % 500 == 0:
                    vutils.save_image(lr_patch.data, './'+result_file+'/input_sample%d.png' % gpuID, normalize=False)
                    vutils.save_image(hr_patch.data, './'+result_file+'/target_sample%d.png' % gpuID, normalize=False)
                    vutils.save_image(output.data, './'+result_file+'/output_sample%d.png' % gpuID, normalize=False)

            print('Train set : Average loss: {:.4f}'.format(train_loss))
            print('time ', time.time() - start_time)

            # check val/test performance
            if epoch % test_interval == 0:
                with torch.no_grad():
                    print('------------------------')
                    for module in model.children():
                        module.train(False)
                    # mild
                    test_psnr = 0
                    test_ssim = 0
                    eps = 1e-10
                    test_ite = 0
                    for i, (input, target) in enumerate(self.val_loader):
                        lr_patch = Variable(input.float(), requires_grad=False).cuda(gpuID)
                        hr_patch = Variable(target.float(), requires_grad=False).cuda(gpuID)
                        output = model(lr_patch)
                        # save images
                        vutils.save_image(output.data, './'+result_file+'/Outputs/%05d.png' % (int(i)), padding=0, normalize=False)
                        vutils.save_image(lr_patch.data, './'+result_file+'/Inputs/%05d.png' % (int(i)), padding=0, normalize=False)
                        vutils.save_image(hr_patch.data, './'+result_file+'/Targets/%05d.png' % (int(i)), padding=0, normalize=False)
                        # Calculation of SSIM and PSNR values
                        output = output.data.cpu().numpy()[0]
                        output[output > 1] = 1
                        output[output < 0] = 0
                        output = output.transpose((1, 2, 0))
                        hr_patch = hr_patch.data.cpu().numpy()[0]
                        hr_patch[hr_patch > 1] = 1
                        hr_patch[hr_patch < 0] = 0
                        hr_patch = hr_patch.transpose((1, 2, 0))
                        # SSIM
                        test_ssim += ski_ssim(output, hr_patch, data_range=1, multichannel=True)
                        # PSNR
                        imdf = (output - hr_patch) ** 2
                        mse = np.mean(imdf) + eps
                        test_psnr += 10 * math.log10(1.0 / mse)
                        test_ite += 1
                    test_psnr /= (test_ite)
                    test_ssim /= (test_ite)
                    print('Valid PSNR: {:.4f}'.format(test_psnr))
                    print('Valid SSIM: {:.4f}'.format(test_ssim))

                    print('------------------------')

                    

                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           './' + result_file + '/Models/model_%d.pth.tar' % int(epoch))

        return train_loss
