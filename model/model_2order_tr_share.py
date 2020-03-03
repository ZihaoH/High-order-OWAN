#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2 order, Tensor ring decomposition, share parameter

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from operations import *
import csv
import torch.optim as optim
import utils
import time

import pynvml
pynvml.nvmlInit()
## Operation layer
class OperationLayer(nn.Module):
    def __init__(self, C, stride, rank):
        super(OperationLayer, self).__init__()
        self._ops = nn.ModuleList()
        for o in Operations:
            op = OPS[o](C, stride, False)
            self._ops.append(op)
            
        self.op_num = len(Operations)

        self.w_cp1 = torch.nn.Parameter(torch.Tensor(rank, C*self.op_num,rank))

        self.w_out3 = torch.nn.Parameter(torch.randn(rank,C,rank))

        with torch.no_grad():
            self.w_cp1.normal_(0, 1/(C*self.op_num*rank))
            self.w_out3.normal_(0, 1/C)

        self.softsign = nn.Softsign()
        self.norm = nn.InstanceNorm2d(C)



    def forward(self, x, weights):
        weights = weights.transpose(1,0)
        states=[]
        for w, op in zip(weights, self._ops):
            states.append(op(x)*w.view([-1, 1, 1, 1]))
        states=torch.cat(states[:], dim=1)

        out1 = torch.einsum('achw,rck->arhwk', (states, self.w_cp1))
        out = torch.einsum('arhwk,akhwp,pom->arohwm', (out1,out1, self.w_out3))

        return self.softsign(self.norm(torch.einsum('arohwr->aohw',[out])))



## a Group of operation layers
class GroupOLs(nn.Module):
    def __init__(self, steps, C,rank):
        super(GroupOLs, self).__init__()
        self.preprocess = ReLUConv(C, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._ops = nn.ModuleList()
        self.relu = nn.ReLU()
        stride = 1

        for _ in range(self._steps):
            op = OperationLayer(C, stride,rank)
            self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess(s0)
        for i in range(self._steps):
            res = s0
            s0 = self._ops[i](s0, weights[:, i, :])
            s0 = self.relu(s0 + res)
        return s0

## Operation-wise Attention Layer (OWAL)
class OALayer(nn.Module):
    def __init__(self, channel, k, num_ops):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
                    nn.Linear(channel, self.output*2),
                    nn.ReLU(),
                    nn.Linear(self.output*2, self.k*self.num_ops))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y

## entire network (the number of layers = layer_num * steps)
class Network(nn.Module):
    def __init__(self, C, layer_num, criterion, steps=4, gpuID=0,rank=16):
        super(Network, self).__init__()
        self._C = C
        self._layer_num = layer_num
        self._criterion = criterion
        self._steps = steps
        self.gpuID = gpuID
        self.num_ops = len(Operations)
        
        self.kernel_size = 3
        # Feature Extraction Block
        self.FEB = nn.Sequential(nn.Conv2d(3, self._C, self.kernel_size, padding=1, bias=False),
                                  ResBlock(self._C, self._C, self.kernel_size, 1, 1, False),
                                  ResBlock(self._C, self._C, self.kernel_size, 1, 1, False),
                                  ResBlock(self._C, self._C, self.kernel_size, 1, 1, False),
                                  ResBlock(self._C, self._C, self.kernel_size, 1, 1, False),)
 
        # a stack of operation-wise attention layers
        self.layers = nn.ModuleList()
        for _ in range(self._layer_num):
            attention = OALayer(self._C, self._steps, self.num_ops)
            self.layers += [attention]
            layer = GroupOLs(steps, self._C,rank)
            self.layers += [layer]
        
        # Output layer
        self.conv2 = nn.Conv2d(self._C, 3, self.kernel_size, padding=1, bias=False)


    def forward(self, input):
        
        
        s0 = self.FEB(input)

        for _, layer in enumerate(self.layers):
            if isinstance(layer, OALayer):
                weights = layer(s0)
                weights = F.softmax(weights, dim=-1)
            else:
                s0 = layer(s0, weights)
                
        
        logits = self.conv2(s0)
        return logits

if __name__=='__main__':
    torch.manual_seed(2018)
    np.random.seed(2018)
    random_x=torch.Tensor(np.random.randn(8, 3,63, 63))
    random_y=random_x
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    L1_loss = nn.L1Loss().cuda()
    model = Network(16, 10, L1_loss,steps=4,rank=16).cuda()
    print('Param:', utils.count_parameters_in_MB(model))
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    lr_patch = Variable(random_x, requires_grad=False).cuda()
    hr_patch = Variable(random_y, requires_grad=False).cuda()

    start=time.time()
    for i in range(50):
        optimizer.zero_grad()
        output = model(lr_patch)
        l1_loss = L1_loss(output, hr_patch)
        l1_loss.backward()
        optimizer.step()
        train_loss = l1_loss.item()
        print(i,':',train_loss)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(meminfo.used/1024/1024/1024)
    print('time:',(time.time()-start)*230080/32/50)
