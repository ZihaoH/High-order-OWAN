#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

class tensor1x1conv_CP(nn.Module):
    def __init__(self, C_in, C_out, rank=16, order=2, share_core = True, add1=False):
        super(tensor1x1conv_CP, self).__init__()

        self.order=order
        self.share_core=share_core
        self.add1=add1

        if add1:
            C_in+=1
        if share_core:
            self.w_core = torch.nn.Parameter(torch.Tensor(rank, C_in, C_out))
        else:
            self.w_core = torch.nn.Parameter(torch.Tensor(order,rank, C_in, C_out))

        self.w_out = torch.nn.Parameter(torch.randn(rank))

        with torch.no_grad():
            self.w_core.normal_(0, 1/(C_in))
            self.w_out.normal_(0, 1/rank)


        self.softsign = nn.Softsign()
        self.norm = nn.InstanceNorm2d(C_out)

    def forward(self, x):
        if self.add1:
            if x.is_cuda:
                one = torch.ones(x.shape[0],1,x.shape[2],x.shape[3]).cuda()
            else:
                one = torch.ones(x.shape[0],1,x.shape[2],x.shape[3])
            x=torch.cat([x,one], dim=1)

        cmd_str = 'arohw,'*self.order+'r->aohw'
        if self.share_core:
            out = torch.einsum('achw,rco->arohw', (x, self.w_core))
            tmp = [out]*self.order
            out = torch.einsum(cmd_str,(*tmp,self.w_out))
        else:
            tmp=[]
            for i in range(self.order):
                tmp.append(torch.einsum('achw,rco->arohw', (x, self.w_core[i])))
            out = torch.einsum(cmd_str,(*tmp,self.w_out))
        return self.softsign(self.norm(out))


class tensor1x1conv_TR(nn.Module):
    def __init__(self, C_in, C_out, rank=16, order=2, share_core = True, add1=False):
        super(tensor1x1conv_TR, self).__init__()

        self.order=order
        self.share_core=share_core
        self.add1=add1

        if add1:
            C_in+=1
        if share_core:
            self.w_core = torch.nn.Parameter(torch.Tensor(rank, C_in, rank))
        else:
            self.w_core = torch.nn.Parameter(torch.Tensor(order,rank, C_in, rank))

        self.w_out = torch.nn.Parameter(torch.randn(rank, C_out, rank))

        with torch.no_grad():
            self.w_core.normal_(0, 1/(C_in*rank))
            self.w_out.normal_(0, 1/rank)


        self.softsign = nn.Softsign()
        self.norm = nn.InstanceNorm2d(C_out)

    def forward(self, x):
        if self.add1:
            if x.is_cuda:
                one = torch.ones(x.shape[0],1,x.shape[2],x.shape[3]).cuda()
            else:
                one = torch.ones(x.shape[0],1,x.shape[2],x.shape[3])
            x=torch.cat([x,one], dim=1)

        all_letter=list('efghijklmnopqrstuvwxyz')
        need_letter = all_letter[:self.order+1]
        cmd_str = 'a{}bc{},'*self.order+'{}d{}->a{}dbc{}'
        double_letter=[]
        for i in range(self.order+1):
            double_letter.append(need_letter[i])
            double_letter.append(need_letter[i])
        double_letter.pop(0)
        double_letter.append(all_letter[self.order+1])
        cmd_str=cmd_str.format(*double_letter,need_letter[0],all_letter[self.order+1])

        if self.share_core:
            out = torch.einsum('achw,rck->arhwk', (x, self.w_core))
            tmp = [out]*self.order
            out = torch.einsum(cmd_str,(*tmp,self.w_out))
        else:
            tmp=[]
            for i in range(self.order):
                tmp.append(torch.einsum('achw,rck->arhwk', (x, self.w_core[i])))
            out = torch.einsum(cmd_str,(*tmp,self.w_out))
        out = torch.einsum('arohwr->aohw',[out])
        return self.softsign(self.norm(out))


class tensor1x1conv_TT(nn.Module):
    def __init__(self, C_in, C_out, rank=16, order=2, add1=False):
        super(tensor1x1conv_TT, self).__init__()

        self.order=order
        self.add1=add1

        if add1:
            C_in+=1

        self.w_core_head = torch.nn.Parameter(torch.Tensor(C_in, rank))
        self.w_core_body = torch.nn.Parameter(torch.Tensor(order-1,rank, C_in, rank))

        self.w_out = torch.nn.Parameter(torch.randn(rank, C_out))

        with torch.no_grad():
            self.w_core_head.normal_(0, 1/(C_in))
            self.w_core_body.normal_(0, 1/C_out)
            self.w_core_body.normal_(0, 1/(C_in*rank))


        self.softsign = nn.Softsign()
        self.norm = nn.InstanceNorm2d(C_out)

    def forward(self, x):
        if self.add1:
            if x.is_cuda:
                one = torch.ones(x.shape[0],1,x.shape[2],x.shape[3]).cuda()
            else:
                one = torch.ones(x.shape[0],1,x.shape[2],x.shape[3])
            x=torch.cat([x,one], dim=1)

        all_letter=list('efghijklmnopqrstuvwxyz')
        need_letter = all_letter[:self.order+1]
        cmd_str = 'abc{},'+'a{}bc{},'*(self.order-1)+'{}d->adbc'
        double_letter=[]
        for i in range(self.order+1):
            double_letter.append(need_letter[i])
            double_letter.append(need_letter[i])
        double_letter.pop(-1)
        cmd_str=cmd_str.format(*double_letter)
        print(cmd_str)

        out = torch.einsum('achw,cr->ahwr', (x, self.w_core_head))
        tmp=[]
        for i in range(self.order-1):
            tmp.append(torch.einsum('achw,rck->arhwk', (x, self.w_core_body[i])))


        out = torch.einsum(cmd_str,(out, *tmp,self.w_out))

        return self.softsign(self.norm(out))


class Tensor1x1Conv(nn.Module):
    def __init__(self, C_in, C_out, rank=16, order=2, share_core = True, add1=False, TN_format = 'CP'):
        '''
        :param C_in: input channel
        :param C_out: output channel
        :param rank: the rank in the tensor network decomposition
        :param order: the order of tensor product
        :param share_core: if share the core tensor in CP and TR
        :param add1: if oncatenating the feature map with a constant of 1 at the end
        :param TN_format: choose one from [CP, TT, TR]
        '''
        super(Tensor1x1Conv, self).__init__()
        if order<2:
            raise('Order must be greater than 2!')
        if TN_format == 'CP':
            self.t1c = tensor1x1conv_CP(C_in, C_out, rank=rank, order=order, share_core = share_core, add1=add1)
        elif TN_format == 'TR':
            self.t1c = tensor1x1conv_TR(C_in, C_out, rank=rank, order=order, share_core = share_core, add1=add1)
        elif TN_format == 'TT':
            self.t1c = tensor1x1conv_TT(C_in, C_out, rank=rank, order=order, add1=add1)
        else:
            raise('Undefined tensor network format!')

    def forward(self,x):
        return self.t1c(x)


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch.optim as optim
    import time


    random_x=torch.Tensor(np.random.randn(32, 16,63, 63))
    random_y=torch.Tensor(np.random.randn(32, 4,63, 63))

    tconv=Tensor1x1Conv(16,4, rank=16, order=4,  share_core = True,add1=True,TN_format='CP').cuda()

    L1_loss = nn.L1Loss().cuda()
    
    optimizer = optim.Adam(tconv.parameters(), lr=0.001, betas=(0.9, 0.999))

    lr_patch = Variable(random_x, requires_grad=False).cuda()
    hr_patch = Variable(random_y, requires_grad=False).cuda()

    start=time.time()
    for i in range(50):
        optimizer.zero_grad()
        output = tconv(lr_patch)
        l1_loss = L1_loss(output, hr_patch)
        l1_loss.backward()
        optimizer.step()
        train_loss = l1_loss.item()
        print(i,':',train_loss)