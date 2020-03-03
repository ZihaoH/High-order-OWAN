
# we borrowed a part of the following code:
# https://github.com/yuke93/RL-Restore

import numpy as np
import os
import h5py
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
#import cv2
from PIL import Image


def load_imgs(list_in, list_gt, size = 63):
    assert len(list_in) == len(list_gt)
    img_num = len(list_in)
    imgs_in = np.zeros([img_num, size, size, 3])
    imgs_gt = np.zeros([img_num, size, size, 3])
    for k in range(img_num):
        imgs_in[k, ...] = np.array(Image.open(list_in[k]).convert('RGB')) / 255.
        imgs_gt[k, ...] = np.array(Image.open(list_gt[k]).convert('RGB')) / 255.
    return imgs_in, imgs_gt

def data_reformat(data):
    """RGB <--> BGR, swap H and W"""
    assert data.ndim == 4
    out = data[:, :, :, ::-1] - np.zeros_like(data)
    out = np.swapaxes(out, 1, 2)
    out = out.astype(np.float32)
    return out

def get_dataset_deform(train_root,val_root,test_root,is_train,class_name='moderate'):
    dataset = DeformedData(
        train_root=train_root,
        val_root=val_root,
        test_root=test_root,
        is_train=is_train,
        transform=transforms.Compose([transforms.ToTensor()]),
        target_transform=transforms.Compose([transforms.ToTensor()]),
        class_name=class_name,
    )
    return dataset

class DeformedData(data.Dataset):
    def __init__(self, train_root, val_root, test_root, is_train=0, transform=None, target_transform=None,class_name='moderate'):
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        self.train_dir = train_root
        self.val_dir = val_root
        self.test_dir = test_root

        if self.is_train == 0:
            # training dataset
            self.train_list = [self.train_dir + file for file in os.listdir(self.train_dir) if file.endswith('.h5')]
            self.train_cur = 0
            self.train_max = len(self.train_list)
            f = h5py.File(self.train_list[self.train_cur], 'r')
            self.data = f['data'][()]
            self.label = f['label'][()]
            f.close()
            self.data_index = 0
            self.data_len = len(self.data)
            print('training images:', self.data_len)
        elif self.is_train == 1:
            # validation dataset
            f = h5py.File(self.val_dir + os.listdir(self.val_dir)[0], 'r')
            self.data = f['data'][()]
            self.label = f['label'][()]
            f.close()
            self.data_index = 0
            self.data_len = len(self.data)
        elif self.is_train == 2:
            # # test dataset
            self.test_in = self.test_dir + class_name + '_in/'
            self.test_gt = self.test_dir + class_name + '_gt/'
            list_in = [self.test_in + name for name in os.listdir(self.test_in)]
            list_in.sort()
            list_gt = [self.test_gt + name for name in os.listdir(self.test_gt)]
            list_gt.sort()
            self.name_list = [os.path.splitext(os.path.basename(file))[0] for file in list_in]
            self.data_all, self.label_all = load_imgs(list_in, list_gt)
            self.test_total = len(list_in)
            self.test_cur = 0
            # dataset reformat, because the dataset for tools training are in a different format
            self.data = self.data_all
            self.label = self.label_all
            self.data_index = 0
            self.data_len = len(self.data)
        else:
            print("not implement yet")
            sys.exit()


    def __getitem__(self, index):
        img = self.data[index]
        img_gt = self.label[index]

        # transforms (numpy -> Tensor)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            img_gt = self.target_transform(img_gt)
        return img, img_gt

    def __len__(self):
        return self.data_len
