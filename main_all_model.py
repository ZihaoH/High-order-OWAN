#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import pandas as pd
from cnn_train_all_model import CNN_train


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='High-order Operation-wise Attention Network')
    parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
    parser.add_argument('--layer_num', '-l', type=int, default=1, help='Num. of Block')
    parser.add_argument('--rank', '-r', type=int, default=16, help='Num. of rank')
    parser.add_argument('--mode', '-m', default='mix', help='Mode (mix / yourdata)')
    parser.add_argument('--model_name', '-m_name', default='HOWAN', help='model name file')
    parser.add_argument('--model_id', '-m_id', type=int, default=2, help='model id')
    args = parser.parse_args()

    # --- Optimization of the CNN architecture ---

    if args.mode == 'mix':
        cnn = CNN_train(args.mode, imgSize=63, batchsize=32)
        acc = cnn(None, 0, epoch_num=100, gpu_num=args.gpu_num,result_file=args.model_name,rank=args.rank,layer_num=args.layer_num,steps=4,model_id=args.model_id)
    elif args.mode == 'yourdata':
        cnn = CNN_train(args.mode, imgSize=128, batchsize=32)
        acc = cnn(None, 0, epoch_num=150, gpu_num=args.gpu_num)
