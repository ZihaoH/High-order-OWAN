# High-order-OWAN
https://arxiv.org/abs/2001.10853
## Dataset
About 3 distortion dataset, please check https://github.com/sg-nm/Operation-wise-attention-network.

About 5 distortion dataset, the raw data can be found in https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K.
It contains 861 image pairs for training, 58 for validation, and 239 for testing. And the way of generation dataset is similar to above, but using generate_train_5noise.m instand of generate_train.m.

## Train
``` python
python3 main_all_model.py -g 1 -l 1 -r 16 -m_name 'num1_rank16' -m_id 2
```

## Test
``` python
python3 test.py -m_path './trained_model/best_model.pth.tar' -m_id 2 -l 10 -cn 'moderate'
```
The trained model parameters in "trained_model" are for 3 distortion dataset. And these two files are all for 2-order H-OWAN which have CP structure, 10 blocks and shares parameters. `./trained_model/best_model.pth.tar` is for `./model/model_2order_ming.py`, and `/trained_model/best_model_function_version.tar` is for `./model/model_2order_test_conv.py`. These two model structures are the same, but the second one uses `tensor1x1_conv.py`. Although the first one also uses tensor 1x1 conv, the tensor 1x1 conv method is not encapsulated as a function. But they do the same performance.
