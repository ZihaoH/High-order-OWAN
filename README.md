# High-order-OWAN
https://arxiv.org/abs/2001.10853
## Dataset
About 3 distortion dataset, please check https://github.com/sg-nm/Operation-wise-attention-network

About 5 distortion dataset, the raw data can be found in https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K
It contains 861 image pairs for training, 58 for validation, and 239 for testing. And the way of generation dataset is similar to above, but using generate_train_5noise.m instand of generate_train.m.

## Train
``` python
python3 main_all_model.py -g 1 -l 1 -r 16 -m_name 'num1_rank16' -m_id 2
```

## Test
``` python
python3 test.py -m_path './trained_model/best_model.pth.tar' -m_id 2 -l 10 -cn 'moderate'
```
