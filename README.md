# High-order-OWAN

## Dataset
About 3 distortion dataset, please check https://github.com/sg-nm/Operation-wise-attention-network

About 5 distortion dataset, 

## Train
``` python
python3 main_all_model.py -g 1 -l 1 -r 16 -m_name 'num1_rank16' -m_id 2
```

## Test
``` python
python3 test.py -m_path './trained_model/best_model.pth.tar' -m_id 2 -l 10 -cn 'moderate'
```
