# [CVPR 2023] Class-wise Calibrated Fair Adversarial Training

Author: [Zeming Wei](https://weizeming.github.io), [Yifei Wang](https://yifeiwang.me), [Yiwen Guo](https://yiwenguo.github.io), [Yisen Wang](https://yisenwang.github.io)


This repo is under construction.

## Checkpoints
The model checkpoints for {AT, TRADES, FAT}+CFA are available at [this url](https://drive.google.com/drive/folders/1uHJTVmZ4EgDqXoShbjgwJfCRFYPFeq_F?usp=sharing).

## Train with CFA

#### Split the validation set
1. Edit ``PATH`` in ``generate_validate.py`` (line 1) as your data path to CIFAR-10 dataset.
2. Run ``python generate_validate.py``

#### Train AT+CFA
An example code for AT+CFA (you can try other hyper-parameters):
```
python train.py --mode 'AT' --fname 'AT_CFA' --ccm --lambda-1 0.5 --threshold 0.2 
```



## Citation
```
@InProceedings{Wei_2023_CVPR,
    author    = {Wei, Zeming and Wang, Yifei and Guo, Yiwen and Wang, Yisen},
    title     = {CFA: Class-Wise Calibrated Fair Adversarial Training},
    booktitle = {CVPR},
    year      = {2023}
}
```
