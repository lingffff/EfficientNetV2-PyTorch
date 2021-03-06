# EfficientNetV2-PyTorch
This is a brief PyTorch implementation of EfficientNetV2[[paper](https://arxiv.org/abs/2104.00298)], providing with experiments on ImageNet and CIFAR-10/100.  
The official TensorFlow implementation is at [google/automl/efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2).  

## Run
### 1. Train
```bash
sh prepare.sh
```
```bash
python train.py efficientnetv2-b0 cifar10
```
if using Distributed Data Parallel Training:  
```bash
python train.py efficientnetv2-b0 cifar10 --ddp
```
### 2. Test
```bash
python eval.py efficientnetv2-b0 cifar10 --ckpt weights/best.pth.tar
```

## Acknowledgement
With sincerely appreciation to [google/automl/efficientnetv2](https://github.com/google/automl/tree/master/efficientnetv2) and [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)!
## Cites
```
@article{tan2021efficientnetv2,
  title={Efficientnetv2: Smaller models and faster training},
  author={Tan, Mingxing and Le, Quoc V},
  journal={arXiv preprint arXiv:2104.00298},
  year={2021}
}
```
