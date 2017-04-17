# Chainer-StackGAN
Chainer implementation of StackGAN on CUB
>"StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks," Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaolei Huang, Xiaogang Wang, Dimitris Metaxas, arXiv:1612.03242.
https://arxiv.org/abs/1612.03242

# todo
- multi gpu

# Requirements
- Python 2.7.6+
  - [Chainer 1.22.0+](https://github.com/pfnet/chainer)
  - numpy 1.12.1+
  - scipy 0.13.3+

# Example result
## CUB
- stage1
![stage1_test](./result/stage1/test_600.jpg)
- stage2
![stage2_test](./result/stage2/test_600.jpg)

# Test
学習済みweight google driveで
置き場は `pretrained/`

```python
cd stage2
python demo.py -g=0
```

# Dataset
データセットの準備はオリジナルのコード見てね
https://github.com/hanzhanggit/StackGAN
```python
python StackGAN/misc/preprocess_birds.py
```

# Train
```python
cd stage1
python train_stackgan.py -g=0 
cd stage2
python train_stackgan.py -g=0 -s1w=path/to/stage1/gen_epoch_600.npz
```
