# Subspace Attack Reproduction

Attempt to reproduce the NeurIPS 2019 paper [Subspace Attack: Exploiting Promising Subspaces for Query-Efficient Black-box Attacks](https://papers.nips.cc/paper/8638-subspace-attack-exploiting-promising-subspaces-for-query-efficient-black-box-attacks).

The original code of the paper can be found [here](https://github.com/ZiangYan/subspace-attack.pytorch). We are trying to reproduce the attack to GDAS model trained on CIFAR-10 dataset, without using and looking at the original code.

This project is done as project for the [CS-433 Machine Learning Course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) at [EPFL](https://epfl.ch/en), and as part of the [NeurIPS 2019 Reproducibility Challenge](https://reproducibility-challenge.github.io/neurips2019/).

Pretrained models in use can be downloaded [here](https://drive.google.com/file/d/1TA-UWYVDkCkNPOy1INjUU9321s-HA6RF/view?usp=sharing). They are a subset of the [models](https://drive.google.com/file/d/1aXTmN2AyNLdZ8zOeyLzpVbRHZRZD0fW0/view?usp=sharing) provided with the code of the original paper. They need to be unzipped and put in the `pretrained/` folder, in the root directory of the repo.

The dataset ([CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)) is automatically downloaded via `torchvision.datasets` when first running the experiment, and will be saved in the `data/` folder (more info [here](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar)).

The paper is implemented and tested using Python 3.7. Dependencies are listed in [requirements.txt](requirements.txt).

For the moment it is possible to run the experiment using [VGG nets](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) and  [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) as reference models and [GDAS](https://arxiv.org/pdf/1910.04465.pdf) as victim model. In order to run the experiment, simply run in a terminal

```bash
python subspace-attack.py
```
