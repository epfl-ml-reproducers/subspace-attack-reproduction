'''GDAS net.
Ported form
https://github.com/D-X-Y/GDAS
(c) Yuanyi Dong
'''
import os
import os.path as osp
import torch

from src.models.cifar.gdas.lib.scheduler import load_config
from src.models.cifar.gdas.lib.scheduler import load_config
from src.models.cifar.gdas.lib.nas import model_types
from src.models.cifar.gdas.lib.nas import NetworkCIFAR

__all__ = ['gdas']


def load_gdas(checkpoint_fname, config_position):
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    xargs = checkpoint['args']
    config = load_config(config_position)
    genotype = model_types[xargs.arch]
    class_num = 10

    model = NetworkCIFAR(xargs.init_channels, class_num, xargs.layers, config.auxiliary, genotype)
    model.load_state_dict(checkpoint['state_dict'])
    return model
