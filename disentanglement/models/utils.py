import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor

from itertools import cycle

transform_config = Compose([ToTensor()])

def reparametrize(mu, logvar):
    std = logvar.mul(.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)

def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0, .05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1, .02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0, .05)
        layer.bias.data.zero_()
