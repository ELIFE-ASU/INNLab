import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np


class LipSwish(nn.Module):
    def __init__(self):
        r'''LipSwish activation function.
        See details in https://arxiv.org/abs/1906.02735
        '''
        super(LipSwish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))
    
    def forward(self, x):
        return (x * torch.sigmoid(x * F.softplus(self.beta))).div_(1.1)


class LeakyLipSwish(LipSwish):
    def __init__(self, negative_slope=0.1):
        super(LeakyLipSwish, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x):
        s = self.negative_slope
        return (x * (torch.sigmoid(x * F.softplus(self.beta)) + s)).div_(1.1) / (1 + s)


def vjp(ys, xs, v):
    vJ = torch.autograd.grad(ys, xs, grad_outputs=v, create_graph=True, retain_graph=True, allow_unused=True)
    return tuple([j for j in vJ])

def Linear(in_channels, out_channels, bias=True):
    block = nn.Linear(in_channels, out_channels, bias)
    nn.init.kaiming_normal_(block.weight.data, nonlinearity='relu')
    if bias:
        block.bias.data.zero_()
    block = spectral_norm(block)
    # pre-forward
    block(torch.randn(1, in_channels))
    return block


class ShiftedGeometric:
    r'''
    Shift geometric distribution

    providing sampling, CDF, and parameters from mean
    '''
    @staticmethod
    def sample(p):
        """
        Sample from a shifted geometric distribution
        The orginal numpy function name may be misleading,
        that is a shifted geometric distribution
        """
        return np.random.geometric(p)
    
    @staticmethod
    def CDF(p, k):
        r'''Cumulative distribution function of shifted geometric distribution'''
        if k >= 1:
            return 1 - (1 - p) ** k
        else:
            return 0
    
    @staticmethod
    def para_from_mean(mean):
        r'''Get parameters from mean'''
        return 1.0 / mean