from .ResidualFlow import ResidualFlow
import torch.nn as nn
#from torch.nn.utils import spectral_norm
from ..SpectralNormGouk import spectral_norm
from .utilities import LipSwish
import warnings

def activation_by_name(name):
    if name.lower() == 'lipswish':
        ac = LipSwish
    elif name.lower() == 'leakyrelu':
        ac = nn.LeakyReLU
    elif name.lower() == 'relu':
        ac = nn.ReLU
    return ac


class Linear(ResidualFlow):
    def __init__(self, dim, hidden=None, n_hidden=2, lipschitz_constrain=0.9, mem_efficient=True, est_steps=5, activation_fn='LipSwish'):
        if hidden is None:
            hidden = dim * 8
            # for design hidden network width, see https://arxiv.org/abs/2010.15327
        ac = activation_by_name(activation_fn)
        
        block = [spectral_norm(nn.Linear(dim, hidden)),
                              ac()]
        for i in range(n_hidden - 1):
            block.append(spectral_norm(nn.Linear(hidden, hidden)))
            block.append(ac())
        
        block.append(spectral_norm(nn.Linear(hidden, dim)))
        block = nn.Sequential(*block)

        super(Linear, self).__init__(block, lipschitz_constrain=lipschitz_constrain, mem_efficient=mem_efficient, est_steps=est_steps)
        self.dim = dim
        self.hidden = hidden
        self.n_hidden = n_hidden
    
    def __repr__(self):
        return f'ResFlowLinear(dim={self.dim}, hidden={self.hidden}, n_hidden={self.n_hidden})'


class Conv1d(ResidualFlow):
    def __init__(self, in_feature, kernel_r, hidden=None, lipschitz_constrain=0.9, mem_efficient=True, est_steps=5, activation_fn='LipSwish'):
        if hidden is None:
            hidden = in_feature * 4
        ac = activation_by_name(activation_fn)

        k = 2 * kernel_r + 1
        padding = kernel_r

        block = nn.Sequential(spectral_norm(nn.Conv1d(in_feature, hidden, k, padding=padding)),
                              ac(),
                              spectral_norm(nn.Conv1d(hidden, in_feature, k, padding=padding)))
        super(Conv1d, self).__init__(block, lipschitz_constrain=lipschitz_constrain, mem_efficient=mem_efficient, est_steps=est_steps)
        self.in_feature = in_feature
        self.hidden = hidden
        self.kernel_r = kernel_r
    
    def __repr__(self):
        return f'ResFlowConv1d(in_feature={self.in_feature}, kernel_r={self.kernel_r}, hidden={self.hidden})'


class Conv2d(ResidualFlow):
    def __init__(self, in_feature, kernel_r, hidden=None, lipschitz_constrain=0.9, mem_efficient=True, est_steps=5, activation_fn='LipSwish'):
        if hidden is None:
            hidden = in_feature * 4
        ac = activation_by_name(activation_fn)

        k = 2 * kernel_r + 1
        padding = kernel_r

        block = nn.Sequential(spectral_norm(nn.Conv2d(in_feature, hidden, k, padding=padding)),
                              ac(),
                              spectral_norm(nn.Conv2d(hidden, in_feature, k, padding=padding)))
        super(Conv2d, self).__init__(block, lipschitz_constrain=lipschitz_constrain, mem_efficient=mem_efficient, est_steps=est_steps)
        self.in_feature = in_feature
        self.hidden = hidden
        self.kernel_r = kernel_r
    
    def __repr__(self):
        return f'ResFlowConv2d(in_feature={self.in_feature}, kernel_r={self.kernel_r}, hidden={self.hidden})'