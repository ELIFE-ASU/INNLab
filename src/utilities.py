import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# compute v.Jacobian, source: https://github.com/jarrelscy/iResnet
def vjp(ys, xs, v):
    vJ = torch.autograd.grad(ys, xs, grad_outputs=v, create_graph=True, retain_graph=True, allow_unused=True)
    return tuple([j for j in vJ])


class SNFCN(nn.Module):
    '''
    spectral normalized fully connected function
    '''
    def __init__(self, dim, w=8, k=0.8, batch_norm=False):
        super(SNFCN, self).__init__()
        self.k = k
        self.dim = dim
        self.g = nn.Sequential(spectral_norm(nn.Linear(dim, w*dim)), nn.GELU(),
                                spectral_norm(nn.Linear(w*dim, w*dim)), nn.GELU(),
                                spectral_norm(nn.Linear(w*dim, dim))
                                )
        
        self._initialize()
    
    def _initialize(self):
        self.forward(torch.randn((2, self.dim))) # doing one compuatation to initialize the spectral_norm
        return
    
    def forward(self, x):
        x = self.g(self.k * x)
        
        return x

class NormalDistribution(nn.Module):
    '''
    Generate normal distribution and compute log probablity
    '''
    def __init__(self):
        super(NormalDistribution, self).__init__()
    
    def logp(self, x):
        return torch.sum(-1 * (x ** 2), dim=-1)
    
    def sample(self, shape):
        return torch.randn(shape)

    def forward(self, x):
        x = self.logp(x)
        
        return x