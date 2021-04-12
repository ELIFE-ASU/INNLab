import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from SpectralNormGouk import spectral_norm as spectral_norm_g

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


class SNCov1d(nn.Module):
    '''
    Spectrum Normalized 1-d Conv Layer stack
    '''
    def __init__(self, channel, kernel_size, w=8, k=0.8):
        super(SNCov1d, self).__init__()
        if kernel_size % 2 != 1:
            raise Exception(f'The kernel_size must be an odd number, but got {kernel_size}.')
        
        padding = (kernel_size - 1) // 2
        self.channel = channel
        self.kernel_size = kernel_size
        self.k = k
        
        self.net = nn.Sequential(spectral_norm_g(nn.Conv1d(channel, w * channel, kernel_size=kernel_size, padding=padding)),
                                 nn.GELU(),
                                 spectral_norm_g(nn.Conv1d(w * channel, w * channel, kernel_size=kernel_size, padding=padding)),
                                 nn.GELU(),
                                 spectral_norm_g(nn.Conv1d(w * channel, channel, kernel_size=kernel_size, padding=padding))
                                )
        
        self._initialize()
    
    def _initialize(self):
        self.forward(torch.randn((2, self.channel, self.kernel_size))) # doing one compuatation to initialize the spectral_norm
        return
    
    def forward(self, x):
        x = self.net(self.k * x)
        return x


class SNCov2d(nn.Module):
    '''
    Spectrum Normalized 1-d Conv Layer stack
    '''
    def __init__(self, channel, kernel_size, w=8, k=0.8):
        super(SNCov2d, self).__init__()
        if kernel_size % 2 != 1:
            raise Exception(f'The kernel_size must be an odd number, but got {kernel_size}.')
        
        padding = (kernel_size - 1) // 2
        self.channel = channel
        self.kernel_size = kernel_size
        self.k = k
        
        self.net = nn.Sequential(spectral_norm_g(nn.Conv2d(channel, w * channel, kernel_size=kernel_size, padding=padding)),
                                 nn.GELU(),
                                 spectral_norm_g(nn.Conv2d(w * channel, w * channel, kernel_size=kernel_size, padding=padding)),
                                 nn.GELU(),
                                 spectral_norm_g(nn.Conv2d(w * channel, channel, kernel_size=kernel_size, padding=padding))
                                )
        
        self._initialize()
    
    def _initialize(self):
        self.forward(torch.randn((2, self.channel, self.kernel_size, self.kernel_size))) # doing one compuatation to initialize the spectral_norm
        return
    
    def forward(self, x):
        x = self.net(self.k * x)
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