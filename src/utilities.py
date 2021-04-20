import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
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
        logps = -1 * (x ** 2)

        if len(x.shape) == 1:
            # linear layer
            raise Exception(f'The input must have a batch dimension, but got dim={x.shape}.')
        if len(x.shape) == 2:
            # [batch, dim]
            return logps.sum(dim=-1)
        if len(x.shape) == 3:
            # [batch, channel, dim_1d], 1d conv
            return logps.reshape(x.shape[0], -1).sum(dim=-1)
        if len(x.shape) == 4:
            # [batch, channel, dim_x, dim_y], 2d conv
            return logps.reshape(x.shape[0], -1).sum(dim=-1)
        
        raise Exception(f'The input dimension should be 1,2,3, or 4, but got {len(x.shape)}.')
    
    def sample(self, shape):
        return torch.randn(shape)

    def forward(self, x):
        x = self.logp(x)
        
        return x

def permutation_matrix(dim):
    # generate a permuation matrix
    x = torch.zeros((dim, dim))
    for i in range(dim):
        x[i, (i+1) % (dim)] = 1
    return x

class InvertibleLinear(nn.Module):
    '''
    Invertible Linear
    ref: https://arxiv.org/pdf/1807.03039.pdf 3.2
    '''
    def __init__(self, dim):
        super(InvertibleLinear, self).__init__()
        self.P = permutation_matrix(dim)

        self.L = nn.Parameter(self.get_L(dim) / (dim))
        self.U = nn.Parameter(self.get_U(dim) / (dim))
        self.log_s = nn.Parameter(torch.zeros(dim))

    def get_L(self, dim):
        L = torch.tril(torch.randn(dim, dim))
        for i in range(dim):
            L[i,i] = 1
        return L

    def get_U(self, dim):
        U = torch.triu(torch.randn(dim, dim))
        for i in range(dim):
            U[i, i] = 0
        return U
    
    def W(self):
        return self.P @ self.L @ (self.U + torch.diag(torch.exp(self.log_s)))
    
    def inv_W(self):
        # need to be optimized based on the LU decomposition
        w = self.W()
        inv_w = torch.inverse(w)
        return inv_w
    
    def logdet(self):
        return torch.sum(self.log_s)

    def forward(self, x):
        weight = self.W()
        return F.linear(x, weight)
    
    def inverse(self, y):
        return F.linear(y, self.inv_W())


class real_nvp_element(nn.Module):
    '''
    The very basic element of real nvp
    '''
    def __init__(self, dim, f_log_s, f_t, mask=None, eps=1e-8, clip=None):
        super(real_nvp_element, self).__init__()

        if mask is None:
            self.mask = self.generate_mask(dim)
        else:
            self.mask = mask
        
        self.f_log_s = f_log_s
        self.f_t = f_t
        self.eps = eps
        self.clip = clip
    
    def generate_mask(self, dim):
        '''
        generate mask for given dimension number `dim`
        '''
        mask = torch.zeros((1, dim))
        for i in range(dim):
            if i % 2 == 0:
                mask[0, i] = 1
        return mask
    
    def get_s(self, x):
        if len(x.shape) == 1:
            b = self.mask.squeeze().to(x.device)
        else:
            b = self.mask.to(x.device)
        
        log_s = self.f_log_s(b * x)

        if self.clip is not None:
            # clip the log(s), to avoid extremely large numbers
            log_s = self.clip * torch.tanh(log_s / self.clip)
        
        s = torch.exp(log_s)
        return s, log_s

    def forward(self, x):
        if len(x.shape) == 1:
            b = self.mask.squeeze().to(x.device)
        else:
            b = self.mask.to(x.device)
        
        s, log_s = self.get_s(b * x)

        log_det_J = torch.sum(log_s * (1-b), dim=-1)

        t = self.f_t(b * x)

        y = b * x + (1 - b) * (x * s + t)

        return y, log_det_J
    
    def inverse(self, y):
        if len(y.shape) == 1:
            b = self.mask.squeeze().to(y.device)
        else:
            b = self.mask.to(y.device)
        
        s, log_s = self.get_s(b * y)

        t = self.f_t(b * y)

        x = b * y + (1 - b) * (y - t) / (s + self.eps)

        return x


class combined_real_nvp(nn.Module):
    '''
    The very basic element of real nvp
    '''
    def __init__(self, dim, f_log_s, f_t, mask=None, clip=None):
        super(combined_real_nvp, self).__init__()

        if mask is None:
            self.mask = self.generate_mask(dim)
        else:
            self.mask = mask
        
        self.nvp_1 = real_nvp_element(dim, f_log_s, f_t, mask=self.mask, clip=clip)
        self.nvp_2 = real_nvp_element(dim, f_log_s, f_t, mask=1 - self.mask, clip=clip)
    
    def generate_mask(self, dim):
        '''
        generate mask for given dimension number `dim`
        '''
        mask = torch.zeros((1, dim))
        for i in range(dim):
            if i % 2 == 0:
                mask[0, i] = 1
        return mask

    def forward(self, x):
        x, log_det_J_1 = self.nvp_1(x)
        x, log_det_J_2 = self.nvp_2(x)

        return x, log_det_J_1 + log_det_J_2
    
    def inverse(self, y):
        y = self.nvp_2.inverse(y)
        y = self.nvp_1.inverse(y)

        return y