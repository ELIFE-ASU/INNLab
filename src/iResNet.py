'''
High-level abstraction of i-ResNet
Author: Yanbo Zhang
'''

import torch
import torch.nn as nn
import utilities
import iResNetAbstract 

# for test only, reload for any changes
import importlib
importlib.reload(iResNetAbstract)
importlib.reload(utilities)
# end

iResNetModule = iResNetAbstract.iResNetModule

class FCN(iResNetModule):
    '''
    i-ResNet which g is a fully connected network
    '''
    def __init__(self, dim_in, dim_out, beta=0.8, w=8, num_iter=1, num_n=3):
        '''
        beta: the Lip constant, beta < 1
        w: the width of the hidden layer
        '''
        super(FCN, self).__init__()
        if dim_out > dim_in:
            raise Exception(f"dim_out ({dim_out}) cannnot be larger than dim_in ({dim_in}).")
        
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.num_iter = num_iter
        self.num_n = num_n
        
        self.net = utilities.SNFCN(dim_in, w=w, k=beta)
        self.noise = utilities.NormalDistribution()
    
    def g(self, x):
        return self.net(x)
    
    def P(self, y):
        '''
        Normal distribution
        '''
        return self.noise(y)
    
    def inject_noise(self, y):
        # inject noise to y
        if self.dim_out == self.dim_in:
            return y
        if len(y.shape) == 1:
            noise = self.noise.sample(self.dim_in - self.dim_out)
            y_hat = torch.cat([y, noise])
            return y_hat.to(y.device)
        if len(y.shape) == 2:
            noise = self.noise.sample((y.shape[0], self.dim_in - self.dim_out))
            y_hat = torch.cat([y, noise], dim=-1)
            return y_hat.to(y.device)
        raise Exception(f"The input shape must be 1-d or 2-d, but got input.shape={y.shape}.")
    
    def cut(self, x):
        '''
        Split output into two parts: y, z
        input: [dim_in] or [batch, dim_in]
        '''
        if len(x.shape) == 1:
            y = x[:self.dim_out]
            z = x[self.dim_out:]
            return y, z
        
        if len(x.shape) == 2:
            y = x[:, :self.dim_out]
            z = x[:, self.dim_out:]
            return y, z
        raise Exception(f"The input shape must be 1-d or 2-d, but got input.shape={x.shape}.")
    
    def logdet(self, x, g):
        self.eval()
        logdet = 0
        for i in range(self.num_iter):
            v = torch.randn(x.shape) # random noise
            v = v.to(x.device)
            w = v
            for k in range(1, self.num_n):
                w = utilities.vjp(g, x, w)[0]
                logdet += (-1)**(k+1) * torch.sum(w * v, dim=-1) / k
        
        logdet /= self.num_iter
        self.train()
        return logdet


class Conv1d(iResNetAbstract.Conv):
    '''
    1-d convolutional i-ResNet
    '''
    def __init__(self, channel, kernel_size, w=8, k=0.8, num_iter=1, num_n=3):
        super(Conv1d, self).__init__(num_iter=num_iter, num_n=num_n)
        
        self.net = utilities.SNCov1d(channel, kernel_size, w=w, k=k)


class Conv2d(iResNetAbstract.Conv):
    '''
    1-d convolutional i-ResNet
    '''
    def __init__(self, channel, kernel_size, w=8, k=0.8, num_iter=1, num_n=3):
        super(Conv2d, self).__init__(num_iter=num_iter, num_n=num_n)
        
        self.net = utilities.SNCov2d(channel, kernel_size, w=w, k=k)


class Sequential(nn.Sequential):

    def __init__(self, *args):
        super(Sequential, self).__init__(*args)
    
    def forward(self, x):
        if self.training:
            logp = 0
            logdet = 0

            for module in self:
                x, logp, logdet = module(x, logp, logdet)
            return x, logp, logdet
        else:
            for module in self:
                x = module(x)
            return x
    
    def inverse(self, y, num_iter=100):

        for module in reversed(self):
            y = module.inverse(y, num_iter=num_iter)
        
        return y


class PixelShuffle2d(iResNetAbstract.PixelShuffleModule):
    '''
    2d invertible pixel shuffle, using the built-in method
    from pytorch. (nn.PixelShuffle, and nn.PixelUnshuffle)
    '''
    def __init__(self, r):
        super(PixelShuffle2d, self).__init__()
        self.r = r
        self.shuffle = nn.PixelShuffle(r)
        self.unshuffle = nn.PixelUnshuffle(r)
    
    def PixelShuffle(self, x):
        return self.shuffle(x)
    
    def PixelUnshuffle(self, x):
        return self.unshuffle(x)


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, dim):
        super(BatchNorm1d, self).__init__(num_features=dim, affine=False)

    def forward(self, x, log_p=0, log_det_J=0):
        if self.training:
            var = torch.var(x, dim=0)

            x = super(BatchNorm1d, self).forward(x)

            log_det = -0.5 * torch.log(var + self.eps)
            log_det = torch.sum(log_det, dim=-1)

            return x, log_p, log_det_J + log_det
        else:
            return super(BatchNorm1d, self).forward(x)
    
    def inverse(self, y, num_iter=1):
        var = self.running_var + self.eps
        mean = self.running_mean
        x = y * torch.sqrt(var) + mean
        return x