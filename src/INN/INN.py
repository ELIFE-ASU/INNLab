'''
High-level abstraction of invertible neural networks
Author: Yanbo Zhang
'''

import torch
import torch.nn as nn
import INN.utilities as utilities
import INN.INNAbstract as INNAbstract

# for test only, reload for any changes
import importlib
importlib.reload(INNAbstract)
importlib.reload(utilities)
# end

iResNetModule = INNAbstract.iResNetModule

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


class Conv1d(INNAbstract.Conv):
    '''
    1-d convolutional i-ResNet
    '''
    def __init__(self, channel, kernel_size, w=8, k=0.8, num_iter=1, num_n=3):
        super(Conv1d, self).__init__(num_iter=num_iter, num_n=num_n)
        
        self.net = utilities.SNCov1d(channel, kernel_size, w=w, k=k)


class Conv2d(INNAbstract.Conv):
    '''
    1-d convolutional i-ResNet
    '''
    def __init__(self, channel, kernel_size, w=8, k=0.8, num_iter=1, num_n=3):
        super(Conv2d, self).__init__(num_iter=num_iter, num_n=num_n)
        
        self.net = utilities.SNCov2d(channel, kernel_size, w=w, k=k)


class Sequential(nn.Sequential, INNAbstract.INNModule):

    def __init__(self, *args):
        #super(Sequential, self).__init__(*args)
        INNAbstract.INNModule.__init__(self)
        nn.Sequential.__init__(self, *args)
    
    def forward(self, x, log_p0=0, log_det_J_=0):
        if self.compute_p:
            logp = 0
            logdet = 0

            for module in self:
                x, logp, logdet = module(x, logp, logdet)
            return x, logp + log_p0, logdet + log_det_J_
        else:
            for module in self:
                x = module(x)
            return x
    
    def inverse(self, y, num_iter=100):

        for module in reversed(self):
            y = module.inverse(y, num_iter=num_iter)
        
        return y


class PixelShuffle2d(INNAbstract.PixelShuffleModule):
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


class BatchNorm1d(nn.BatchNorm1d, INNAbstract.INNModule):
    def __init__(self, dim):
        INNAbstract.INNModule.__init__(self)
        nn.BatchNorm1d.__init__(self, num_features=dim, affine=False)

    def forward(self, x, log_p=0, log_det_J=0):
        
        if self.compute_p:
            if not self.training:
                # if in self.eval()
                var = self.running_var # [dim]
            else:
                # if in training
                var = torch.var(x, dim=0, unbiased=False).detach() # [dim]

            x = super(BatchNorm1d, self).forward(x)

            log_det = -0.5 * torch.log(var + self.eps)
            log_det = torch.sum(log_det, dim=-1)

            return x, log_p, log_det_J + log_det
        else:
            return super(BatchNorm1d, self).forward(x)
    
    def inverse(self, y, **args):
        var = self.running_var + self.eps
        mean = self.running_mean
        x = y * torch.sqrt(var) + mean
        return x

class Linear(utilities.InvertibleLinear):
    def __init__(self, dim):
        super(Linear, self).__init__(dim)
    
    def forward(self, x, log_p0=0, log_det_J=0):
        if len(x.shape) == 1:
            log_det = self.logdet()
        if len(x.shape) == 2:
            # [batch, dim]
            log_det = self.logdet().repeat(x.shape[0])
        x = super(Linear, self).forward(x)

        if self.compute_p:
            return x, log_p0, log_det_J + log_det
        else:
            return x
    
    def inverse(self, y, **args):
        return super(Linear, self).inverse(y)


class RealNVP(INNAbstract.INNModule):

    def __init__(self, dim=None, f_log_s=None, f_t=None, k=4, mask=None, clip=1):
        super(RealNVP, self).__init__()
        if (f_log_s is None) and (f_t is None):
            log_s = utilities.default_net(dim, k)#self.default_net(dim, k)
            t = utilities.default_net(dim, k)#self.default_net(dim, k)
            self.net = utilities.combined_real_nvp(dim, log_s, t, mask, clip)
        else:
            self.net = utilities.combined_real_nvp(dim, f_log_s, f_t, mask, clip)
    
    def forward(self, x, log_p0=0, log_det_J_=0):
        x, log_det = self.net(x)
        if self.compute_p:
            return x, log_p0, log_det + log_det_J_
        else:
            return x
    
    def inverse(self, y, **args):
        y = self.net.inverse(y)
        return y


class NICE(INNAbstract.INNModule):

    def __init__(self, dim=None, m=None, mask=None, k=4):
        super(NICE, self).__init__()
        
        if m is None:
            m_ = utilities.default_net(dim, k)
            self.net = utilities.NICE(dim, m=m_, mask=mask)
        else:
            self.net = utilities.NICE(dim, m=m, mask=mask)
    
    def forward(self, x, log_p0=0, log_det_J=0):
        x = self.net(x)
        if self.compute_p:
            return x, log_p0, log_det_J
        else:
            return x
    
    def inverse(self, y, **args):
        y = self.net.inverse(y)
        return y

class iResNet(utilities.iResNet):
    def __init__(self, dim=None, g=None, beta=0.8, w=8, num_iter=1, num_n=10):
        super(iResNet, self).__init__(dim, g, beta, w, num_iter, num_n)

def _default_dict(key, _dict, default):
    if key in _dict:
        return _dict[key]
    else:
        return default

class Nonlinear(INNAbstract.INNModule):
    '''
    Nonlinear invertible block
    '''
    def __init__(self, dim, method='NICE', m=None, mask=None, k=4, **args):
        super(Nonlinear, self).__init__()
        
        self.method = method
        if method == 'NICE':
            self.block = NICE(dim, m=m, mask=mask, k=k)
        if method == 'RealNVP':
            clip = _default_dict('clip', args, 1)
            f_log_s = _default_dict('f_log_s', args, None)
            f_t = _default_dict('f_t', args, None)

            self.block = RealNVP(dim=dim, f_log_s=f_log_s, f_t=f_t, k=k, mask=mask, clip=clip)
        if method == 'iResNet':
            g = _default_dict('g', args, None)
            beta = _default_dict('beta', args, 0.8)
            w = _default_dict('w', args, 8)
            num_iter = _default_dict('num_iter', args, 1)
            num_n = _default_dict('num_n', args, 10)
            self.block = iResNet(dim=dim, g=g, beta=beta, w=w, num_iter=num_iter, num_n=num_n)
            
    
    def forward(self, x, log_p0=0, log_det_J=0):
        return self.block(x, log_p0, log_det_J)
    
    def inverse(self, y, **args):
        return self.block.inverse(y, **args)