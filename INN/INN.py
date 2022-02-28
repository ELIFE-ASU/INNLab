'''
High-level abstraction of invertible neural networks
Author: Yanbo Zhang
'''

import torch
import torch.nn as nn
import INN.utilities as utilities
import INN.INNAbstract as INNAbstract
import INN.cnn as cnn
import torch.nn.functional as F
import INN.pixel_shuffle_1d as ps
from ._ResFlow_modules import NonlinearResFlow, Conv2dResFlow, Conv1dResFlow, ResidualFlow
from ._NICE_modules import NonlinearNICE, Conv1dNICE, Conv2dNICE
from ._RealNVP_modules import NonlinearRealNVP

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


class PixelShuffle1d(INNAbstract.PixelShuffleModule):
    '''
    2d invertible pixel shuffle, using the built-in method
    from pytorch. (nn.PixelShuffle, and nn.PixelUnshuffle)
    '''
    def __init__(self, r):
        super(PixelShuffle1d, self).__init__()
        self.r = r
        self.shuffle = ps.PixelShuffle1D(r)
        self.unshuffle = ps.PixelUnshuffle1D(r)
    
    def PixelShuffle(self, x):
        return self.shuffle(x)
    
    def PixelUnshuffle(self, x):
        return self.unshuffle(x)


class BatchNorm1d(nn.BatchNorm1d, INNAbstract.INNModule):
    def __init__(self, dim, requires_grad=True):
        INNAbstract.INNModule.__init__(self)
        nn.BatchNorm1d.__init__(self, num_features=dim, affine=False)
        self.requires_grad = requires_grad
    
    def _scale(self, x):
        '''The scale factor of x to compute Jacobian'''
        if len(x.shape) == 2:
            return 1

        s = 1
        for dim in x.shape[2:]:
            s *= dim
        return s
    
    def var(self, x):
        x_ = x.transpose(0,1).contiguous().view(self.num_features, -1)
        return x_.var(1, unbiased=False)
    
    def mean(self, x):
        return x.transpose(0,1).contiguous().view(self.num_features, -1).mean(1)

    def forward(self, x, log_p=0, log_det_J=0):
        '''
        Apply batch normalization to x
        x.shape = [batch_size, dim, *]
        '''
        if self.compute_p:
            if not self.training:
                # if in self.eval()
                var = self.running_var # [dim]
            else:
                # if in training
                var = self.var(x)
                if not self.requires_grad:
                    var = var.detach()

            x = super(BatchNorm1d, self).forward(x)

            log_det = -0.5 * torch.log(var + self.eps)
            log_det = torch.sum(log_det, dim=-1) * self._scale(x)

            return x, log_p, log_det_J + log_det.repeat(x.shape[0])
        else:
            return super(BatchNorm1d, self).forward(x)
    
    def inverse(self, y, **args):
        '''
        inverse y to un-batch-normed numbers
        The shape of y can be:
            a. Linear: [batch_size, dim]
            b. n-d: [batch_size, dim, *]
        '''
        batch_size, dim = y.shape[0], y.shape[1]
        var = self.running_var + self.eps
        mean = self.running_mean
        var = var.reshape(1, dim, *([1]*(len(y.shape) - 2)))
        mean = mean.reshape(1, dim, *([1]*(len(y.shape) - 2)))
        x = y * torch.sqrt(var) + mean
        return x

class BatchNorm2d(nn.BatchNorm2d, INNAbstract.INNModule):
    def __init__(self, dim, requires_grad=True):
        INNAbstract.INNModule.__init__(self)
        nn.BatchNorm2d.__init__(self, num_features=dim, affine=False)
        self.requires_grad = requires_grad
    
    def _scale(self, x):
        '''The scale factor of x to compute Jacobian'''
        if len(x.shape) == 2:
            return 1

        s = 1
        for dim in x.shape[2:]:
            s *= dim
        return s
    
    def var(self, x):
        return x.transpose(0,1).contiguous().view(self.num_features, -1).var(1, unbiased=False)
    
    def mean(self, x):
        return x.transpose(0,1).contiguous().view(self.num_features, -1).mean(1)

    def forward(self, x, log_p=0, log_det_J=0):
        '''
        Apply batch normalization to x
        x.shape = [batch_size, dim, *]
        '''
        if self.compute_p:
            if not self.training:
                # if in self.eval()
                var = self.running_var # [dim]
            else:
                # if in training
                var = self.var(x)
                if not self.requires_grad:
                    var = var.detach()

            x = super(BatchNorm2d, self).forward(x)

            log_det = -0.5 * torch.log(var + self.eps)
            log_det = torch.sum(log_det, dim=-1) * self._scale(x)

            return x, log_p, log_det_J + log_det.repeat(x.shape[0])
        else:
            return super(BatchNorm2d, self).forward(x)
    
    def inverse(self, y, **args):
        '''
        inverse y to un-batch-normed numbers
        The shape of y can be:
            a. Linear: [batch_size, dim]
            b. n-d: [batch_size, dim, *]
        '''
        batch_size, dim = y.shape[0], y.shape[1]
        var = self.running_var + self.eps
        mean = self.running_mean
        var = var.reshape(1, dim, *([1]*(len(y.shape) - 2)))
        mean = mean.reshape(1, dim, *([1]*(len(y.shape) - 2)))
        x = y * torch.sqrt(var) + mean
        return x

class Linear(utilities.InvertibleLinear):
    def __init__(self, dim, positive_s=False, eps=1e-8):
        super(Linear, self).__init__(dim, positive_s=positive_s, eps=eps)
    
    def forward(self, x, log_p0=0, log_det_J=0):
        if len(x.shape) == 1:
            log_det = self.logdet(x)
        if len(x.shape) == 2:
            # [batch, dim]
            log_det = self.logdet(x)
        x = super(Linear, self).forward(x)

        if self.compute_p:
            return x, log_p0, log_det_J + log_det
        else:
            return x
    
    def inverse(self, y, **args):
        return super(Linear, self).inverse(y)


class NICE(INNAbstract.INNModule):

    def __init__(self, dim=None, m=None, mask=None, k=4, activation_fn=None):
        super(NICE, self).__init__()
        
        if m is None:
            m_ = utilities.default_net(dim, k, activation_fn)
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


def Nonlinear(dim, method, **kwargs):
    if method == 'ResFlow':
        return NonlinearResFlow(dim, **kwargs)
    elif method == 'NICE':
        return NonlinearNICE(dim, **kwargs)
    elif method == 'RealNVP':
        #return Nonlinear_old(dim, method=method, **kwargs)
        return NonlinearRealNVP(dim, **kwargs)
    else:
        raise NotImplementedError

class ResizeFeatures(INNAbstract.INNModule):
    '''
    Resize for n-d input, include linear or multi-channel inputs
    '''
    def __init__(self, feature_in, feature_out, dist='normal', conditional=False):
        super(ResizeFeatures, self).__init__()
        self.feature_in = feature_in
        self.feature_out = feature_out

        if dist == 'normal':
            self.dist = utilities.NormalDistribution()
        elif isinstance(dist, INNAbstract.Distribution):
            self.dist = dist
        
        self.initialized = False
        self.conditional = conditional
        self.mu_var = utilities.MuVar(feature_in, feature_out)
    
    def resize(self, x, feature_in, feature_out):
        '''
        x has two kinds of shapes:
            1. [feature_in]
            2. [batch_size, feature_in, *]
        '''
        if len(x.shape) == 1:
            # [feature_in]
            if x.shape[0] != feature_in:
                raise Exception(f'Expect to get {self.feature_in} features, but got {x.shape[0]}.')
            y, z = x[:feature_out], x[feature_out:]
        
        if len(x.shape) >= 2:
            # [batch_size, feature_in, *]
            if x.shape[1] != feature_in:
                raise Exception(f'Expect to get {self.feature_in} features, but got {x.shape[1]}.')
            y, z = x[:, :feature_out], x[:, feature_out:]
        
        return y, z

    def forward(self, x, log_p0=0, log_det_J=0):
        y, z = self.resize(x, self.feature_in, self.feature_out)
        
        if self.conditional:
            mu, var, log_det = self.mu_var(y)
            z = (z - mu) / var
        else:
            log_det = 0
        
        if self.compute_p:
            p = self.dist.logp(z)
            return y, log_p0 + p, log_det_J + log_det
        else:
            return y
    
    def inverse(self, y, **args):
        '''
        y has two kinds of shapes:
            1. [feature_in]
            2. [batch_size, feature_in, *]
        '''
        mu, var, log_det = self.mu_var(y)

        if len(y.shape) == 1:
            # [feature_in]
            if y.shape[0] != self.feature_out:
                raise Exception(f'Expect to get {self.feature_out} features, but got {y.shape[0]}.')
            z = self.dist.sample(self.feature_in-self.feature_out).to(y.device)
            z = z * var + mu
            y = torch.cat([y, z])
        
        if len(y.shape) >= 2:
            # [batch_size, feature_in, *]
            if y.shape[1] != self.feature_out:
                raise Exception(f'Expect to get {self.feature_out} features, but got {y.shape[1]}.')
            shape = list(y.shape)
            shape[1] = self.feature_in-self.feature_out
            z = self.dist.sample(shape).to(y.device)
            z = z * var + mu
            y = torch.cat([y, z], dim=1)
        
        return y


class Conv1d_old(INNAbstract.INNModule):
    def __init__(self, channels, kernel_size, method='NICE', w=4, activation_fn=nn.ReLU, m=None, s=None, t=None, mask=None, k=0.8, num_iter=1, num_n=10):
        super(Conv1d_old, self).__init__()

        self.method = method
        if method == 'NICE':
            self.iresnet = False
            self.f = cnn.Conv1dNICE(channels, kernel_size, w=w, activation_fn=activation_fn, m=m, mask=mask)
        if method == 'RealNVP':
            self.iresnet = False
            self.f = cnn.Conv1dNVP(channels, kernel_size, w=w, activation_fn=activation_fn, s=s, t=t, mask=mask)
        if method == 'iResNet':
            self.iresnet = True
            self.f = cnn.Conv1diResNet(channels, kernel_size, w=w, k=k, num_iter=num_iter, num_n=num_n)
    
    def forward(self, x, log_p0=0, log_det_J=0):
        if not self.iresnet:
            y = self.f(x)
            log_det = self.f.logdet()
            if self.compute_p:
                return y, log_p0, log_det_J + log_det
            else:
                return y
        else:
            # special for i-ResNet
            return self.f(x, log_p0, log_det_J)
    
    def inverse(self, x, **args):
        if not self.iresnet:
            return self.f.inverse(x)
        else:
            return self.f.inverse(x, **args)


def Conv1d(channels, kernel_size, method='NICE', **args):
    if method == 'ResFlow':
        r = kernel_size // 2
        return Conv1dResFlow(channels, r, **args)
    elif method == 'NICE':
        return Conv1dNICE(channels, kernel_size, **args)
    elif method == 'RealNVP':
        return Conv1d_old(channels, kernel_size, method=method, **args)


class Conv2d_old(INNAbstract.INNModule):
    def __init__(self, channels, kernel_size, method='NICE', w=4, activation_fn=nn.ReLU, m=None, s=None, t=None, mask=None, k=0.8, num_iter=1, num_n=10):
        super(Conv2d_old, self).__init__()

        self.method = method
        if method == 'NICE':
            self.iresnet = False
            self.f = cnn.Conv2dNICE(channels, kernel_size, w=w, activation_fn=activation_fn, m=m, mask=mask)
        if method == 'RealNVP':
            self.iresnet = False
            self.f = cnn.Conv2dNVP(channels, kernel_size, w=w, activation_fn=activation_fn, s=s, t=t, mask=mask)
        if method == 'iResNet':
            self.iresnet = True
            self.f = cnn.Conv2diResNet(channels, kernel_size, w=w, k=k, num_iter=num_iter, num_n=num_n)
    
    def forward(self, x, log_p0=0, log_det_J=0):
        if not self.iresnet:
            y = self.f(x)
            log_det = self.f.logdet()
            if self.compute_p:
                return y, log_p0, log_det_J + log_det
            else:
                return y
        else:
            # special for i-ResNet
            return self.f(x, log_p0, log_det_J)
    
    def inverse(self, x, **args):
        if not self.iresnet:
            return self.f.inverse(x)
        else:
            return self.f.inverse(x, **args)


def Conv2d(channels, kernel_size, method='NICE', **args):
    if method == 'ResFlow':
        r = kernel_size // 2
        return Conv2dResFlow(channels, r, **args)
    elif method == 'NICE':
        return Conv2dNICE(channels, kernel_size)
    elif method == 'RealNVP':
        return Conv2d_old(channels, kernel_size, method=method, **args)


class Linear1d(INNAbstract.INNModule):
    def __init__(self, num_feature, mat=None):
        super(Linear1d, self).__init__()
        if mat is None:
            self.mat = utilities.PLUMatrix(num_feature)
        else:
            self.mat = mat
    
    def _to_conv_weight(self, m):
        return m.unsqueeze(-1)

    def weight(self):
        return self._to_conv_weight(self.mat.W())
    
    def weight_inv(self):
        return self._to_conv_weight(self.mat.inv_W())
    
    def _x_scale(self, x):
        return x.shape[-1]

    def logdet(self, x):
        scale = self._x_scale(x)
        logdet = self.mat.logdet().repeat(x.shape[0])

        return logdet * scale
    
    def conv(self, x):
        return F.conv1d(x, self.weight())
    
    def forward(self, x, log_p0=0, log_det_J=0):
        if self.compute_p:
            log_det = self.logdet(x)
            return self.conv(x), log_p0, log_det_J + log_det
        else:
            return self.conv(x)
    
    def inverse(self, y, **args):
        return F.conv1d(y, self.weight_inv())


class Linear2d(Linear1d):
    def __init__(self, num_feature, mat=None):
        super(Linear2d, self).__init__(num_feature, mat)
    
    def _to_conv_weight(self, m):
        return m.unsqueeze(-1).unsqueeze(-1)

    def _x_scale(self, x):
        batch_size, feature, x_dim, y_dim = x.shape
        return x_dim * y_dim
    
    def conv(self, x):
        return F.conv2d(x, self.weight())


class Reshape(INNAbstract.INNModule):
    def __init__(self, shape_in, shape_out):
        super(Reshape, self).__init__()
        self.reshaper = utilities.reshape(shape_in, shape_out)
    
    def forward(self, x, log_p0=0, log_det_J=0):
        if self.compute_p:
            return self.reshaper(x), log_p0, log_det_J
        else:
            return self.reshaper(x)
    
    def inverse(self, y, **args):
        return self.reshaper.inverse(y)