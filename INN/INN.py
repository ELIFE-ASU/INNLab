'''
High-level abstraction of invertible neural networks
Author: Yanbo Zhang
'''

import torch
import torch.nn as nn
import INN.utilities as utilities
import INN.INNAbstract as INNAbstract
#import torch.nn.functional as F
import INN.pixel_shuffle_1d as ps
from ._ResFlow_modules import NonlinearResFlow, Conv2dResFlow, Conv1dResFlow, ResidualFlow
from ._NICE_modules import NonlinearNICE, Conv1dNICE, Conv2dNICE
from ._RealNVP_modules import NonlinearRealNVP, Conv1dRealNVP, Conv2dRealNVP


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


def _default_dict(key, _dict, default):
    if key in _dict:
        return _dict[key]
    else:
        return default


def Nonlinear(dim, method='NICE', **kwargs):
    if method == 'ResFlow':
        return NonlinearResFlow(dim, **kwargs)
    elif method == 'NICE':
        return NonlinearNICE(dim, **kwargs)
    elif method == 'RealNVP':
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


def Conv1d(channels, kernel_size, method='NICE', **args):
    if method == 'ResFlow':
        r = kernel_size // 2
        return Conv1dResFlow(channels, r, **args)
    elif method == 'NICE':
        return Conv1dNICE(channels, kernel_size, **args)
    elif method == 'RealNVP':
        #return Conv1d_old(channels, kernel_size, method=method, **args)
        return Conv1dRealNVP(channels, kernel_size, **args)


def Conv2d(channels, kernel_size, method='NICE', **args):
    if method == 'ResFlow':
        r = kernel_size // 2
        return Conv2dResFlow(channels, r, **args)
    elif method == 'NICE':
        return Conv2dNICE(channels, kernel_size)
    elif method == 'RealNVP':
        #return Conv2d_old(channels, kernel_size, method=method, **args)
        return Conv2dRealNVP(channels, kernel_size, **args)


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