import torch
import torch.nn as nn
import INN.INNAbstract as INNAbstract
import INN.utilities as utilities
#from INN import INNAbstract

class _default_1d_coupling_function(nn.Module):
    def __init__(self, channels, kernel_size, activation_fn=nn.ReLU, w=4):
        super(_default_1d_coupling_function, self).__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f'kernel_size must be an odd number, but got {kernel_size}')
        r = kernel_size // 2

        self.activation_fn = activation_fn
        
        self.f = nn.Sequential(nn.Conv1d(channels, channels*w, kernel_size, padding=r),
                               activation_fn(),
                               nn.Conv1d(w*channels, w*channels, kernel_size, padding=r),
                               activation_fn(),
                               nn.Conv1d(w*channels, channels, kernel_size, padding=r)
                              )
        self.f.apply(self._init_weights)
    
    def _init_weights(self, m):
        nonlinearity = 'leaky_relu' # set to leaky_relu by default

        if self.activation_fn is nn.ReLU:
            nonlinearity = 'relu'
        if self.activation_fn is nn.SELU:
            nonlinearity = 'selu'
        if self.activation_fn is nn.Tanh:
            nonlinearity = 'tanh'
        if self.activation_fn is nn.Sigmoid:
            nonlinearity = 'sigmoid'
        
        if type(m) == nn.Linear:
            # doing Kaiming initialization
            torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity=nonlinearity)
            torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.f(x)


class CouplingConv(nn.Module):
    '''
    General invertible covolution layer for coupling methods
    '''
    def __init__(self, num_feature, mask=None):
        super(CouplingConv, self).__init__()
        self.num_feature = num_feature
        if mask is None:
            self.mask = self._mask(num_feature)
        else:
            self.mask = mask
    
    def _mask(self, n):
        m = torch.zeros(n)
        m[:(n // 2)] = 1
        return m
    
    def working_mask(self, x):
        '''
        Generate feature mask for 1d inputs
        x.shape = [batch_size, feature, *]
        mask.shape = [1, feature, *(1)]
        '''
        batch_size, feature, *other = x.shape
        mask = self.mask.reshape(1, self.num_feature, *[1] * len(other)).to(x.device)
        return mask


class CouplingConv1d(CouplingConv):
    '''
    General 1-d invertible convolution layer for coupling methods
    '''
    def __init__(self, num_feature, mask=None):
        super(CouplingConv1d, self).__init__(num_feature, mask=mask)


class Conv1dNICE(CouplingConv1d):
    '''
    1-d invertible convolution layer by NICE method
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, m=None, mask=None):
        super(Conv1dNICE, self).__init__(num_feature=channels, mask=mask)
        if m is None:
            self.m = _default_1d_coupling_function(channels, kernel_size, activation_fn, w=w)
        else:
            self.m = m
    
    def forward(self, x):
        mask = self.working_mask(x)
        
        x_ = mask * x
        x = x + (1-mask) * self.m(x_)
        
        x_ = (1-mask) * x
        x = x + mask * self.m(x_)
        return x
    
    def inverse(self, y):
        mask = self.working_mask(y)
        
        y_ = (1-mask) * y
        y = y - mask * self.m(y_)
        
        y_ = mask * y
        y = y - (1-mask) * self.m(y_)
        
        return y
    
    def logdet(self, **args):
        return 0


class Conv1dNVP(CouplingConv1d):
    '''
    1-d invertible convolution layer by NICE method
    TODO: inverse error is too large
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, s=None, t=None, mask=None):
        super(Conv1dNVP, self).__init__(num_feature=channels, mask=mask)
        if s is None:
            self.log_s = _default_1d_coupling_function(channels, kernel_size, activation_fn, w=w)
        else:
            self.log_s = s
        
        if t is None:
            self.t = _default_1d_coupling_function(channels, kernel_size, activation_fn, w=w)
        else:
            self.t = t
        self.log_det_ = 0
    
    def s(self, x):
        logs = self.log_s(x)
        return torch.exp(logs), logs
    
    def forward(self, x):
        mask = self.working_mask(x)
        
        x_ = mask * x
        s, log_s_1 = self.s(x_)
        log_s_1 = (1-mask) * log_s_1
        x = (1-mask) * (s * x + self.t(x_)) + x_
        
        mask = 1 - mask
        x_ = mask * x
        s, log_s_2 = self.s(x_)
        log_s_2 = (1-mask) * log_s_2
        x = (1-mask) * (s * x + self.t(x_)) + x_

        log_det = log_s_1 + log_s_2
        log_det = log_det.reshape(x.shape[0], -1).sum(dim=1)

        self.log_det_ = log_det
        
        return x
    
    def logdet(self, **args):
        return self.log_det_
    
    def inverse(self, y):
        mask = 1 - self.working_mask(y)
        
        y_ = mask * y
        y = (1-mask) * (y - self.t(y_)) / torch.exp(self.log_s(y_)) + y_
        
        mask = 1 - mask
        y_ = mask * y
        y = (1-mask) * (y - self.t(y_)) / torch.exp(self.log_s(y_)) + y_
        
        return y


class Conv1diResNet(INNAbstract.Conv):
    '''
    1-d convolutional i-ResNet
    '''
    def __init__(self, channel, kernel_size, w=8, k=0.8, num_iter=1, num_n=10):
        super(Conv1diResNet, self).__init__(num_iter=num_iter, num_n=num_n)
        
        self.net = utilities.SNCov1d(channel, kernel_size, w=w, k=k)


class Conv2diResNet(INNAbstract.Conv):
    '''
    1-d convolutional i-ResNet
    '''
    def __init__(self, channel, kernel_size, w=8, k=0.8, num_iter=1, num_n=10):
        super(Conv1diResNet, self).__init__(num_iter=num_iter, num_n=num_n)
        
        self.net = utilities.SNCov2d(channel, kernel_size, w=w, k=k)