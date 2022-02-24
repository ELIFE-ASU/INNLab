from ..CouplingModels import CouplingConv, _default_1d_coupling_function, _default_2d_coupling_function
import torch
import torch.nn as nn


class ConvNICE(CouplingConv):
    '''
    1-d invertible convolution layer by NICE method
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, m=None, mask=None):
        super(ConvNICE, self).__init__(num_feature=channels, mask=mask)
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


class Conv1dNICE(ConvNICE):
    '''
    1-d invertible convolution layer by NICE method
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, m=None, mask=None):
        super(Conv1dNICE, self).__init__(channels, kernel_size, w=w, activation_fn=activation_fn, m=m, mask=mask)
        self.kernel_size = kernel_size
        if m is None:
            self.m = _default_1d_coupling_function(channels, kernel_size, activation_fn, w=w)
        else:
            self.m = m
    
    def forward(self, x, log_p0=0, log_det_J=0):
        y = super(Conv1dNICE, self).forward(x)
        if self.compute_p:
            return y, log_p0, log_det_J + self.logdet()
        else:
            return y
    
    def inverse(self, y, **args):
        x = super(Conv1dNICE, self).inverse(y)
        return x
    
    def __repr__(self):
        return f'Conv1dNICE(channels={self.num_feature}, kernel_size={self.kernel_size})'


class Conv2dNICE(ConvNICE):
    '''
    1-d invertible convolution layer by NICE method
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, m=None, mask=None):
        super(Conv2dNICE, self).__init__(channels, kernel_size, w=w, activation_fn=activation_fn, m=m, mask=mask)
        self.kernel_size = kernel_size
        if m is None:
            self.m = _default_2d_coupling_function(channels, kernel_size, activation_fn, w=w)
        else:
            self.m = m
    
    def forward(self, x, log_p0=0, log_det_J=0):
        y = super(Conv2dNICE, self).forward(x)
        if self.compute_p:
            return y, log_p0, log_det_J + self.logdet()
        else:
            return y
    
    def inverse(self, y, **args):
        x = super(Conv2dNICE, self).inverse(y)
        return x
    
    def __repr__(self):
        return f'Conv2dNICE(channels={self.num_feature}, kernel_size={self.kernel_size})'