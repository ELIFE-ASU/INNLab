import torch.nn as nn
from .conv import ConvNVP
from ..utils import _default_1d_coupling_function


class Conv1dRealNVP(ConvNVP):
    '''
    1-d invertible convolution layer by NICE method
    TODO: inverse error is too large
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, s=None, t=None, mask=None, clip=True, clip_n=1.0):
        super(Conv1dRealNVP, self).__init__(channels, kernel_size, w=w, activation_fn=activation_fn, s=s, t=t, mask=mask, clip=clip, clip_n=clip_n)
        if s is None:
            self.log_s = _default_1d_coupling_function(channels, kernel_size, activation_fn, w=w)
        else:
            self.log_s = s
        
        if t is None:
            self.t = _default_1d_coupling_function(channels, kernel_size, activation_fn, w=w)
        else:
            self.t = t
        self.log_det_ = 0
    
    def forward(self, x, log_p0=0, log_det_J=0):
        y = super(Conv1dRealNVP, self).forward(x)
        if self.compute_p:
            return y, log_p0, log_det_J + self.logdet()
        else:
            return y