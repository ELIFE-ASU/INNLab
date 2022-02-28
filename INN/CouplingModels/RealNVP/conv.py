import torch
import torch.nn as nn
from ..conv import CouplingConv

class ConvNVP(CouplingConv):
    '''
    1-d invertible convolution layer by NICE method
    TODO: inverse error is too large
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, s=None, t=None, mask=None, clip=True, clip_n=1.0):
        super(ConvNVP, self).__init__(num_feature=channels, mask=mask)
        self.log_s = s
        self.t = t
        self.log_det_ = 0
        self.clip = clip
        self.clip_n = clip_n
    
    def clipping_log_s(self, logs):
        return torch.tanh(logs / self.clip_n) * self.clip_n
    
    def s(self, x):
        logs = self.log_s(x)
        if self.clip:
            logs = self.clipping_log_s(logs)
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
    
    def inverse(self, y, **args):
        mask = 1 - self.working_mask(y)
        
        y_ = mask * y
        y = (1-mask) * (y - self.t(y_)) / torch.exp(self.log_s(y_)) + y_
        
        mask = 1 - mask
        y_ = mask * y
        y = (1-mask) * (y - self.t(y_)) / torch.exp(self.log_s(y_)) + y_
        
        return y