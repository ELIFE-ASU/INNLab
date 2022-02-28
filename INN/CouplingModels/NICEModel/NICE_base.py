# Basic NICE architecture
from INN.INNAbstract import INNModule
from .utils import linear_mask


class NICE(INNModule):
    '''
    dim: dimension of input / output
    m: function m
    '''
    def __init__(self, dim, m, mask=None):
        super(NICE, self).__init__()

        self.dim = dim
        if mask is None:
            self.mask = linear_mask(dim)
        else:
            self.mask = mask
        self.m = m
    
    def forward(self, x):
        if len(x.shape) == 1:
            b = self.mask.squeeze().to(x.device)
        else:
            b = self.mask.to(x.device)
        
        x = x + (1-b) * self.m(b * x)
        x = x + b * self.m((1-b) * x)
        return x
    
    def logdet(self):
        return 0
    
    def inverse(self, y):
        if len(y.shape) == 1:
            b = self.mask.squeeze().to(y.device)
        else:
            b = self.mask.to(y.device)
        y = y - b * self.m((1-b) * y)
        y = y - (1 - b) * self.m(b * y)
        return y