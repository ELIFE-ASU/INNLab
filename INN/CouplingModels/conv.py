import torch
import torch.nn as nn
from INN.INNAbstract import INNModule


class CouplingConv(INNModule):
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