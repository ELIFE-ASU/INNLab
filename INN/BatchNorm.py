import torch
import torch.nn as nn
from . import INNAbstract


class BatchNorm1d(nn.BatchNorm1d, INNAbstract.INNModule):
    def __init__(self, dim, requires_grad=True, eps=1e-05, momentum=0.01):
        INNAbstract.INNModule.__init__(self)
        nn.BatchNorm1d.__init__(self, num_features=dim, affine=False, eps=eps, momentum=momentum)
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