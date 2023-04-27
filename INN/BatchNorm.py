import torch
import torch.nn as nn
from . import INNAbstract


class BatchNorm1d(INNAbstract.INNModule):
    def __init__(self, dim, requires_grad=True, eps=1e-05, momentum=0.01):
        INNAbstract.INNModule.__init__(self)
        self.momentum = momentum
        self.eps = eps
        self.requires_grad = requires_grad
        self.num_features = dim

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def _scale(self, x):
        '''The scale factor of x to compute Jacobian'''
        if len(x.shape) == 2:
            return 1

        s = 1
        for dim in x.shape[2:]:
            s *= dim
        return s

    def var(self, x):
        # using custom var function to get consistent results in both training and eval
        x_ = x.transpose(0, 1).contiguous().view(self.num_features, -1)
        return x_.var(1, unbiased=False)

    def mean(self, x):
        return x.transpose(0, 1).contiguous().view(self.num_features, -1).mean(1)

    def forward(self, x, log_p=0, log_det_J=0):
        '''
        Apply batch normalization to x
        x.shape = [batch_size, dim, *]
        '''
        if self.compute_p:
            if not self.training:
                # if in self.eval()
                x = (x - self.running_mean) / \
                    torch.sqrt(self.running_var + self.eps)
                var = self.running_var
            else:
                # if in training
                var = self.var(x)
                mean = self.mean(x)
                if not self.requires_grad:
                    var = var.detach()
                    mean = mean.detach()

                x = (x - mean) / torch.sqrt(var + self.eps)
                # update running mean and var
                self.running_mean = (
                    1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
                self.running_var = (1 - self.momentum) * \
                    self.running_var + self.momentum * var.detach()

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
    def __init__(self, dim, requires_grad=True, eps=1e-05, momentum=0.01):
        INNAbstract.INNModule.__init__(self)
        nn.BatchNorm2d.__init__(self, num_features=dim,
                                affine=False, eps=eps, momentum=momentum)
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
        return x.transpose(0, 1).contiguous().view(self.num_features, -1).var(1, unbiased=False)

    def mean(self, x):
        return x.transpose(0, 1).contiguous().view(self.num_features, -1).mean(1)

    def forward(self, x, log_p=0, log_det_J=0):
        '''
        Apply batch normalization to x
        x.shape = [batch_size, dim, *]
        '''
        if self.compute_p:
            if not self.training:
                # if in self.eval()
                var = self.running_var  # [dim]
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
