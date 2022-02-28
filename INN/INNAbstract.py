import torch
import torch.nn as nn
#from INN.utilities import vjp

def vjp(ys, xs, v):
    vJ = torch.autograd.grad(ys, xs, grad_outputs=v, create_graph=True, retain_graph=True, allow_unused=True)
    return tuple([j for j in vJ])

class INNModule(nn.Module):
    def __init__(self):
        super(INNModule, self).__init__()
        self.compute_p = True

    def computing_p(self, b):
        self.compute_p = b
        for sub_m in self.modules():
            if isinstance(sub_m, INNModule):
                sub_m.compute_p = b


class PixelShuffleModule(INNModule):
    '''
    Module for invertible pixel shuffle
    > Pixel Shuffle: https://arxiv.org/abs/1609.05158

    Override required:
        PixelShuffle(self, x) --> x
        PixelUnshuffle(self, x) --> x
    '''
    def __init__(self):
        super(PixelShuffleModule, self).__init__()
    
    def PixelShuffle(self, x):
        pass
    
    def PixelUnshuffle(self, x):
        pass

    def forward(self, x, log_p0=0, log_det_J=0):
        # The log(p_0) and log|det J| will not change under this transformation
        if self.compute_p:
            return self.PixelUnshuffle(x), log_p0, log_det_J
        else:
            return self.PixelUnshuffle(x)
    
    def inverse(self, y, **args):
        return self.PixelShuffle(y)


class Distribution(nn.Module):

    def __init__(self):
        super(Distribution, self).__init__()
    
    def logp(self, x):
        raise NotImplementedError('logp() not implemented')
    
    def sample(self, shape):
        raise NotImplementedError('sample() not implemented')

    def forward(self, x):
        x = self.logp(x)
        
        return x