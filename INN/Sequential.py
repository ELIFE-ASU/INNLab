import torch.nn as nn
import INN.INNAbstract as INNAbstract


class Sequential(nn.Sequential, INNAbstract.INNModule):
    def __init__(self, *args):
        INNAbstract.INNModule.__init__(self)
        nn.Sequential.__init__(self, *args)
    
    def forward(self, x, log_p0=0, log_det_J_=0):
        if self.compute_p:
            logp = 0
            logdet = 0

            for module in self:
                x, logp, logdet = module(x, logp, logdet)
            return x, logp + log_p0, logdet + log_det_J_
        else:
            for module in self:
                x = module(x)
            return x
    
    def inverse(self, y, num_iter=100):

        for module in reversed(self):
            y = module.inverse(y, num_iter=num_iter)
        
        return y