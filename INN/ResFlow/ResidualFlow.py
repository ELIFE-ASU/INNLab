import torch
import torch.nn as nn
from .utilities import ShiftedGeometric, vjp
from .BackwardInForward import MemoryEfficientLogDetEstimator


def Jacobian(x, gx, N, max_n=100):
    r'''Compute the Jacobian term by given x and g(x)
    For more details, please refer to the paper: https://arxiv.org/abs/1906.02735

    Inputs:
        * x: x, input tensor
        * gx: g(x)

    Parameters:
        * N: expected number of steps
        * max_n: maximum number of iterations
    '''
    assert N > 1
    sign = -1; p = ShiftedGeometric.para_from_mean(N); batch = x.shape[0]
    n = ShiftedGeometric.sample(p) # sample from geometric distribution
    n = min(n, max_n) # constrain the number of steps

    J = 0
    v = torch.randn(x.shape).to(x.device)
    w = v

    for k in range(1, n+1):
        sign *= -1
        w, = vjp(gx, (x,), w)
        term = sign * torch.sum((v * w).reshape(batch, -1), dim=1) / k
        term /= 1 - ShiftedGeometric.CDF(p, k-1) # reweight
        J += term
    return J


class _lipschitz_constrained(nn.Module):
    def __init__(self, module, lipschitz_constrain=0.9):
        super(_lipschitz_constrained, self).__init__()
        self.module = module
        self.lipschitz_constrain = lipschitz_constrain

    def forward(self, x):
        return self.module(x) * self.lipschitz_constrain


class ResidualFlow(nn.Module):
    def __init__(self, residual, lipschitz_constrain=0.9, mem_efficient=True, est_steps=5):
        super(ResidualFlow, self).__init__()
        self.g = _lipschitz_constrained(residual, lipschitz_constrain=lipschitz_constrain)
        self.lipschitz_constrain = lipschitz_constrain
        self.mem_efficient = mem_efficient
        self.est_steps = est_steps
        self.init()
    
    def init(self):
        r'''Initialize the self.g to avoid negative infinity log |det J|'''
        pass
    
    def compute(self, x):
        r'''Compute output and residual term'''
        gx = self.g(x)
        return x + gx, gx
    
    def jacobian(self, x, N):
        gx, logdet = MemoryEfficientLogDetEstimator().apply(Jacobian, self.g, x, N)
        return logdet
    
    def forward(self, x, log_p=None, log_det=None):
        if not x.requires_grad:
            x.requires_grad = True

        if self.mem_efficient:
            gx, logdet = MemoryEfficientLogDetEstimator().apply(Jacobian, self.g, x, self.est_steps, *list(self.g.parameters()))
        else:
            gx = self.g(x)
            logdet = Jacobian(x, gx, N)
        y = x + gx

        if log_det is not None:
            logdet += log_det

        return y, log_p, logdet
    
    def inverse(self, y, num_iter=100):
        '''
        The following code is not working.
        This may caused by a bug of PyTorch 1.9.0.post2 on Mac M1
        see: https://github.com/pytorch/pytorch/issues/72594

        with torch.no_grad():
            x = y
            for i in range(num_iter):
                x = y - self.g(x)
        '''
        
        x = y.detach()
        for i in range(num_iter):
            x = y - self.g(x).detach()
        return x