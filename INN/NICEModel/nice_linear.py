# NICE with 1d input
from .NICE_base import NICE
from ..INNAbstract import INNModule
from . import utils


class LinearNICE(NICE):

    def __init__(self, dim=None, m=None, mask=None, k=4, activation_fn=None):
        if m is None:
            m_ = utils.default_net(dim, k, activation_fn)
        super(LinearNICE, self).__init__(dim, m=m_, mask=mask)
    
    def forward(self, x, log_p0=0, log_det_J=0):
        y = super(LinearNICE, self).forward(x)
        if self.compute_p:
            return y, log_p0, log_det_J + self.logdet()
        else:
            return y
    
    def inverse(self, y, **args):
        y = super(LinearNICE, self).inverse(y)
        return y
    
    def __repr__(self):
        return f'LinearNICE(dim={self.dim})'