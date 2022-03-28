from INN.INNAbstract import INNModule
from . import utils
from .. import utils as coupling_utils


class NonlinearRealNVP(INNModule):
    def __init__(self, dim=None, f_log_s=None, f_t=None, k=4, mask=None, clip=1, activation_fn=None):
        super(NonlinearRealNVP, self).__init__()
        self.dim = dim
        
        if f_log_s is None:
            f_log_s = coupling_utils.default_nonlinear_net(dim, k, activation_fn, zero=True)
        if f_t is None:
            f_t = coupling_utils.default_nonlinear_net(dim, k, activation_fn)

        self.net = utils.combined_real_nvp(dim, f_log_s, f_t, mask, clip)
    
    def forward(self, x, log_p0=0, log_det_J_=0):
        x, log_det = self.net(x)
        if self.compute_p:
            return x, log_p0, log_det + log_det_J_
        else:
            return x
    
    def inverse(self, y, **args):
        y = self.net.inverse(y)
        return y
    
    def __repr__(self):
        return f'NonlinearRealNVP(dim={self.dim})'