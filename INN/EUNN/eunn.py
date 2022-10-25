'''Unitary linear layer
Reference: Jing, L., Shen, Y., Dubček, T., Peurifoy, J., Skirlo, S., LeCun, Y., Tegmark, M.,
           & Soljačić, M. (2017). Tunable Efficient Unitary Neural Networks (EUNN) and their
           application to RNNs. ArXiv:1612.05231 [Cs, Stat]. http://arxiv.org/abs/1612.05231

'''

import torch
import torch.nn as nn
from .funtional import unitary, _ind1, _ind2
from ..INNAbstract import INNModule


def _make_ind1(dim):
    indA = _ind1(dim, deepth=0)
    indB = _ind1(dim, deepth=1)

    ind = torch.stack([indA, indB] * (dim // 2))
    return ind

def _make_ind2(dim):
    indA = _ind2(dim, deepth=0)
    indB = _ind2(dim, deepth=1)

    ind = torch.stack([indA, indB] * (dim // 2))
    return ind

class TunableEUNN(INNModule):
    def __init__(self, dim, deepth=None, complex_space=False):
        r'''Tunable Efficient Unitary Neural Network (EUNN)
        
        Parameters:
            dim (int): dimension of the input and output
            deepth (int): number of passes of the unitary layer
            complex_space (bool): whether to use complex space
        
        Methods:
            forward(x): forward pass
            get_matrix(): get the matrix representation of the unitary layer
        '''
        super().__init__()
        self.dim = dim
        self.complex_space = complex_space
        self.deepth = deepth if deepth is not None else dim

        self.theta = nn.Parameter(torch.rand(self.deepth, dim // 2))
        self.phi = nn.Parameter(torch.rand(self.deepth, dim // 2)) if complex_space else None

        self.ind1 = _make_ind1(dim)
        self.ind2 = _make_ind2(dim)
    
    def forward(self, x, log_p=None, log_det=None):
        y = unitary(x, self.theta, self.ind1, self.ind2, phis=self.phi, complex_space=self.complex_space)

        if self.compute_p:
            return y, log_p, log_det
        else:
            return y
    
    def inverse(self, y, **args):
        # inverse the first dimension
        inv_theta = -torch.flip(self.theta, dims=[0])
        inv_phi = -torch.flip(self.phi, dims=[0]) if self.complex_space else None
        inv_ind1 = torch.flip(self.ind1, dims=[0])
        inv_ind2 = torch.flip(self.ind2, dims=[0])
        x = unitary(y, inv_theta, inv_ind1, inv_ind2, phis=inv_phi, complex_space=self.complex_space)
        return x
    
    def get_matrix(self):
        x = torch.eye(self.dim)
        return self.forward(x)

def EUNN(dim, deepth=None, complex_space=False, method='tunable'):
    if not method in ['tunable']:
        raise ValueError('method should be one of [tunable]')
    if method == 'tunable':
        return TunableEUNN(dim, deepth, complex_space)