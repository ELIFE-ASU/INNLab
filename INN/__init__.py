from INN.INN import *
from .Sequential import Sequential
from .BatchNorm import BatchNorm1d, BatchNorm2d
import INN.utilities as utilities
#from .jacobian_linear import JacobianLinear
from .Linears import JacobianLinear
from .ResFlow import NonlinearResFlow, Conv2dResFlow, Conv1dResFlow, ResidualFlow
from .ResFlow.SpectralNormGouk import spectral_norm, remove_spectral_norm
from . import _NICE_modules as NICEModel