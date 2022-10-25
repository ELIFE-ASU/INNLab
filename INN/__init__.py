from INN.INN import *
import INN as INN
from .Sequential import Sequential
from .BatchNorm import BatchNorm1d, BatchNorm2d
import INN.utilities as utilities
from .Linears import JacobianLinear

from . import ResFlow
from .ResFlow import NonlinearResFlow, Conv2dResFlow, Conv1dResFlow, ResidualFlow
from .ResFlow.SpectralNormGouk import spectral_norm, remove_spectral_norm

from . import CouplingModels
from .CouplingModels import NICEModel, RealNVP

from .EUNN import EUNN, TunableEUNN