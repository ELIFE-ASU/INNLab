# Implement invertible linear transformations
from .jacobian_linear import JacobianLinear


def Linear(in_features, out_features=None, method='free', bias=True):
    '''
    Linear layer with invertible weight matrix
    '''
    if method == 'free':
        if out_features is None:
            out_features = in_features
        return JacobianLinear(dim_in=in_features, dim_out=out_features, bias=bias)