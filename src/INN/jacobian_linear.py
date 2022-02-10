import INN
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version


def svd_sigma(tensor):
    if version.parse(torch.__version__) > version.parse('1.8.1'):
        sigma = torch.linalg.svd(tensor).S
    else:
        sigma = torch.svd(tensor, compute_uv=True).S
    
    return sigma


class jacobian_linear(INN.INNAbstract.INNModule):
    '''
    A simplified version of invertable neural network
    works for low dimensions of vectors (O(n^3) time complexity)
    '''
    def __init__(self, dim_in, dim_out, bias=True):
        super(jacobian_linear, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.dim_out = dim_out
        self.dim_in = dim_in

        nn.init.orthogonal_(self.linear.weight.data)

    def logdet(self, x):
        sigma = svd_sigma(self.linear.weight)
        logdet = torch.sum(torch.log(sigma.abs()))
        logdet = logdet.repeat(x.shape[0])
        return logdet

    def forward(self, x):
        return self.linear(x)

    def inverse(self, y):
        if self.dim_in != self.dim_out:
            raise NotImplementedError('inverse is not available if dim_in != dim_out!')
        
        return F.linear(y-self.linear.bias, torch.inverse(self.linear.weight))


class JacobianLinear(jacobian_linear):
    def __init__(self, dim_in, dim_out=None, bias=True):
        if dim_out is None:
            dim_out = dim_in
        super(JacobianLinear, self).__init__(dim_in, dim_out, bias=bias)

    def forward(self, x, log_p0=0, log_det_J=0):
        log_det = self.logdet(x)
        x = self.linear(x)

        if self.compute_p:
            return x, log_p0, log_det_J + log_det
        else:
            return x

    def inverse(self, y, **args):
        return super(JacobianLinear, self).inverse(y)



if __name__ == '__main__':
    print('< test >')
    def test():
        model = JacobianLinear(5, 1)
        logdet = model.logdet(torch.randn(4, 5))
        print(logdet)

    test()

