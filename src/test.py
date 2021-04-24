'''
Code for test
'''
import torch
import torch.nn as nn

def RandomInput(dim, batch_size):
    return torch.randn(batch_size, dim)

lf = nn.MSELoss()

def _test_inverse_error(model, dim, batch_size):
    x = RandomInput(dim, batch_size)
    x.requires_grad = True
    y, log_p, log_det = model(x)
    x_hat = model.inverse(y)
    diff = lf(x, x_hat)
    return torch.sqrt(diff).cpu().detach().item()

def TestInverse(model, dim, batch_size, time_n):
    '''
    Test the inverse process with time_n times
    '''
    results = [_test_inverse_error(model, dim, batch_size) for i in range(time_n)]
    return sum(results) / len(results)