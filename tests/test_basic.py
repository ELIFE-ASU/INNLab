import INN
import torch
import torch.nn as nn

'''
Test tasks:

Test unit:
1. Forward
    a. requires p
    b. not requires p
2. Inverse (eval mode)
    a. L1 error

Test Modules:
1. Basic Blocks (CPU / CUDA)
2. Sequential of Blocks (CPU / CUDA)
3. Blocks with custom networks (CPU / CUDA)
'''

# Defining test blocks
method = 'RealNVP'
requires_grad = True
block = INN.Nonlinear(dim=3, method=method)
block_seq = INN.Sequential(INN.Nonlinear(dim=3, method=method),
                           INN.Nonlinear(dim=3, method=method),
                           INN.Nonlinear(dim=3, method=method))


# Basic test functions

def _forward_test(model, dim, requires_grad=False, batch_size=8, device='cpu'):
    x = torch.randn((batch_size, dim))
    x.requires_grad = requires_grad
    x = x.to(device)
    model.to(device)

    # reqiores_p = True
    model.computing_p(True)
    print('model.computing_p(True)')
    y, log_p, log_det = model(x)
    print(f'y={y},\nlog_p={log_p}\nlog_det={log_det}')
    
    # reqiores_p = False
    model.computing_p(False)
    print('model.computing_p(False)')
    y = model(x)
    print(f'y={y}')
    
    return 0

def _inverse_test(model, dim, requires_grad=False, batch_size=8, device='cpu'):
    x = torch.randn((batch_size, dim))
    x.requires_grad = requires_grad
    x = x.to(device)
    model.to(device)
    model.eval()
    lf = nn.L1Loss()

    model.computing_p(False)
    y = model(x)
    x_hat = model.inverse(y)
    loss = lf(x, x_hat)
    print(f'loss={loss}')
    return


print('\ntesting block (CPU) ...')
_forward_test(block, dim=3, requires_grad=requires_grad, device='cpu')
_inverse_test(block, dim=3, requires_grad=requires_grad, device='cpu')
print('#' * 64)

print('\ntesting block (CUDA) ...')
_forward_test(block, dim=3, requires_grad=requires_grad, device='cuda:0')
_inverse_test(block, dim=3, requires_grad=requires_grad, device='cuda:0')
print('#' * 64)

print('\ntesting seq (CPU) ...')
_forward_test(block_seq, dim=3, requires_grad=requires_grad, device='cpu')
_inverse_test(block_seq, dim=3, requires_grad=requires_grad, device='cpu')
print('#' * 64)

print('\ntesting seq (CUDA) ...')
_forward_test(block_seq, dim=3, requires_grad=requires_grad, device='cuda:0')
_inverse_test(block_seq, dim=3, requires_grad=requires_grad, device='cuda:0')
print('#' * 64)