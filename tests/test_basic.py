import INN
import torch

'''
Test tasks:

Test unit:
1. Forward
    a. requires p
    b. not requires p
2. Inverse (eval mode)
    a. MSE error

Test Modules:
1. Basic Blocks (CPU / CUDA)
2. Sequential of Blocks (CPU / CUDA)
3. Blocks with custom networks (CPU / CUDA)
'''

# Defining test blocks
block = INN.Nonlinear(dim=3, method='iResNet')
block_seq = INN.Sequential(INN.Nonlinear(dim=3, method='iResNet'),
                           INN.Nonlinear(dim=3, method='iResNet'),
                           INN.Nonlinear(dim=3, method='iResNet'))


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


requires_grad = True
print('\ntesting block (CPU) ...')
_forward_test(block, dim=3, requires_grad=requires_grad, device='cpu')
print('#' * 64)

print('\ntesting block (CUDA) ...')
_forward_test(block, dim=3, requires_grad=requires_grad, device='cuda:0')
print('#' * 64)

print('\ntesting seq (CPU) ...')
_forward_test(block_seq, dim=3, requires_grad=requires_grad, device='cpu')
print('#' * 64)

print('\ntesting seq (CUDA) ...')
_forward_test(block_seq, dim=3, requires_grad=requires_grad, device='cuda:0')
print('#' * 64)