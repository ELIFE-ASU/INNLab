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
block = INN.Nonlinear(dim=3, method='NICE')
block_seq = INN.Sequential(INN.Nonlinear(dim=3, method='NICE'),
                           INN.Nonlinear(dim=3, method='NICE'),
                           INN.Nonlinear(dim=3, method='NICE'))


# Basic test functions

def _forward_test(model, dim, requires_grad=False, batch_size=64, device='cpu'):
    x = torch.randn((dim, batch_size))
    x.requires_grad = requires_grad
    x.to(device)
    model.to(device)

    # reqiores_p = True
    model.computing_p(True)
    try:
        y, log_p, log_det = model(x)
        print(f'y={y},\nlog_p={log_p}\nlog_det={log_det}')
    except:
        print('fail at forward (requires p)')
    
    # reqiores_p = False
    model.computing_p(False)
    try:
        y = model(x)
        print(f'y={y}')
    except:
        print('fail at forward (not requires p)')
    
    return 0


print('testing block (CPU) ...')
_forward_test(block, dim=3, device='cpu')

print('testing block (CUDA) ...')
_forward_test(block, dim=3, device='cuda:0')

print('testing seq (CPU) ...')
_forward_test(block_seq, dim=3, device='cpu')

print('testing seq (CUDA) ...')
_forward_test(block_seq, dim=3, device='cuda:0')