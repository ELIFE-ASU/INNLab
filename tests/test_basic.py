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
dim = 5
block = INN.Linear(dim, positive_s=False)#INN.ResizeFeatures(feature_in=dim, feature_out=dim-1)
block_seq = INN.Sequential(INN.Nonlinear(dim=dim, method=method),
                           INN.Nonlinear(dim=dim, method=method),
                           INN.Nonlinear(dim=dim, method=method),
                           INN.Linear(dim, positive_s=False)
                           )


# Basic test functions

def _forward_test(model, dim, requires_grad=False, batch_size=8, device='cpu'):
    print(f' forward test ({device}) ', end=' : ')
    x = torch.randn((batch_size, *dim))
    x.requires_grad = requires_grad
    x = x.to(device)
    model.to(device)

    # reqiores_p = True
    model.computing_p(True)
    y, log_p, log_det = model(x)
    
    # reqiores_p = False
    model.computing_p(False)
    y = model(x)

    print(f'pass')
    
    return 0

def _inverse_test(model, dim, requires_grad=False, batch_size=8, device='cpu', th=1e-6):
    print(f' inverse test ({device}) ', end=' : ')
    x = torch.randn((batch_size, *dim))
    x.requires_grad = requires_grad
    x = x.to(device)
    model.to(device)
    model.eval()
    lf = nn.L1Loss()

    model.computing_p(False)
    print('compute_p=False', end=', ')
    y = model(x)
    x_hat = model.inverse(y)
    loss = lf(x, x_hat)
    print(f'loss={loss}', end=', ')
    if loss.item() < th:
        print('pass')
    else:
        raise Exception(f'L1 loss of inverse is too high!')
    return


def BasicTest(model, dim, requires_grad=False, batch_size=8):
    _forward_test(model, dim, requires_grad, batch_size, device='cpu')
    _forward_test(model, dim, requires_grad, batch_size, device='cuda:0')
    _inverse_test(model, dim, requires_grad, batch_size, device='cpu')
    _inverse_test(model, dim, requires_grad, batch_size, device='cuda:0')


'''
########################################################################
                            Start Tests
########################################################################
'''

'''
print('#'*32 + ' Testing Nonlinear (RealNVP)' + '#'*32)
model = INN.Nonlinear(5, method='RealNVP')
BasicTest(model, [5], requires_grad=False)
print('Sequential:')
BasicTest(INN.Sequential(model, model), [5], requires_grad=False)'''

print('#'*32 + ' Testing Conv1d (RealNVP)' + '#'*32)
model = INN.Conv1d(channels=5, kernel_size=3, method='RealNVP')
BasicTest(model, [5, 8], requires_grad=False)
print('Sequential:')
BasicTest(INN.Sequential(model, model), [5, 8], requires_grad=False)

print('#'*32 + ' Testing Conv1d (NICE)' + '#'*32)
model = INN.Conv1d(channels=5, kernel_size=3, method='NICE')
BasicTest(model, [5, 8], requires_grad=False)
print('Sequential:')
BasicTest(INN.Sequential(model, model), [5, 8], requires_grad=False)

print('#'*32 + ' Testing Conv1d (iResNet)' + '#'*32)
model = INN.Conv1d(channels=5, kernel_size=3, method='iResNet')
BasicTest(model, [5, 8], requires_grad=True)
print('Sequential:')
BasicTest(INN.Sequential(model, model), [5, 8], requires_grad=True)

print('#'*32 + ' Testing Linear1d (PLU)' + '#'*32)
model = INN.Linear1d(num_feature=5)
BasicTest(model, [5, 8], requires_grad=False)
print('Sequential:')
BasicTest(INN.Sequential(model, model), [5, 8], requires_grad=False)

print('#'*32 + ' Testing Linear2d (PLU)' + '#'*32)
model = INN.Linear2d(num_feature=5)
BasicTest(model, [5, 8, 8], requires_grad=False)
print('Sequential:')
BasicTest(INN.Sequential(model, model), [5, 8, 8], requires_grad=False)

print('#'*32 + ' Testing Reshape' + '#'*32)
model = INN.Reshape(shape_in=(8,8), shape_out=(64,))
BasicTest(model, [8, 8], requires_grad=False)
print('Sequential:')
model = INN.Sequential(INN.Reshape(shape_in=(8,8), shape_out=(64,)),
                       INN.Reshape(shape_in=(64,), shape_out=(32, 2)))
BasicTest(model, [8, 8], requires_grad=False)

print('#'*32 + ' Testing BatchNorm1d (Linear inputs)' + '#'*32)
model = INN.BatchNorm1d(5).eval()
BasicTest(model, [5], requires_grad=False)
print('Sequential:')
model = INN.Sequential(model,
                       model)
BasicTest(model, [5], requires_grad=False)

print('#'*32 + ' Testing BatchNorm1d (1d CNN)' + '#'*32)
model = INN.BatchNorm1d(5).eval()
BasicTest(model, [5, 8], requires_grad=False)
print('Sequential:')
model = INN.Sequential(model,
                       model)
BasicTest(model, [5, 8], requires_grad=False)

print('#'*32 + ' Testing PixelShuffle1d ' + '#'*32)
model = INN.PixelShuffle1d(2)
BasicTest(model, [5, 8], requires_grad=False)
print('Sequential:')
model = INN.Sequential(model,
                       model)
BasicTest(model, [5, 8], requires_grad=False)