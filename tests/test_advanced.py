import INN
import torch
import torch.nn as nn

'''
TODO: directly compute log|det(J)| by a strict way, compare to the output
'''

def linear_Jacobian_matrix(model, x):
    '''
    Compute Jacobian for linear input / outputs
    '''
    batch_size, dim = x.shape
    x.requires_grad = True
    model.computing_p(True)
    y, log_p, log_det = model(x)
    
    grad_list = []
    for i in range(dim):
        v = torch.zeros((batch_size, dim))
        v[:, i] = 1
        grad = INN.utilities.vjp(y, x, v)[0]
        grad_list.append(grad.detach())
    return torch.stack(grad_list, dim=1), log_det

def Jacobian_matrix(model, x):
    '''
    Test Jacobian for any shapes
    [NOTE: Batch Norm not supported]
    '''
    shape = x.shape
    dim = int(torch.prod(torch.Tensor(list(x.shape))).item())
    repeats = [dim]
    for i in range(len(x.shape)):
        repeats.append(1)
    
    x_hat = x.unsqueeze(0).repeat(tuple(repeats))
    #print(f'x.shape={x.shape}, x_hat.shape={x_hat.shape}')
    x_hat.requires_grad = True
    model.computing_p(True)
    y, log_p, log_det = model(x_hat)
    
    v = torch.diag(torch.ones(dim)).reshape((y.shape))
    grad = INN.utilities.vjp(y, x_hat, v)[0]
    
    return grad.detach().reshape(y.shape), log_det.detach()

def JacobianShapeTest(model, shape):
    if isinstance(shape, int):
        input_shape = (10, shape)
    else:
        input_shape = (10, *shape)
    input = torch.randn(input_shape) # batch_size = 10
    input.requires_grad = True

    # The log|det J| should also have shape [batch_size]
    model.eval()
    model.computing_p(True)
    output, log_p, log_det = model(input)
    if log_det.shape == (10,):
        print('shape test pass', end=', ')
    else:
        raise Exception(f'expect to got log_det.shape=[10], but got {log_det.shape}')

def TestJacobian(model, shape, th=1e-6):
    JacobianShapeTest(model, shape)
    model.eval()
    J, logdet = Jacobian_matrix(model, x=torch.randn(shape))
    #print(J.shape)
    log_det_J = torch.log(torch.abs(torch.det(J)))
    print(f'J={log_det_J:.10f}, estimated={torch.mean(logdet):.10f}', end=' , ')
    diff = nn.L1Loss()(log_det_J, torch.mean(logdet))
    if abs(diff / log_det_J) <= th:
        print('pass')
    else:
        print(f'estimation error is too big (relative loss={abs(diff / log_det_J):.8f})')

'''
########################################################################
                            Start Tests
########################################################################
'''

print('#'*8 + ' Nonlinear (RealNVP) ' + '#'*8)
model = INN.Nonlinear(5)
TestJacobian(model, shape=5)

print('#'*8 + ' Nonlinear (NICE) ' + '#'*8)
model = INN.Sequential(INN.Nonlinear(5), INN.Nonlinear(5, method='NICE'))
TestJacobian(model, shape=5)

print('#'*8 + ' Nonlinear (iResNet) ' + '#'*8)
model = INN.Sequential(INN.Nonlinear(5), INN.Nonlinear(5, method='iResNet'))
TestJacobian(model, shape=5)

print('#'*8 + ' Conv1d (RealNVP, NICE) ' + '#'*8)
model = INN.Sequential(INN.Conv1d(5, kernel_size=1, method='RealNVP'),
                       INN.Conv1d(5, kernel_size=1, method='NICE'),
                       INN.Reshape(shape_in=(5,8), shape_out=(40,)))
TestJacobian(model, shape=(5, 8))

print('#'*8 + ' 1x1 Conv1d ' + '#'*8)
model = INN.Sequential(INN.Conv1d(5, kernel_size=1, method='RealNVP'),
                       INN.Linear1d(5),
                       INN.Reshape(shape_in=(5,8), shape_out=(40,)))
TestJacobian(model, shape=(5, 8))

print('#'*8 + ' BatchNorm1d (Linear) ' + '#'*8)
model = INN.BatchNorm1d(5)
model.running_var *= torch.exp(torch.randn(1))
TestJacobian(model, shape=(5,))

print('#'*8 + ' BatchNorm1d (1d) ' + '#'*8)
model = INN.Sequential(INN.Conv1d(5, kernel_size=1, method='RealNVP'),
                       INN.BatchNorm1d(5),
                       INN.Reshape(shape_in=(5,8), shape_out=(40,)))
model[1].running_var *= torch.exp(torch.randn(1))
TestJacobian(model, shape=(5, 8))


print('#'*8 + ' PixelShuffle1d ' + '#'*8)
model = INN.Sequential(INN.Conv1d(5, kernel_size=1, method='RealNVP'),
                       INN.PixelShuffle1d(2),
                       INN.Reshape(shape_in=(10,4), shape_out=(40,)))
TestJacobian(model, shape=(5, 8))

print('#'*8 + ' Conv2d (NICE) ' + '#'*8)
model = INN.Sequential(INN.Conv2d(5, kernel_size=3, method='NICE'),
                       INN.BatchNorm2d(5),
                       INN.Reshape(shape_in=(5,4,4), shape_out=(80,)))
model[1].running_var *= torch.exp(torch.randn(1))
TestJacobian(model, shape=(5, 4, 4))

print('#'*8 + ' Conv2d (RealNVP) ' + '#'*8)
model = INN.Sequential(INN.Conv2d(5, kernel_size=3, method='RealNVP'),
                       INN.BatchNorm2d(5),
                       INN.Reshape(shape_in=(5,4,4), shape_out=(80,)))
model[1].running_var *= torch.exp(torch.randn(1))
TestJacobian(model, shape=(5, 4, 4))

print('#'*8 + ' Conv2d (iResNet) ' + '#'*8)
model = INN.Sequential(INN.Conv2d(5, kernel_size=3, method='iResNet'),
                       INN.BatchNorm2d(5),
                       INN.Reshape(shape_in=(5,4,4), shape_out=(80,)))
model[1].running_var *= torch.exp(torch.randn(1))
TestJacobian(model, shape=(5, 4, 4))

print('#'*8 + ' Linear2d ' + '#'*8)
model = INN.Sequential(INN.Linear2d(5),
                       INN.BatchNorm2d(5),
                       INN.PixelShuffle2d(2),
                       INN.Reshape(shape_in=(20,2,2), shape_out=(80,)))
model[1].running_var *= torch.exp(torch.randn(1))
TestJacobian(model, shape=(5, 4, 4))