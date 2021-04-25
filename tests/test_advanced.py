import INN

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
    x_hat.requires_grad = True
    model.computing_p(True)
    y, log_p, log_det = model(x_hat)
    
    v = torch.diag(torch.ones(dim)).reshape((dim, *x.shape))
    grad = INN.utilities.vjp(y, x_hat, v)[0]
    
    return grad.detach(), log_det.detach()