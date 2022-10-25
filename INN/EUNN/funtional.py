import torch


def _ind1(dim, deepth=0):
    if dim % 2 != 0:
        raise ValueError('dim must be even')
    idx = torch.arange(dim)

    imag = idx[dim // 2:]
    real = idx[:dim // 2]

    idx = torch.stack([imag, real]).t().reshape(-1)
    
    if deepth % 2 != 0:
        idx = torch.cat([idx[-1:], idx[:-1]])

    return idx

def _ind2(dim, deepth=0):
    if dim % 2 != 0:
        raise ValueError('dim must be even')
    idx = torch.arange(dim)

    even = idx[0::2]
    odd = idx[1::2]

    if deepth % 2 == 1:
        even = torch.cat([even[1:], even[:1]])
        odd = torch.cat([odd[-1:], odd[:-1]])
    
    idx = torch.stack([odd, even]).t().reshape(-1)

    return idx

def _rotation_factor(theta, phi=None, complex_space=False):
    if complex_space:
        v1 = torch.cat([torch.cos(theta), torch.cos(theta) * torch.exp(1j * phi)], dim=-1)
        v2 = torch.cat([torch.sin(theta), -torch.sin(theta) * torch.exp(1j * phi)], dim=-1)
    else:
        v1 = torch.cat([torch.cos(theta), torch.cos(theta)], dim=-1)
        v2 = torch.cat([torch.sin(theta), -torch.sin(theta)], dim=-1)
    return v1, v2

def rotation(x, theta, phi, ind1, ind2, complex_space=False):
    '''Efficenty Unirary Neural Network'''
    v1, v2 = _rotation_factor(theta, phi=phi, complex_space=complex_space)
    
    v1 = v1[ind1].unsqueeze(0)
    v2 = v2[ind1].unsqueeze(0)

    y = v1 * x + v2 * x[:, ind2]

    return y

def unitary(x, thetas, ind1, ind2, phis=None, complex_space=False):
    num_pass, dim = thetas.shape

    for i in range(num_pass):
        phi = phis[i] if phis is not None else None

        x = rotation(x, thetas[i], phi, ind1[i], ind2[i], complex_space=complex_space)
    
    return x