"""
Copy from: https://github.com/jarrelscy/iResnet/blob/master/SpectralNormGouk.py
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import torch
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class SpectralNorm(object):
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 2
    # At version 2:
    #   used Gouk 2018 method.
    #   will only normalize if largest singular value > magnitude    

    def __init__(self, name='weight', n_power_iterations=1, magnitude=1.0, eps=1e-12):
        self.name = name
        self.magnitude = magnitude
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
    def l2norm(self, t):
        return torch.sqrt((t ** 2).sum())
    def compute_weight(self, module, do_power_iteration, num_iter=0):      
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(max(self.n_power_iterations, num_iter)):
                    u = module.iteration_function(u, weight=weight)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                sv = self.l2norm(module.forward_function(u, weight=weight)) / self.l2norm(u)      
                sigma = F.relu(sv / self.magnitude - 1.0) + 1.0
                module.sigma = sigma
        else:
            sigma = module.sigma
        
        return weight / sigma

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_sigma')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs, n_power_iterations=0):
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training, num_iter=n_power_iterations))

    @staticmethod
    def apply(module, name, n_power_iterations, magnitude, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, magnitude, eps)
        weight = module._parameters[name]
        
        functions_dict = {torch.nn.Conv1d : (F.conv1d, F.conv_transpose1d),
             torch.nn.Conv2d : (F.conv2d, F.conv_transpose2d),
             torch.nn.Conv3d : (F.conv3d, F.conv_transpose3d),
             torch.nn.ConvTranspose1d : (F.conv_transpose1d, F.conv1d),
             torch.nn.ConvTranspose2d : (F.conv_transpose2d, F.conv2d),
             torch.nn.ConvTranspose3d : (F.conv_transpose3d, F.conv3d),            
            }
        
        if isinstance(module, torch.nn.Linear):  
            module.forward_function = lambda inp,weight=weight: F.linear(inp, weight)
            module.iteration_function = lambda inp,weight=weight: F.linear(F.linear(inp, weight), weight.transpose(1,0))
        elif isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d,
                               torch.nn.Conv1d,
                               torch.nn.Conv2d,
                               torch.nn.Conv3d,)):
            k = weight.shape[2:]
            s = module.stride
            g = module.groups
            d = module.dilation
            p = module.padding
            functions = functions_dict[module.__class__ ]
            module.forward_function = lambda inp,weight=weight,s=s,g=g,d=d,p=p: functions[0](inp, weight, stride=s, padding=p, dilation=d, groups=g)            
            module.iteration_function = lambda inp,weight=weight,s=s,g=g,d=d,p=p: functions[1](functions[0](inp, weight, stride=s, padding=p, dilation=d, groups=g), 
                                                                                             weight, stride=s, padding=p, dilation=d, groups=g)
            
            
        with torch.no_grad():
            shape = (1,weight.shape[1])
            for i in range(0,len(weight.shape)-2):
                shape += (max(k[i]*d[i],1),)
            u = torch.randn(shape).to(weight.device)
            

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        sigma = torch.tensor(1).to(weight.device)
        module.register_buffer(fn.name + "_sigma", sigma)

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormLoadStateDictPreHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        pass


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        pass

def spectral_norm(module, name='weight', n_power_iterations=1, magnitude=1.0,eps=1e-12):
    r"""Applies spectral normalization to a parameter in the given module.
    .. math::
         \mathbf{W} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.
    See `Spectral Normalization for Generative Adversarial Networks`_ .
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is 0, except for modules that are instances of
            ConvTranspose1/2/3d, when it is 1
    Returns:
        The original module with the spectal norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40))
        Linear (20 -> 40)
        >>> m.weight_u.size()
        torch.Size([20])
    """    
    SpectralNorm.apply(module, name, n_power_iterations, magnitude, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(
        name, module))