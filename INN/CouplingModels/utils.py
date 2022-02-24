import torch
import torch.nn as nn

class _default_1d_coupling_function(nn.Module):
    def __init__(self, channels, kernel_size, activation_fn=nn.ReLU, w=4):
        super(_default_1d_coupling_function, self).__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f'kernel_size must be an odd number, but got {kernel_size}')
        r = kernel_size // 2

        self.activation_fn = activation_fn
        
        self.f = nn.Sequential(nn.Conv1d(channels, channels*w, kernel_size, padding=r),
                               activation_fn(),
                               nn.Conv1d(w*channels, w*channels, kernel_size, padding=r),
                               activation_fn(),
                               nn.Conv1d(w*channels, channels, kernel_size, padding=r)
                              )
        self.f.apply(self._init_weights)
    
    def _init_weights(self, m):
        nonlinearity = 'leaky_relu' # set to leaky_relu by default

        if self.activation_fn is nn.LeakyReLU:
            nonlinearity = 'leaky_relu'
        if self.activation_fn is nn.ReLU:
            nonlinearity = 'relu'
        if self.activation_fn is nn.SELU:
            nonlinearity = 'selu'
        if self.activation_fn is nn.Tanh:
            nonlinearity = 'tanh'
        if self.activation_fn is nn.Sigmoid:
            nonlinearity = 'sigmoid'
        
        if type(m) == nn.Conv1d:
            # doing xavier initialization
            # NOTE: Kaiming initialization will make the output too high, which leads to nan
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.f(x)


class _default_2d_coupling_function(nn.Module):
    def __init__(self, channels, kernel_size, activation_fn=nn.ReLU, w=4):
        super(_default_2d_coupling_function, self).__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f'kernel_size must be an odd number, but got {kernel_size}')
        r = kernel_size // 2

        self.activation_fn = activation_fn
        
        self.f = nn.Sequential(nn.Conv2d(channels, channels * w, kernel_size, padding=r),
                               activation_fn(),
                               nn.Conv2d(w * channels, w * channels, kernel_size, padding=r),
                               activation_fn(),
                               nn.Conv2d(w * channels, channels, kernel_size, padding=r)
                              )
        self.f.apply(self._init_weights)
    
    def _init_weights(self, m):
        nonlinearity = 'leaky_relu' # set to leaky_relu by default

        if self.activation_fn is nn.LeakyReLU:
            nonlinearity = 'leaky_relu'
        if self.activation_fn is nn.ReLU:
            nonlinearity = 'relu'
        if self.activation_fn is nn.SELU:
            nonlinearity = 'selu'
        if self.activation_fn is nn.Tanh:
            nonlinearity = 'tanh'
        if self.activation_fn is nn.Sigmoid:
            nonlinearity = 'sigmoid'
        
        if type(m) == nn.Conv2d:
            # doing xavier initialization
            # NOTE: Kaiming initialization will make the output too high, which leads to nan
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.f(x)