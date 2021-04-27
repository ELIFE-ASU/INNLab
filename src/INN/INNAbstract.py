import torch
import torch.nn as nn
#from INN.utilities import vjp

def vjp(ys, xs, v):
    vJ = torch.autograd.grad(ys, xs, grad_outputs=v, create_graph=True, retain_graph=True, allow_unused=True)
    return tuple([j for j in vJ])

class INNModule(nn.Module):
    def __init__(self):
        super(INNModule, self).__init__()
        self.compute_p = True

    def computing_p(self, b):
        self.compute_p = b
        for sub_m in self.modules():
            if isinstance(sub_m, INNModule):
                #sub_m.computing_p(b)
                sub_m.compute_p = b

# i-ResNet Modules

class iResNetModule(INNModule):
    '''
    Basic Module of i-ResNet
    '''
    def __init__(self):
        super(iResNetModule, self).__init__()
    
    def g(self, x):
        # The g function in ResNet, y = x + g(x)
        pass

    def logdet(self, x, g):
        # Compute log(det J)
        pass
    
    def forward(self, x, log_p0=0, log_det_J_=0):
        if self.compute_p:
            g = self.g(x)
            y = x + g

            log_det_J = log_det_J_ + self.logdet(x, g) 
            log_p = log_p0 
            return y, log_p, log_det_J

        else:
            y = x + self.g(x)
            return y
    
    def inverse(self, y, num_iter=100):
        '''
        The inverse process from y to x
        If dim_in > dim_out, this process will inject noise to the input.
        '''
        orginal_state = self.training

        #y_hat = self.inject_noise(y)

        if self.training:
            self.eval()
        
        with torch.no_grad():
            x = torch.zeros(y.shape).to(y.device)
            for i in range(num_iter):
                x = y - self.g(x)
        
        self.train(orginal_state)
        return x


class Conv(iResNetModule):
    '''
    1-d convolutional i-ResNet
    '''
    def __init__(self, num_iter=1, num_n=3):
        super(Conv, self).__init__()
        
        self.num_iter = num_iter
        self.num_n = num_n
        
        self.net = nn.Sequential()
    
    def g(self, x):
        return self.net(x)
    
    def P(self, y):
        '''
        Normal distribution
        '''
        return 0
    
    def inject_noise(self, y):
        return y
    
    def cut(self, x):
        return x, 0
    
    def logdet(self, x, g):
        batch = x.shape[0]
        self.eval()
        logdet = 0
        for i in range(self.num_iter):
            v = torch.randn(x.shape) # random noise
            v = v.to(x.device)
            w = v
            for k in range(1, self.num_n):
                w = vjp(g, x, w)[0]
                logdet += (-1)**(k+1) * (w * v) / k # groug at the batch level
        
        logdet = logdet.reshape(batch, -1).sum(-1)
        logdet /= self.num_iter
        self.train()
        return logdet


class PixelShuffleModule(INNModule):
    '''
    Module for invertible pixel shuffle
    > Pixel Shuffle: https://arxiv.org/abs/1609.05158

    Override required:
        PixelShuffle(self, x) --> x
        PixelUnshuffle(self, x) --> x
    '''
    def __init__(self):
        super(PixelShuffleModule, self).__init__()
    
    def PixelShuffle(self, x):
        pass
    
    def PixelUnshuffle(self, x):
        pass

    def forward(self, x, log_p0, log_det_J):
        # The log(p_0) and log|det J| will not change under this transformation
        if self.compute_p:
            return self.PixelUnshuffle(x), log_p0, log_det_J
        else:
            return self.PixelUnshuffle(x)
    
    def inverse(self, y, num_iter=100):
        return self.PixelShuffle(y)


class Distribution(nn.Module):

    def __init__(self):
        super(Distribution, self).__init__()
    
    def logp(self, x):
        raise NotImplementedError('logp() not implemented')
    
    def sample(self, shape):
        raise NotImplementedError('sample() not implemented')

    def forward(self, x):
        x = self.logp(x)
        
        return x