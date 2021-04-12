import torch
import torch.nn as nn
from utilities import vjp

# i-ResNet Modules

class iResNetModule(nn.Module):
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

    def P(self, y):
        # compute log probability of abandoned information
        pass

    def cut(self, y):
        # resize y
        pass

    def inject_noise(self, y):
        # inject noise to y
        pass
    
    def forward(self, x, log_p0=0, log_det_J_=0):
        if self.training:
            # if in the training mode, we need to compute log(det(J))

            # ResNet: y = x + g(x)
            g = self.g(x)
            y = x + g
            # Resize if dim_out < dim_in
            y, z = self.cut(y) # split y in to output y' and abandoned information z
            
            # compute Jacobian and probability
            log_det_J = log_det_J_ + self.logdet(x, g) # compute log(det|J_i|) and inherit log(det|J_{i-1}|)
            log_p = log_p0 + self.P(z) # compute probability of z, and inherit log(p_0)
            return y, log_p, log_det_J

        else:
            y = x + self.g(x)
            y, z = self.cut(y)
            return y
    
    def inverse(self, y, num_iter=100):
        '''
        The inverse process from y to x
        If dim_in > dim_out, this process will inject noise to the input.
        '''
        orginal_state = self.training

        y_hat = self.inject_noise(y)

        if self.training:
            self.eval()
        
        with torch.no_grad():
            x = torch.zeros(y_hat.shape).to(y_hat.device)
            for i in range(num_iter):
                x = y_hat - self.g(x)
        
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