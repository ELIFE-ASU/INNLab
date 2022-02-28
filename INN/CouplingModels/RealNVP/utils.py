import torch
from INN.INNAbstract import INNModule
from .. import utils


class real_nvp_element(INNModule):
    '''
    The very basic element of real nvp
    '''
    def __init__(self, dim, f_log_s, f_t, mask=None, eps=1e-8, clip=None):
        super(real_nvp_element, self).__init__()

        if mask is None:
            self.mask = utils.generate_mask(dim)
        else:
            self.mask = mask
        
        self.f_log_s = f_log_s
        self.f_t = f_t
        self.eps = eps
        self.clip = clip
    
    def get_s(self, x):
        if len(x.shape) == 1:
            b = self.mask.squeeze().to(x.device)
        else:
            b = self.mask.to(x.device)
        
        log_s = self.f_log_s(b * x)

        if self.clip is not None:
            # clip the log(s), to avoid extremely large numbers
            log_s = self.clip * torch.tanh(log_s / self.clip)
        
        s = torch.exp(log_s)
        return s, log_s

    def forward(self, x):
        if len(x.shape) == 1:
            b = self.mask.squeeze().to(x.device)
        else:
            b = self.mask.to(x.device)
        
        s, log_s = self.get_s(b * x)

        log_det_J = torch.sum(log_s * (1-b), dim=-1)

        t = self.f_t(b * x)

        y = b * x + (1 - b) * (x * (s + self.eps) + t)

        return y, log_det_J
    
    def inverse(self, y):
        if len(y.shape) == 1:
            b = self.mask.squeeze().to(y.device)
        else:
            b = self.mask.to(y.device)
        
        s, log_s = self.get_s(b * y)

        t = self.f_t(b * y)

        x = b * y + (1 - b) * (y - t) / (s + self.eps)

        return x


class combined_real_nvp(INNModule):
    '''
    The very basic element of real nvp
    '''
    def __init__(self, dim, f_log_s, f_t, mask=None, clip=None):
        super(combined_real_nvp, self).__init__()

        if mask is None:
            self.mask = utils.generate_mask(dim)
        else:
            self.mask = mask
        
        self.nvp_1 = real_nvp_element(dim, f_log_s, f_t, mask=self.mask, clip=clip)
        self.nvp_2 = real_nvp_element(dim, f_log_s, f_t, mask=1 - self.mask, clip=clip)

    def forward(self, x):
        x, log_det_J_1 = self.nvp_1(x)
        x, log_det_J_2 = self.nvp_2(x)

        return x, log_det_J_1 + log_det_J_2
    
    def inverse(self, y):
        y = self.nvp_2.inverse(y)
        y = self.nvp_1.inverse(y)

        return y