import torch
import unittest

def forward_test_normal(model, x):
    model.computing_p(True)
    y, logp, logdet = model(x)

def forward_test_not_compute_p(model, x):
    model.computing_p(False)
    y = model(x)

def inverse_test(model, x):
    model.computing_p(False)
    y = model(x)
    x_hat = model.inverse(y).detach()

    diff = torch.mean((x - x_hat) ** 2) ** 0.5
    #print(f'diff={diff.item()}')
    assert diff.item() < 1e-5

def forward_test(model, x):
    forward_test_normal(model, x)
    forward_test_not_compute_p(model, x)


class BasicTest(unittest.TestCase):
    def forward(self, model, x):
        forward_test(model, x)
    
    def inverse(self, model, x):
        inverse_test(model, x)
    
    def cpu_test(self, model, x):
        model.cpu()
        x = x.cpu()
        self.forward(model, x)
        self.inverse(model, x)
    
    def cuda_test(self, model, x, dev='cuda:0'):
        if not torch.cuda.is_available():
            return 0
        
        model.to(dev)
        x = x.to(dev)
        self.forward(model, x)
        self.inverse(model, x)
