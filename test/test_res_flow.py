import unittest
import INN
import torch
from .inn_module_test import forward_test, inverse_test, BasicTest
from torch.nn.utils import spectral_norm
import torch.nn as nn


class TestNonlinear_expe(BasicTest):
    def test_cpu(self):
        model = INN.NonlinearResFlow(2)
        x = torch.Tensor([[1, 2], [3, 4]])
        self.cpu_test(model, x)
        self.cuda_test(model, x)
    
    def test_cuda(self):
        model = INN.NonlinearResFlow(2)
        x = torch.Tensor([[1, 2], [3, 4]])
        self.cuda_test(model, x)


class TestConv1d_expe(unittest.TestCase):
    def test_forward(self):
        model = INN.Conv1dResFlow(2, 1)
        x = torch.ones(16, 2, 16)

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.Conv1dResFlow(2, 1)
        x = torch.ones(16, 2, 16)
        
        inverse_test(model, x)


class TestConv2d_expe(unittest.TestCase):
    def test_forward(self):
        model = INN.Conv2dResFlow(2, 1)
        x = torch.ones(16, 2, 16, 16)

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.Conv2dResFlow(2, 1)
        x = torch.ones(16, 2, 16, 16)
        
        inverse_test(model, x)

# Test ResFlow inside the framwork of INN

class TestNonlinear(unittest.TestCase):
    def test_forward(self):
        model = INN.Nonlinear(dim=2, method='ResFlow')
        x = torch.Tensor([[1, 2], [3, 4]])

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.Nonlinear(dim=2, method='ResFlow')
        x = torch.Tensor([[1, 2], [3, 4]])
        inverse_test(model, x)


class TestConv1d(unittest.TestCase):
    def test_forward(self):
        model = INN.Conv1d(2, 3, method='ResFlow')
        x = torch.ones(16, 2, 64)

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.Conv1d(2, 3, method='ResFlow')
        x = torch.ones(16, 2, 64)

        inverse_test(model, x)


class TestConv2d(unittest.TestCase):
    def test_forward(self):
        model = INN.Conv2d(2, 3, method='ResFlow')
        x = torch.ones(16, 2, 16, 16)

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.Conv2d(2, 3, method='ResFlow')
        x = torch.ones(16, 2, 16, 16)
        
        inverse_test(model, x)


class TestGoukSigma(unittest.TestCase):
    def test_reload(self):
        model = INN.Sequential(INN.NonlinearResFlow(2), INN.NonlinearResFlow(2))
        x = torch.ones(10, 2)
        y, _, _ = model(x)
        model.eval()
        state_dict = model.state_dict()

        new_model = INN.Sequential(INN.NonlinearResFlow(2), INN.NonlinearResFlow(2))
        new_model.eval()
        new_model.load_state_dict(state_dict)

        x_h = new_model(y)
    
    def test_original_bn(self):
        model = spectral_norm(nn.Linear(2, 2))
        x = torch.ones(10, 2)
        y = model(x)

        model.eval()
        state_dict = model.state_dict()

        new_model = spectral_norm(nn.Linear(2, 2))
        new_model.eval()
        new_model.load_state_dict(state_dict)

        y = new_model(x)
    
    def test_reload_consistent(self):
        model = INN.Sequential(INN.NonlinearResFlow(2), INN.NonlinearResFlow(2))
        x = torch.ones(10, 2)
        y, _, _ = model(x)
        model.eval()
        state_dict = model.state_dict()

        new_model = INN.Sequential(INN.NonlinearResFlow(2), INN.NonlinearResFlow(2))
        new_model.eval()
        new_model.load_state_dict(state_dict)

        y2,_,_ = new_model(x)

        assert torch.all(y2 == y)

        new_model = INN.Sequential(INN.NonlinearResFlow(2), INN.NonlinearResFlow(2))
        new_model.eval()

        y2,_,_ = new_model(x)

        assert not torch.all(y2 == y)