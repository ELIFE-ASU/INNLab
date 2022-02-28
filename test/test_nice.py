import unittest
import INN
import torch
from .inn_module_test import forward_test, inverse_test

class TestNonlinear(unittest.TestCase):
    def test_forward(self):
        model = INN.Nonlinear(dim=2, method='NICE')
        x = torch.Tensor([[1, 2], [3, 4]])

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.Nonlinear(dim=2, method='NICE')
        x = torch.Tensor([[1, 2], [3, 4]])
        inverse_test(model, x)


class TestConv1d(unittest.TestCase):
    def test_forward(self):
        model = INN.Conv1d(2, 3, method='NICE')
        assert not isinstance(model, INN.Conv1d_old)
        x = torch.ones(16, 2, 64)

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.Conv1d(2, 3, method='NICE')
        x = torch.ones(16, 2, 64)

        inverse_test(model, x)


class TestConv2d(unittest.TestCase):
    def test_forward(self):
        model = INN.Conv2d(2, 3, method='NICE')
        assert not isinstance(model, INN.Conv2d_old)
        x = torch.ones(16, 2, 16, 16)

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.Conv2d(2, 3, method='NICE')
        x = torch.ones(16, 2, 16, 16)
        
        inverse_test(model, x)