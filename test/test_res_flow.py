import unittest
import INN
import torch
from .inn_module_test import forward_test, inverse_test

class TestNonlinear(unittest.TestCase):
    def test_forward(self):
        model = INN.ResFlowLinear(2)
        x = torch.Tensor([[1, 2], [3, 4]])

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.ResFlowLinear(2)
        x = torch.Tensor([[1, 2], [3, 4]])
        inverse_test(model, x)


class TestConv2d(unittest.TestCase):
    def test_forward(self):
        model = INN.ResFlowConv2d(2, 1)
        x = torch.ones(16, 2, 16, 16)

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.ResFlowConv2d(2, 1)
        x = torch.ones(16, 2, 16, 16)
        
        inverse_test(model, x)