import unittest
import INN
import torch
from .inn_module_test import forward_test, inverse_test, BasicTest


class TestNonlinear_expe(BasicTest):
    def test_cpu(self):
        model = INN.ResFlowLinear(2)
        x = torch.Tensor([[1, 2], [3, 4]])
        self.cpu_test(model, x)
        self.cuda_test(model, x)
    
    def test_cuda(self):
        model = INN.ResFlowLinear(2)
        x = torch.Tensor([[1, 2], [3, 4]])
        self.cuda_test(model, x)


class TestConv2d_expe(unittest.TestCase):
    def test_forward(self):
        model = INN.ResFlowConv2d(2, 1)
        x = torch.ones(16, 2, 16, 16)

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.ResFlowConv2d(2, 1)
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