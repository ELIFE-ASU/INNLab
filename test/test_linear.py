import unittest
import INN
import torch
from .inn_module_test import forward_test, inverse_test

class TestLinear(unittest.TestCase):
    def test_forward(self):
        model = INN.JacobianLinear(2)
        x = torch.Tensor([[1, 2], [3, 4]])

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.JacobianLinear(2)
        x = torch.Tensor([[1, 2], [3, 4]])
        inverse_test(model, x)

