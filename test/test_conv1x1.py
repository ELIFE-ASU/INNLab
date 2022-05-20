import unittest
import INN
import torch
from .inn_module_test import forward_test, inverse_test


class TestConv1x1(unittest.TestCase):
    def test_forward(self):
        model = INN.Linear2d(2)
        x = torch.randn(8, 2, 8, 8)

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.Linear2d(2)
        x = torch.randn(8, 2, 8, 8)
        inverse_test(model, x)