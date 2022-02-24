import unittest
from INN.NICEModel import LinearNICE
import torch
from .inn_module_test import forward_test, inverse_test

class TestNice(unittest.TestCase):
    def test_forward(self):
        model = LinearNICE(dim=2)
        x = torch.Tensor([[1, 2], [3, 4]])

        forward_test(model, x)
    
    def test_inverse(self):
        model = LinearNICE(dim=2)
        x = torch.Tensor([[1, 2], [3, 4]])
        inverse_test(model, x)