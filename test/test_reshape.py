import unittest
import INN
import torch
from .inn_module_test import forward_test, inverse_test, BasicTest


class TestReshape(BasicTest):
    def test_cpu(self):
        model = INN.Reshape((3, 2,2), (12,))
        x = torch.randn(16, 3, 2, 2)
        self.cpu_test(model, x)
        self.cuda_test(model, x)

    def test_cuda(self):
        model = INN.Reshape((3, 2,2), (12,))
        x = torch.randn(16, 3, 2, 2)
        self.cuda_test(model, x)