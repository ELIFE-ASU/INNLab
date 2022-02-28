import unittest
import INN
import torch
from .inn_module_test import forward_test, inverse_test, BasicTest


class TestSequential(BasicTest):
    def test_mix_nonlinear(self):
        model = INN.Sequential(INN.Nonlinear(8, method='NICE'),
                               INN.Nonlinear(8, method='RealNVP'),
                               INN.Nonlinear(8, method='ResFlow'))
        x = torch.ones(16, 8)
        self.cpu_test(model, x)
        self.cuda_test(model, x)
    
    def test_mix_conv1d(self):
        model = INN.Sequential(INN.Conv1d(2, 3, method='NICE'),
                               INN.Conv1d(2, 3, method='RealNVP'),
                               INN.Conv1d(2, 3, method='ResFlow'))
        x = torch.ones(16, 1, 8)
        self.cpu_test(model, x)
        self.cuda_test(model, x)
    
    def test_mix_conv2d(self):
        model = INN.Sequential(INN.Conv2d(2, 3, method='NICE'),
                               INN.Conv2d(2, 3, method='RealNVP'),
                               INN.Conv2d(2, 3, method='ResFlow'))
        x = torch.ones(16, 1, 8, 8)
        self.cpu_test(model, x)
        self.cuda_test(model, x)