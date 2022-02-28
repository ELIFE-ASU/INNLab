import unittest
import INN
import torch
from .inn_module_test import forward_test, inverse_test, BasicTest


class TestBNVec(BasicTest):
    def test_cpu(self):
        model = INN.BatchNorm1d(2)
        model.eval()
        x = torch.randn(16, 2)

        self.cpu_test(model, x)
        self.cuda_test(model, x)

    def test_cuda(self):
        model = INN.BatchNorm1d(2)
        model.eval()
        x = torch.randn(16, 2)
        
        self.cuda_test(model, x)


class TestBN1d(BasicTest):
    def test_cpu(self):
        model = INN.BatchNorm1d(2)
        model.eval()
        x = torch.randn(16, 2, 16)

        self.cpu_test(model, x)
        self.cuda_test(model, x)

    def test_cuda(self):
        model = INN.BatchNorm1d(2)
        model.eval()
        x = torch.randn(16, 2, 16)
        
        self.cuda_test(model, x)


class TestBN2d(BasicTest):
    def test_cpu(self):
        model = INN.BatchNorm2d(2)
        model.eval()
        x = torch.randn(16, 2, 16, 16)

        self.cpu_test(model, x)
        self.cuda_test(model, x)

    def test_cuda(self):
        model = INN.BatchNorm2d(2)
        model.eval()
        x = torch.randn(16, 2, 16, 16)
        
        self.cuda_test(model, x)