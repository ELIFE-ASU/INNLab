import unittest
import INN
import torch


class TestRescale(unittest.TestCase):
    def test_mix_forward(self):
        model = INN.Sequential(INN.Nonlinear(dim=8, method='NICE'), INN.ResizeFeatures(8, 1))

        x = torch.randn(10, 8)
        y, logp, logdet = model(x)
        assert y.shape[0] == 10 and y.shape[1] == 1
    
    def test_forward(self):
        model = INN.ResizeFeatures(4, 2)
        x = torch.randn(10, 4)
        y, logp, logdet = model(x)

        assert logdet == 0