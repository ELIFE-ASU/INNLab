import unittest
import INN
import torch
from .inn_module_test import forward_test, inverse_test

class TestEUNN(unittest.TestCase):
    def test_forward(self):
        model = INN.EUNN(2)
        x = torch.Tensor([[1, 2], [3, 4]])

        forward_test(model, x)
    
    def test_inverse(self):
        model = INN.EUNN(2)
        x = torch.Tensor([[1, 2], [3, 4]])
        inverse_test(model, x)
    
    def test_get_matrix(self):
        model = INN.EUNN(2)
        matrix = model.get_matrix()
        self.assertEqual(matrix.shape, (2, 2))
    
    def test_get_matrix_cuda(self):
        model = INN.EUNN(2)
        model.cuda()
        matrix = model.get_matrix()
        self.assertEqual(matrix.shape, (2, 2))