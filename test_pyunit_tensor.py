#!/usr/bin/python3
import unittest

import leicht

class TestLLAS(unittest.TestCase):
    def test_asum(self):
        pass
    def test_gemm(self):
        pass

class TestTensorMethods(unittest.TestCase):
    def test_creation(self):
        a0 = leicht.fp64Tensor()
        a1 = leicht.fp64Tensor(10)
        a2 = leicht.fp64Tensor(10,10)
        a3 = leicht.fp64Tensor(10,10,10)

if __name__=='__main__':
    unittest.main()
