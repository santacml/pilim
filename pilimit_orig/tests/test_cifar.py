import unittest
import numpy as np
import tqdm
from cifar10.cifar10infmlp import main
from itertools import product

class TestRuns(unittest.TestCase):

  def test_PiNetInfDepth2(self):
    arglst = '--float --cuda --epochs 1 --quiet'
    arglst = arglst.split(' ')
    best_value = main(arglst)
    self.assertAlmostEqual(best_value, 0.3867)
    
  def test_PiNet200Depth2(self):
    arglst = '--float --cuda --epochs 1 --width 200 --quiet'
    arglst = arglst.split(' ')
    best_value = main(arglst)
    self.assertAlmostEqual(best_value, 0.4085)

  def test_PiNetInfDepth1(self):
    arglst = '--float --cuda --epochs 1 --depth 1 --quiet'
    arglst = arglst.split(' ')
    best_value = main(arglst)
    self.assertAlmostEqual(best_value, 0.3778)

  def test_PiNet200Depth1(self):
    arglst = '--float --cuda --epochs 1 --width 200 --depth 1 --quiet'
    arglst = arglst.split(' ')
    best_value = main(arglst)
    self.assertAlmostEqual(best_value, 0.375)

if __name__ == '__main__':
    unittest.main()