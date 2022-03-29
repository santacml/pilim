import unittest
import numpy as np
import tqdm
from meta.train import parse_main
from itertools import product

class TestRuns(unittest.TestCase):

  def test_FOMAML_GP1LP(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 0.2 --grad-clip 0.5 --num-shots-test 1 --normalize None --hidden-size -1 --scheduler multistep --first-order --gp1lp --sigma1 1 --sigmab 1 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    self.assertAlmostEqual(best_value, 0.3760)

  def test_FOMAML_NTK1LP(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 0.05 --grad-clip 0.5 --num-shots-test 1 --normalize None --hidden-size -1 --scheduler multistep --first-order --ntk1lp --sigma1 0.25 --sigmab 1 --sigma2 1 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.481625)
    
  def test_FOMAML_LinNTK1LP(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 0.2 --grad-clip 0.5 --num-shots-test 1 --normalize None --hidden-size 8000 --scheduler multistep --depth 0 --orig-model --lin-model --first-order --train-last-layer-only --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.41750)

  def test_FOMAML_PiNet2000Depth1(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.5 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 8 --grad-clip 0.05 --num-shots-test 1 --normalize None --first-order --hidden-size 2000 --bias-alpha 1 --scheduler multistep --infnet_r 2000 --first-layer-alpha 1 --depth 1 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.4366875112056733)

    
  def test_FOMAML_PiNetInfDepth1(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.5 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 8 --grad-clip 0.05 --num-shots-test 1 --normalize None --first-order --hidden-size -1 --bias-alpha 1 --scheduler multistep --infnet_r 2000 --first-layer-alpha 1 --depth 1 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.4258125114440917)

  def test_FOMAML_PiNetInfDepth2(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 1 --grad-clip 0.5 --num-shots-test 1 --normalize None --first-order --hidden-size -1 --bias-alpha 1 --scheduler multistep --infnet_r 200 --first-layer-alpha 1 --depth 2 --no-adapt-readout --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.3285000073909762)

  def test_FOMAML_OrigAdamBN(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 0.003 --grad-clip 0.5 --num-shots-test 1 --first-order --optimizer adam --orig-model --normalize BN --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.6847500032186506)

  def test_MAML_OrigSGD(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 2 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 20 --grad-clip 0.05 --num-shots-test 1 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.4795625105500221)

  
class TestShortRuns(unittest.TestCase):

  def test_FOMAML_GP1LP(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 0.2 --grad-clip 0.5 --num-shots-test 1 --normalize None --hidden-size -1 --scheduler multistep --first-order --gp1lp --sigma1 1 --sigmab 1 --num-batches 5 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.3900000095367432)

  def test_FOMAML_NTK1LP(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 0.05 --grad-clip 0.5 --num-shots-test 1 --normalize None --hidden-size -1 --scheduler multistep --first-order --ntk1lp --sigma1 0.25 --sigmab 1 --sigma2 1 --num-batches 5 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.49625000953674314)
    
  def test_FOMAML_LinNTK1LP(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 0.2 --grad-clip 0.5 --num-shots-test 1 --normalize None --hidden-size 8000 --scheduler multistep --depth 0 --orig-model --lin-model --first-order --train-last-layer-only --num-batches 5 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.47375001311302184)

  def test_FOMAML_PiNet2000Depth1(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.5 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 8 --grad-clip 0.05 --num-shots-test 1 --normalize None --first-order --hidden-size 2000 --bias-alpha 1 --scheduler multistep --infnet_r 2000 --first-layer-alpha 1 --depth 1 --num-batches 5 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.22750000059604644)

    
  def test_FOMAML_PiNetInfDepth1(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.5 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 8 --grad-clip 0.05 --num-shots-test 1 --normalize None --first-order --hidden-size -1 --bias-alpha 1 --scheduler multistep --infnet_r 2000 --first-layer-alpha 1 --depth 1 --num-batches 5 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.3600000023841858)

  def test_FOMAML_PiNetInfDepth2(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 1 --grad-clip 0.5 --num-shots-test 1 --normalize None --first-order --hidden-size -1 --bias-alpha 1 --scheduler multistep --infnet_r 200 --first-layer-alpha 1 --depth 2 --no-adapt-readout --num-batches 5 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.20000000298023224)

  def test_FOMAML_OrigAdamBN(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 0.003 --grad-clip 0.5 --num-shots-test 1 --first-order --optimizer adam --orig-model --normalize BN --num-batches 5 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.5287500143051147)

  def test_MAML_OrigSGD(self):
    arglst = 'dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 2 --batch-size 32 --num-workers 8 --num-epochs 1 --output-folder results --meta-lr 20 --grad-clip 0.05 --num-shots-test 1 --num-batches 5 --Gproj-inner --Gproj-outer'
    arglst = arglst.split(' ')
    best_value = parse_main(arglst, True, False)
    # print(best_value)
    self.assertAlmostEqual(best_value, 0.26375000476837157)



if __name__ == '__main__':
    # unittest.main()
    unittest.main(TestShortRuns())
    # suite = unittest.TestSuite()
    # suite.addTest(TestRuns('test_MAML_OrigSGD'))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
