import torch
from torch import nn
from pilimit_lib.inf.layers import InfPiInputLinearReLU, InfPiLinearReLU
from experiments.networks.networks import InfMLP

# python -m cifar10.cifar10infmlp --lr 1.0 --gclip-per-param --gclip 0.4 --lr-drop-ratio 0.15 --lr-drop-milestones 40 --scheduler multistep --wd 0.00001 --r 5 --batch-size 8 --epochs 1 --width 0  --seed 0  --depth 2 --bias-alpha 0.5 --first-layer-lr-mult 0.1 --last-layer-lr-mult 4.0 --first-layer-alpha 1.0 --last-layer-alpha 0.5 --no-apply-lr-mult-to-wd --save-dir ./output/ --float --human --train-subset-size=100 --save-model

# python -m experiments.cifar10.cifar10test --load-model-path C:\repos\pilim\pilimit_orig\output\checkpoints\converted_epoch1.th --r 5 --first-layer-alpha 1.0 --last-layer-alpha 0.5 --depth 1 --bias-alpha 0.5 --load-from-pilimit-orig --float --test-kernel    

if __name__ == "__main__":
    net = InfMLP(d_in=32*32*3, d_out=10, r=5, L=1)

    orig_net = torch.load(r"C:\repos\pilim\pilimit_orig\output\checkpoints\converted_epoch1.th")

    # print(orig_net)

    net.load_pilimit_orig_net(orig_net)