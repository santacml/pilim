from inf.pimlp import *
import os
import torch
import inf.dynamicarray as dynamicarray
from inf.dynamicarray import DynArr, CycArr

# command for making a small network for debugging...
# python -m cifar10.cifar10infmlp --lr 1.0 --gclip-per-param --gclip 0.4 --lr-drop-ratio 0.15 --lr-drop-milestones 40 --scheduler multistep --wd 0.00001 --r 5 --batch-size 8 --epochs 1 --width 0  --seed 0  --depth 2 --bias-alpha 0.5 --first-layer-lr-mult 0.1 --last-layer-lr-mult 4.0 --first-layer-alpha 1.0 --last-layer-alpha 0.5 --no-apply-lr-mult-to-wd --save-dir ./output/ --float --human --train-subset-size=100 --save-model

import sys

sys.modules['dynamicarray'] = dynamicarray



if __name__ == "__main__":
    # orig_net_path = r"C:\Users\misantac\Downloads\pilimit_orig_imagenet_r200.th"
    orig_net_path = r"./checkpoint.th"

    net_dir = os.path.dirname(orig_net_path)
    net_name = os.path.basename(orig_net_path)
    converted_net_name = os.path.join(net_dir, "converted_" + net_name)
    if os.path.exists(converted_net_name):
        print("Converted network exists, exiting...")
        0/0

    try:
        orig_net = torch.load(orig_net_path,  map_location=torch.device('cpu'))
    except:
      print("using alternate loading method")
      din = 32**2 * 3

      num_cls = 10

      dout = num_cls
      #
      orig_net = InfPiMLP(d=din, dout=dout, L=2, r=200,
                        first_layer_alpha=1,
                        last_layer_alpha=1,
                        initbuffersize=1000, device="cuda",
                        # bias_alpha=0.5,
                        # last_bias_alpha=0.5,
                        bias_alpha=1,
                        last_bias_alpha=1,
                        _last_layer_grad_no_alpha=True)

      orig_net.load(orig_net_path)


    print(orig_net)

    output_dict = {
        "first_layer_alpha": orig_net.first_layer_alpha,
        "last_layer_alpha": orig_net.last_layer_alpha,
        "layernorm": orig_net.layernorm,
        "bias_alpha": orig_net.bias_alpha,
        "last_bias_alpha": orig_net.last_bias_alpha,
        "As": {
            1: orig_net.As[1]
        },
        "Amult": {},
        "Bs": {},
        "biases": {
            1: orig_net.biases[1]
        },
    }

    
    for l in range(2, orig_net.L+2):
        output_dict["As"][l] = orig_net.As[l].a
        output_dict["Amult"][l] = orig_net.Amult[l].a
        output_dict["Bs"][l] = orig_net.Bs[l].a
        output_dict["biases"][l] = orig_net.biases[l]


    torch.save(output_dict, converted_net_name)



