from inf.kernel import InfReLUGPModel, InfReLUNTKModel
import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose

from meta.cached_omniglot import Omniglot as Omniglot
from meta.maml.model import MetaFinReLUGPModel, ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid, ModelMLPOmniglot, MetaFinPiMLP, ModelLinMLPOmniglot, ModelInfLin1LPOmniglot, MetaMLPModel, MetaFinGP1LP
from meta.maml.utils import ToTensor1D

from inf.inf1lp import InfGP1LP, InfNTK1LP
from examples.networks import FinPiMLPSample
from examples.meta import MetaInfMLP

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')

def get_benchmark_by_name(name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None,
                          normalize=None,
                          seed=None,
                          depth=2,
                          infnet_r=100,
                          readout_zero_init=False,
                          bias_alpha=16,
                          last_bias_alpha=0,
                          first_layer_alpha=1,
                          infnet_device='cuda',
                          orig_model=False,
                          lin_model=False,
                          sigma1=1,
                          sigma2=1,
                          sigmab=1,
                          train_last_layer_only=False,
                          gp1lp=False,
                          ntk1lp=False,
                          gp=False,
                          ntk=False,
                          varw=1,
                          varb=0,
                          last_layer_lr_mult=1,
                          first_layer_lr_mult=1,
                          bias_lr_mult=1):
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    if name == 'sinusoid':
        transform = ToTensor1D()

        meta_train_dataset = Sinusoid(num_shots + num_shots_test,
                                      num_tasks=1000000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform,
                                      seed=seed)
        meta_val_dataset = Sinusoid(num_shots + num_shots_test,
                                    num_tasks=1000000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform,
                                    seed=seed)
        meta_test_dataset = Sinusoid(num_shots + num_shots_test,
                                     num_tasks=1000000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform,
                                     seed=seed)
        if seed is not None:
            import random; random.seed(seed)
            import torch; torch.manual_seed(seed)
            import numpy as np; np.random.seed(seed)
        if hidden_size < 0:
            model = MetaInfMLP(1, 1, infnet_r, depth, device='cpu',
                            first_layer_alpha=first_layer_alpha,
                            last_layer_alpha=1,
                            bias_alpha=bias_alpha,
                            last_bias_alpha=last_bias_alpha,
                            layernorm=False)
        else:
            infnet = MetaInfMLP(1, 1, infnet_r, depth, device='cpu',
                            first_layer_alpha=first_layer_alpha,
                            last_layer_alpha=1,
                            bias_alpha=bias_alpha,
                            last_bias_alpha=last_bias_alpha,
                            layernorm=False)
            model = infnet.sample(hidden_size, fincls=MetaFinPiMLP)  # misantac todo metafinmlp
            
            model = FinPiMLPSample(infnet, hidden_size)
            model.cuda()
        loss_function = F.mse_loss

    elif name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)
        meta_val_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)
        if seed is not None:
            import random; random.seed(seed)
            import torch; torch.manual_seed(seed)
            import numpy as np; np.random.seed(seed)
        if gp1lp:
            model = InfGP1LP(28**2, num_ways, varw=sigma1**2, varb=sigmab**2, initbuffersize=10000)
            if hidden_size >= 0:
                model = model.sample(hidden_size, fincls=MetaFinGP1LP)
            model = model.cuda()
        elif gp:
            model = InfReLUGPModel([varw]*depth, [varb]*depth, 28**2, num_ways, initbuffersize=10000)
            if hidden_size >= 0:
                model = model.sample(hidden_size, fincls=MetaFinReLUGPModel)
            model = model.cuda()
        elif ntk1lp:
            model = InfNTK1LP(28**2, num_ways, varw=sigma1**2, varb=sigmab**2, varw2=sigma2**2, initbuffersize=10000)
            if hidden_size >= 0:
                raise NotImplementedError
            model = model.cuda()
        elif ntk:
            model = InfReLUNTKModel([varw]*(depth+1), [varb]*(depth+1),
                [first_layer_lr_mult] + [1]*(depth-1) + [last_layer_lr_mult],
                [bias_lr_mult] * (depth+1),
                28**2, num_ways, initbuffersize=10000)
            if hidden_size >= 0:
                raise NotImplementedError()
                model = model.sample(hidden_size, fincls=MetaFinReLUGPModel)
            model = model.cuda()
        elif orig_model:
            if lin_model:                
                # print(f'linear model: width {hidden_size}, depth {depth}')
                model = ModelLinMLPOmniglot(num_ways, [hidden_size]*depth, normalize=normalize, bias=bias_alpha!=0, train_last_layer_only=train_last_layer_only)
            else:
                model = MetaMLPModel(28**2, num_ways, [hidden_size]*depth, normalize=normalize,
                      bias=bias_alpha!=0, train_last_layer_only=train_last_layer_only)
        elif lin_model:
            # print(f'linear model: width {hidden_size}, depth {depth}')
            print(f'inflin: alpha={first_layer_alpha}, sigma1={sigma1}, sigma2={sigma2}')
            model = ModelInfLin1LPOmniglot(num_ways, alpha=first_layer_alpha, sigma1=sigma1, sigma2=sigma2, bias_alpha1=bias_alpha, bias_alpha2=last_bias_alpha)
            if hidden_size >= 0:
                print(f'finlin, width {hidden_size}')
                model = model.sample(hidden_size)
            model = model.cuda()
        else:
            # infnet = InfPiMLP(28**2, num_ways, L=depth, r=infnet_r,
            #                 initbuffersize=10000, device=infnet_device, bias_alpha=bias_alpha, last_bias_alpha=last_bias_alpha,
            #                 first_layer_alpha=first_layer_alpha,
            #                 layernorm=normalize=='LN',
            #                 readout_zero_init=readout_zero_init)
            
            readout_zero_init # new output layer auto initializes with 0s
            infnet = MetaInfMLP(28**2, num_ways, infnet_r, depth, device='cpu',
                            first_layer_alpha=first_layer_alpha,
                            last_layer_alpha=1,
                            bias_alpha=bias_alpha,
                            last_bias_alpha=last_bias_alpha,
                            layernorm=normalize=='LN')
            if hidden_size < 0:
                model = infnet
            else:
                model = infnet.sample(hidden_size, fincls=MetaFinPiMLP) # misantac todo metafinmlp
                model.cuda()
        loss_function = F.cross_entropy

    elif name == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = MiniImagenet(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = MiniImagenet(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform)
        meta_test_dataset = MiniImagenet(folder,
                                         transform=transform,
                                         target_transform=Categorical(num_ways),
                                         num_classes_per_task=num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform)

        model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)
