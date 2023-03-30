from numpy import diff
import torch
from torch import nn
from pilimit_lib.inf.layers import *
from pilimit_lib.inf.optim import *
from torchvision import models

class PiNet(torch.nn.Module):
    '''
    This is a base class only to help with easy loading of pi-networks. 
    Resizing parameters is necessary before loading.
    '''

    def load_state_dict(self, state_dict):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in state_dict.keys() and state_dict[name].shape != param.shape:
                    print("resizing", name)
                    param.resize_(state_dict[name].size())
        
        super().load_state_dict(state_dict)

class InfMLP(PiNet):
    def __init__(
            self, 
            d_in, 
            d_out, 
            r, 
            L, 
            first_layer_alpha=1, 
            last_layer_alpha=1, 
            bias_alpha=1, 
            last_bias_alpha=None, 
            layernorm=False, 
            cuda_batch_size=None, 
            return_hidden=False,
            device="cpu"):
        super(InfMLP, self).__init__()
        '''
        This class creates a L+2 (for input and output) layer Infinite-width Pi-MLP as specified in our paper: https://openreview.net/forum?id=tUMr0Iox8XW

        We create this class as example usage of the inf-width layers. 

        Inputs:
            d_in: dim of input
            d_out: dim of output
            r: rank of probability space
            L: number of hidden layers
            first_layer_alpha: scalar to multiply to layer outputs
            last_layer_alpha: different scalar to multiply to last layer outptus
            bias_alpha: scalar to multiply to bias  
            last_bias_alpha: different scalar to multiply to bias of last layer
            layernorm: use layernorm in between layers (not used in paper)
            cuda_batch_size: batch data to cuda in chunks (not used in paper)
            device: torch device to use
        '''

        self.d_in = d_in
        self.d_out = d_out
        self.r = r
        self.L = L
        self.return_hidden = return_hidden

        # save as buffers for saving
        self.register_buffer("first_layer_alpha", torch.tensor(first_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("last_layer_alpha", torch.tensor(last_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        if last_bias_alpha is None:
            last_bias_alpha = bias_alpha
        self.register_buffer("last_bias_alpha", torch.tensor(last_bias_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm

        self.layers = nn.ModuleList()

        self.layers.append(InfPiInputLinearReLU(d_in, r, bias_alpha=bias_alpha, device=device, return_hidden=return_hidden))
        for n in range(1, L+1):
            self.layers.append(InfPiLinearReLU(r, device=device, bias_alpha=bias_alpha, layernorm=layernorm, cuda_batch_size=cuda_batch_size, return_hidden=return_hidden))
        
        self.layers.append(InfPiLinearReLU(r, r_out=d_out, output_layer=True, bias_alpha=last_bias_alpha, device=device, layernorm=layernorm, cuda_batch_size=cuda_batch_size, return_hidden=return_hidden))

        
    def forward(self, x, save_kernel_output=False):
        '''
        Forward propogate through the infnet with a given x.
        
        Note that:
            - we don't need to define a backward, as torch will do that for us
            - layer alphas are accounted for automatically with autograd
        '''
        if self.return_hidden:
            for n in range(0, self.L+2):
                x, gbar_in, q_in, s_in  = self.layers[n](x)

                if n == self.L+1 and save_kernel_output:
                    kernel_g = x.clone()
                    kernel_gbar = kernel_g / kernel_g.norm(dim=1, keepdim=True)

                if n == 0: 
                    x *= self.first_layer_alpha
                if n == self.L+1: 
                    x *= self.last_layer_alpha
            
            if save_kernel_output:
                return x, kernel_g, kernel_gbar
            else:
                return x

        else:
            assert not save_kernel_output, "Cannot save kernel outputs without returning hidden states."

            for n in range(0, self.L+2):
                x = self.layers[n](x)
                if n == 0: 
                    x *= self.first_layer_alpha
                if n == self.L+1: 
                    x *= self.last_layer_alpha
            return x
    
    @torch.no_grad()
    def load_pilimit_orig_net(self, orig_net):
        assert len(self.layers) == len(orig_net["As"].keys())
        assert self.r == orig_net["As"][1].shape[1]

        self.first_layer_alpha = torch.tensor(orig_net["first_layer_alpha"])
        self.last_layer_alpha = torch.tensor(orig_net["last_layer_alpha"])
        self.layernorm = orig_net["layernorm"]
        self.bias_alpha = torch.tensor(orig_net["bias_alpha"])
        self.last_bias_alpha = torch.tensor(orig_net["last_bias_alpha"])

        self.layers[0].A[:] = orig_net["As"][1]
        if self.bias_alpha != 0:
            self.layers[0].bias[:] = orig_net["biases"][1]
        self.layers[0].bias_alpha = self.bias_alpha
        self.layers[0].layernorm = self.layernorm


        for l in range(1, self.L+2):
            self.layers[l].A.resize_(orig_net["As"][l+1].size())
            self.layers[l].A[:] = orig_net["As"][l+1]

            self.layers[l].Amult.resize_(orig_net["Amult"][l+1].size())
            self.layers[l].Amult[:] = orig_net["Amult"][l+1]

            self.layers[l].B.resize_(orig_net["Bs"][l+1].size())
            self.layers[l].B[:] = orig_net["Bs"][l+1]

            if self.bias_alpha != 0:
                self.layers[l].bias[:] = orig_net["biases"][l+1]
            self.layers[l].bias_alpha = self.bias_alpha
            self.layers[l].layernorm = self.layernorm

        print("Finished loading from converted pilimit_orig net.")



class FinPiMLPSample(torch.nn.Module):
    def __init__(self, infnet, n):
        '''
        Sample an infinite-width network using a finite n.

        Inputs:
            infnet: infnet to use
            n: size of hidden dimension to sample to

        Note that all other parameters (alphas, r, L, etc) are obtained from the infinite-width network.
        '''
        super(FinPiMLPSample, self).__init__()

        self.d_in = infnet.d_in
        self.d_out = infnet.d_out
        self.r = infnet.r
        self.L = infnet.L
        self.first_layer_alpha = infnet.first_layer_alpha
        self.last_layer_alpha = infnet.last_layer_alpha
        self.bias_alpha = infnet.bias_alpha
        self.last_bias_alpha = infnet.last_bias_alpha
        self.layernorm = infnet.layernorm
        self.n = n

        self.layers = nn.ModuleList()

        self.layers.append(infnet.layers[0].sample(self.d_in, self.n))

        for n in range(1, self.L+1):
            self.layers.append(infnet.layers[n].sample(self.n, self.n, self.layers[n-1].omega))

        self.layers.append(infnet.layers[self.L+1].sample(self.n, self.d_out, self.layers[self.L].omega))

    def forward(self, x, save_kernel_output=False):
        '''
        Forward propogate through the finite network with a given x.
        '''
        for n in range(0, self.L+1):
            x = self.layers[n](x)
            if n == 0: 
                x = x * self.first_layer_alpha
            if self.layernorm:
                x = divbystd(x)
            x = nn.ReLU()(x)

        if save_kernel_output: # for imagenet transfer kernel creation
            kernel_output = x.clone()
        
        x = self.layers[self.L+1](x)
        x *= self.last_layer_alpha

        if save_kernel_output:
            return x, kernel_output

        return x