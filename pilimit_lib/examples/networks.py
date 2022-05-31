from numpy import diff
import torch
from torch import nn
from inf.layers import *
from inf.optim import *
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

        # save as buffers for saving
        self.register_buffer("first_layer_alpha", torch.tensor(first_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("last_layer_alpha", torch.tensor(last_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        if last_bias_alpha is None:
            last_bias_alpha = bias_alpha
        self.register_buffer("last_bias_alpha", torch.tensor(last_bias_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm

        self.layers = nn.ModuleList()

        self.layers.append(InfPiInputLinearReLU(d_in, r, bias_alpha=bias_alpha, device=device))
        for n in range(1, L+1):
            self.layers.append(InfPiLinearReLU(r, device=device, bias_alpha=bias_alpha, layernorm=layernorm, cuda_batch_size=cuda_batch_size))
        
        self.layers.append(InfPiLinearReLU(r, r_out=d_out, output_layer=True, bias_alpha=last_bias_alpha, device=device, layernorm=layernorm, cuda_batch_size=cuda_batch_size))

        
    def forward(self, x):
        '''
        Forward propogate through the infnet with a given x.
        
        Note that:
            - we don't need to define a backward, as torch will do that for us
            - layer alphas are accounted for automatically with autograd
        '''
        for n in range(0, self.L+2):
            x = self.layers[n](x)
            if n == 0: 
                x *= self.first_layer_alpha
            if n == self.L+1: 
                x *= self.last_layer_alpha
        return x
        
        
class MetaInfMLP(PiNet):
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
            device="cpu"):
        super(InfMLP, self).__init__()
        '''
        This class creates a L+2 (for input and output) layer META LEARNING Infinite-width Pi-MLP as specified in our paper: https://openreview.net/forum?id=tUMr0Iox8XW

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

        # save as buffers for saving
        self.register_buffer("first_layer_alpha", torch.tensor(first_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("last_layer_alpha", torch.tensor(last_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        if last_bias_alpha is None:
            last_bias_alpha = bias_alpha
        self.register_buffer("last_bias_alpha", torch.tensor(last_bias_alpha, dtype=torch.get_default_dtype()))
        self.layernorm = layernorm

        self.layers = nn.ModuleList()

        self.layers.append(InfPiInputLinearReLU(d_in, r, bias_alpha=bias_alpha, device=device, return_hidden=True))
        for n in range(1, L+1):
            self.layers.append(InfPiLinearReLU(r, device=device, bias_alpha=bias_alpha, layernorm=layernorm, cuda_batch_size=cuda_batch_size, return_hidden=True))
        
        self.layers.append(InfPiLinearReLU(r, r_out=d_out, output_layer=True, bias_alpha=last_bias_alpha, device=device, layernorm=layernorm, cuda_batch_size=cuda_batch_size, return_hidden=True))

        
    def forward(self, x):
        '''
        Forward propogate through the infnet with a given x.
        
        Note that:
            - we don't need to define a backward, as torch will do that for us
            - layer alphas are accounted for automatically with autograd
        '''
        gs = [] 
        ss = []
        qs = []
        for n in range(0, self.L+2):
            g, s, q = self.layers[n](x)
            if n == 0: 
                g *= self.first_layer_alpha
            if n == self.L+1: 
                g *= self.last_layer_alpha
            gs.append(g.clone())
            if n > 0: # offset by 1 as we don't need ss and qs for last layer - each layer returns s and q for the input
                ss.append(s.clone())
                qs.append(q.clone())

        return gs, ss, qs
        

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

    def forward(self, x):
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
        x = self.layers[self.L+1](x)
        x *= self.last_layer_alpha

        return x

class MetaInfMLPOps(object):
    def __init__(self, infnet):
        '''
        A class for infnet operations specific to meta-learning like gradient loading/deleting, checkpointing, etc.

        Backwards is completely rewritten - in the future, would be nice to use autograd for all this somehow.

        For now, implementation would be difficult, so just do all this manually.

        '''
        self.infnet = infnet


    def zero_grad(self):
        '''
        Delete the stored gradient for the network.
        '''
        self.dAs[0].zero_()
        for l in range(1, self.L+2):
            self.dBs[l] = []
            self.dAs[l] = []
        if self.dbiases is not None:
            for _, v in self.dbiases.items():
                v.zero_()

    def zero_readout_grad(self):
        '''
        Delete only the readout gradient for the network.
        '''
        L = self.infnet.L
        for d in list(self.dAs[L+1]) + list(self.dBs[L+1]):
            d.zero_()
        if self.dbiases is not None:
            for d in list(self.dbiases[L+1]):
                d.zero_()

    def checkpoint(self):
        '''
        Checkpoint all of the parameters in the network.
        '''
        with torch.no_grad():
            self.A1_chkpt = self.infnet.layers[0].A.clone()
            if self.infnet.layers[0].bias is not None:
                # self.biases_chkpt = {k: v.clone() for k, v in self.biases.items()}
                self.biases_chkpt = {}
                for l in range(0, self.L+2):
                    self.biases_chkpt[l] = self.infnet.layers[l].bias.clone()

        for l in range(1, self.L+2):
            self.infnet.layers[l].A.checkpoint()
            self.infnet.layers[l].Amult.checkpoint()
            self.infnet.layers[l].B.checkpoint()

    def restore(self):
        '''
        Restore all of the parameters in the network from the latest checkpoint.
        '''
        self.infnet.layers[0].A[:] = self.A1_chkpt
        for l in range(1, self.L+2):
            self.infnet.layers[l].A.restore()
            self.infnet.layers[l].Amult.restore()
            self.infnet.layers[l].B.restore()
        if self.infnet.layers[0].bias is not None:
            for l in range(0, self.L+2):
                self.infnet.layers[l].bias[:] = self.biases_chkpt[l]

    def assign_intermediate(self, X, gs, gbars, qs, ss, out):
        '''
        Assign intermediate outputs from the network for MAML purposes.
        '''
        self.X = X
        self.gs = gs
        self.gbars = gbars
        self.qs = qs
        self.ss = ss
        self.out = out

    def save_intermediate(self, out_grad):
        '''
        Save intermediate outputs from the network for MAML purposes.
        '''
        self.X_ = self.X
        self.gs_ = self.gs
        self.gbars_ = self.gbars
        self.qs_ = self.qs
        self.ss_ = self.ss
        self.out_ = self.out
        self.out_grad_ = out_grad

    def del_intermediate(self):
        '''
        Delete saved intermediate outputs for the network.
        '''
        del self.X_
        del self.gs_
        del self.gbars_
        del self.qs_
        del self.ss_
        del self.out_
        del self.out_grad_

    def readout_backward(self, delta, buffer=None):
        '''
        Perform gradient backward for only the readout (last) layer.
        '''
        # accumulate gradients
        dAs = self.dAs
        dBs = self.dBs
        dbiases = self.dbiases
        if buffer is not None:
            dAs = buffer[0]
            dBs = buffer[1]
            if self.dbiases is not None:
                dbiases = buffer[2]
        
        L = self.infnet.L
        if self.layernorm:
            s = 1
        else:
            s = self.ss[L+1]
        dAs[L+1].append(delta * s * self.last_layer_alpha)
        dBs[L+1].append(self.gbars[L])
        if self.dbiases is not None:
            dbiases[L+1] += self.last_bias_alpha * self.last_layer_alpha * delta.sum(dim=0)

    def newgradbuffer(self):
        '''
        Create new gradient buffers for MAML.
        '''
        dAs = {0: torch.zeros_like(self.infnet.layers[0].A)}
        dBs = {}
        for l in range(1, self.L+3):
            dAs[l] = []
            dBs[l] = []
        if self.infnet.layers[0].bias is not None:
            dbiases = {}
            for l in range(1, self.L+2):
                dbiases[l] = torch.zeros_like(self.infnet.layers[l].bias)
            return dAs, dBs, dbiases
        return dAs, dBs
        

    def resetbuffer(self, buffer):
        '''
        Reset the gradient buffer for MAML.
        '''
        dAs = buffer[0]
        dBs = buffer[1]
        dAs[0].zero_()
        for l in range(1, self.L+2):
            del dAs[l][:]
            del dBs[l][:]
        if self.infnet.layers[0].bias is not None:
            dbiases = buffer[2]
            for l in range(0, self.L+2):
                dbiases[l].zero_()
            return dAs, dBs, dbiases
        return dAs, dBs

    @torch.no_grad()
    def assign_pi_grad(self):
        '''
        assign pi grads from dAs and dBs into network
        '''
        self.infnet.layers[0].A.grad[:] = self.dAs[0]

        for n in range(1, self.infnet.L+1):
            dA = torch.cat(self.dAs[n])
            dB = torch.cat(self.dBs[n])

            self.infnet.layers[n].A.set_pi_size(dA.shape[0])
            self.infnet.layers[n].A.pi[:] = dA
            self.infnet.layers[n].B.set_pi_size(dB.shape[0])
            self.infnet.layers[n].B.pi[:] = dB


    def backward_somaml(self, delta, buffer=None, readout_fixed_at_zero=False):
        '''
        Input:
        delta: (batch, dout) loss derivative
        buffer: Used for metalearning. If not None, then backprop into `buffer` instead. Should be a pair (dAs, dBs), as returned by `newgradbuffer`.
        readout_fixed_at_zero: no gradient after readout if they are 0
        '''
        # first order backward
        self.backward(delta, buffer=buffer)
        # backward through the step 1 final embeddings in the last layer gradients
        self._backward_somaml(delta, buffer=buffer)
        # # backward through loss derivatives of step 1
        if readout_fixed_at_zero:
        # if readout weights and biases are fixed at 0, then
        # no gradient through loss derivatives of step 1
            return
        L = self.infnet.L
        ckpt = self.infnet.layers[L+1].A.checkpoint_size
        
        # Below, B is test batch and B' was train batch for 2nd order maml
        # shape (B, B')
        q = F00ReLUsqrt(self.qs[L+1][:, ckpt:], self.ss_[L].T, self.ss[L])
        # multiply by the multipliers on loss derivatives from train batch
        q *=  self.infnet.layers[L+1].Amult[L+1][ckpt:].flatten() * self.last_layer_alpha
        # shape (B', dout)
        c = q.T @ delta
        # self.restore()
        self.infnet.layers[L+1].A.restore()
        self.infnet.layers[L+1].Amult.restore()
        self.infnet.layers[L+1].B.restore()
        # shape (B', dout)
        delta2 = torch.autograd.grad(
                    self.out_grad_,
                    [self.out_],
                    c)[0].detach()
        self._backward(delta2, buffer=buffer,
                    gbars=self.gbars_, ss=self.ss_, gs=self.gs_, qs=self.qs_,
                    X=self.X_)

    def _backward_somaml(self, delta, buffer=None):
        '''
        Backprop through the step 1 final embeddings in the last layer gradients
        '''
        self._backward(delta, buffer,
                    gbars=self.gbars_, ss=self.ss_, gs=self.gs_, qs=self.qs_,
                    X=self.X_, somaml=True)

    def backward(self, delta, buffer=None):
        '''
        Call backwards.

        Input:
        delta: (batch, dout) loss derivative
        buffer: Used for metalearning. If not None, then backprop into `buffer` instead. Should be a pair (dAs, dBs), as returned by `newgradbuffer`.
        '''
        self._backward(delta, buffer)

    def _backward(self, delta, buffer=None,
                    gbars=None, ss=None, gs=None, qs=None,
                    As=None, Bs=None, X=None,
                    somaml=False):
        '''
        Perform backpropogation on the infinite-width network. 
        Note that this requires the saved items from the forward pass (gs, ss, qs).

        There will be minimal comments in this function. For a more in-depth explanation, see pilimit_lib.

        Input:
        delta: (batch, dout) loss derivative
        buffer: Used for metalearning. If not None, then backprop into `buffer` instead. Should be a pair (dAs, dBs), as returned by `newgradbuffer`.
        '''
        L = self.L
        self.dgammas = {}
        self.dalpha02s = {}
        self.dalpha11s = {}
        self.dbeta02s = {}
        self.dbeta11s = {}
        self._dAs = {}
        if gbars is None:
            gbars = self.gbars
        if ss is None:
            ss = self.ss
        if gs is None:
            gs = self.gs
        if qs is None:
            qs = self.qs
        if As is None:
            # As = self.As
            As = {}
            for n in range(0, self.infnet.L+2):
                As[n] = self.infnet.layers[n].A.clone()
        if Bs is None:
            # Bs = self.Bs
            Bs = {}
            for n in range(1, self.infnet.L+2):
                Bs[n] = self.infnet.layers[n].B.clone()
        if X is None:
            X = self.X

        # (B, M)
        if not somaml:
            self.dgammas[L+1] = delta @ As[L+1].T \
                *  self.infnet.layers[L+1].Amult.type_as(delta) * self.last_layer_alpha
        else:
            ckpt = As[L+1]._checkpoint
            # (B', B)
            self.dgammas[L+1] = (
                # (B, dout)
                delta \
                # (dout, B')
                @ self.out_grad_.detach().T \
                # Amult contains lr used in train batch
                * self.infnet.layers[L+1].Amult[ckpt:].type_as(delta) \
                # 1 copy from train backprop, 1 copy from test backprop
                * self.last_layer_alpha**2
            # using self.ss here for the ss on test batch
                    # shape (B)
            ).T * self.ss[L].flatten()

        for l in range(L+1, 1, -1):
            if self.layernorm:
                s = 1
                g = gbars[l-1]
            else:
                s = ss[l-1]
                g = gs[l-1]
            if l == L+1 and somaml:
                # using self.gbars here for gbars on test batch
                # (B, r)
                B = self.gbars[L]
                # (B', B)
                # q = gbars[l-1] @ B.T
                q = self.qs[l][:, ckpt:].T
            else:
                # (M, r)
                B = Bs[l]
                q = qs[l]
            # (B', M')
            self.dalpha02s[l] = F02ReLUsqrt(q, 1, s)
            # (B', M')
            self.dalpha11s[l] = F11ReLUsqrt(q, 1, s)
            # (B', r)
            self.dbeta11s[l-1] = (self.dgammas[l] * self.dalpha11s[l]) @ B
            # (B', M)
            self.dbeta02s[l-1] = torch.einsum('bm,bm,br->br', self.dalpha02s[l], self.dgammas[l], g)
            # (B', M)
            self._dAs[l-1] = (self.dbeta11s[l-1] + self.dbeta02s[l-1])
            if self.layernorm:
                drho = torch.einsum('br,br->b', self._dAs[l-1], g)
                self._dAs[l-1] -= drho[:, None] * g
                self._dAs[l-1] /= ss[l-1]
            if l > 2:
                self.dgammas[l-1] = self._dAs[l-1] @ As[l-1].T \
                    * self.infnet.layers[l-1].Amult.type_as(delta)

        
        # accumulate gradients
        dAs = self.dAs
        dBs = self.dBs
        dbiases = self.dbiases
        if buffer is not None:
            dAs = buffer[0]
            dBs = buffer[1]
            if self.dbiases is not None:
                dbiases = buffer[2]
        # (B', r)
        dAs[1] += X.T @ self._dAs[1] * self.first_layer_alpha
        for l in range(2, L+2):
            if l == L+1 and somaml:
                continue
            if self.layernorm:
                s = 1
            else:
                s = ss[l-1]
            if l == L+1:
                if self._last_layer_grad_no_alpha:
                    mul = 1
                else:
                    mul = self.last_layer_alpha
                    dAs[l].append(delta * s * mul)
            else:
                dAs[l].append(self._dAs[l] * s)
            dBs[l].append(gbars[l-1])
        if self.dbiases is not None:
            for l in range(1, L+1):
                dbiases[l] += self.bias_alpha * self._dAs[l].sum(dim=0) * (
                self.first_layer_alpha if l == 1 else 1)
            if not somaml:
                dbiases[L+1] += self.last_bias_alpha * self.last_layer_alpha * delta.sum(dim=0)
