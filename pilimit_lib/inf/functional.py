import torch
import torch.functional
from torch import nn
from .math import *

class FunBatcher():
    def __init__(self, cuda_batch_size):
        '''
        This class allows for batching an arbitrary function along a given 
        batch dimension to CUDA, using cuda_batch_size.

        It's useful to do this when working with GPUs of small memory. If one has 
        16gb of RAM and only 2gb of GPU memory, it's possible to "chunk"
        a memory-heavy operation in pieces to GPU, so it does not overflow.

        Of course, this is very, very slow.

        We did not use this functionality in our original paper.
        '''

        self.cuda_batch_size = cuda_batch_size

    def batch_fun(self, fun, batch_axis, **kwargs):
        '''
        Perform a function in chunks along batch_axis on a GPU,
        storing everything else in RAM.
        '''
        unbatched_args = [arg.cuda() if isinstance(arg, torch.Tensor) else arg for arg in kwargs["unbatched_args"]]

        m = kwargs["batched_args"][0].shape[batch_axis]

        if batch_axis != 0:
            raise NotImplementedError("Batch axis must be 0 because index_select is slow otherwise.")

        out = None
        
        for n in range(int(m / self.cuda_batch_size) + 1):
            idx1 = int(n*self.cuda_batch_size)
            idx2 = int((n+1)*self.cuda_batch_size)
            if (idx2 > m): idx2 = m
            if idx1 == idx2: break
            
            # batched_args = [torch.index_select(arg, batch_axis, torch.arange(idx1,idx2, dtype=torch.int32, device=arg.device)).cuda() for arg in kwargs["batched_args"]]

            # batched_args = [arg[idx1:idx2].cuda() if isinstance(arg, torch.Tensor) else arg for arg in kwargs["batched_args"]]

            # print([arg.shape for arg in kwargs["batched_args"]])

            # args = batched_args + unbatched_args
            args = [arg[idx1:idx2].cuda() if isinstance(arg, torch.Tensor) else arg for arg in kwargs["batched_args"]] + unbatched_args
            
            if out == None:
                out = fun(*args)
            else:
                out += fun(*args)

            del args
            # del batched_args
            torch.cuda.empty_cache()

        del unbatched_args
        torch.cuda.empty_cache()

        return out

class MultiGPUFunBatcher():
    def __init__(self, cuda_batch_size):
        '''
        A further extension of FunBatcher to operate in multi-gpu environments.

        This is also extraordinarily slow, but faster than the 1 GPU case.

        We did not use this functionality in our original paper.
        '''

        self.cuda_batch_size = cuda_batch_size
        self.device_count = torch.cuda.device_count()
        self.devices = list(range(0, self.device_count))

    def batch_fun(self, fun, batch_axis, **kwargs):
        '''
        Perform a function in chunks along batch_axis on multiple GPUs in parallel,
        storing everything else in RAM.
        '''
        unbatched_args = {}
        for n in self.devices:
            unbatched_args[n] = [arg.to("cuda:" + str(n)) if isinstance(arg, torch.Tensor) else arg for arg in kwargs["unbatched_args"]]


        m = kwargs["batched_args"][0].shape[batch_axis]

        if batch_axis != 0:
            raise NotImplementedError("Batch axis must be 0 because index_select is slow otherwise.")

        out = None
        
        groups = []
        idxs = []
        for n in range(int(m / self.cuda_batch_size) + 1):
            idx1 = int(n*self.cuda_batch_size)
            idx2 = int((n+1)*self.cuda_batch_size)
            if (idx2 > m): idx2 = m
            if idx1 == idx2:
                break
            idxs.append((idx1, idx2))
            if len(idxs) == self.device_count:
                groups.append(idxs.copy())
                idxs = []
        if len(idxs) > 0: groups.append(idxs)

        for group in groups:
            parallel_args = []
            for i, (idx1, idx2) in enumerate(group):
                args = [arg[idx1:idx2].to("cuda:" + str(i)) if isinstance(arg, torch.Tensor) else arg for arg in kwargs["batched_args"]] + unbatched_args[i]
                parallel_args.append(args)

            results = nn.parallel.parallel_apply( [fun] * len(group) , parallel_args, devices=range(len(group)) )
            
            del parallel_args
            # del batched_args
            torch.cuda.empty_cache()

            results = [result.to("cuda:0") for result in results]
            
            if out == None:
                out = sum( results )
            else:
                out += sum( results )


        del unbatched_args
        torch.cuda.empty_cache()

        return out


def inf_linear_forward_batched(A, Amult, B, g, gbar, s):
    g_out = None

    if s is None:
        s = g.norm(dim=1, keepdim=True)

    if gbar is None:
        gbar = g / s

    q = gbar @ B.T

    g_out = (F00ReLUsqrt(q, 1, s) * Amult.type_as(q)) @ A

    # if bias is not None:
    #     g_out += bias * bias_alpha if bias_alpha else bias

    return g_out

def inf_linear_forward_return_q(
        A, 
        Amult, 
        B, 
        g, 
        gbar, 
        s, 
        bias, 
        bias_alpha=None):
    '''
    Perform the forward pass of an inf-width pi layer.

    See the paper for derivation of this function.

    Inputs:
        A: A matrix
        Amult: Amult (stores lr and wd)
        B: B matrix
        g: the incoming inf-width representation of preactivations
        gbar: normed incoming preactivations
        s: standard deviations of the preactivations
        bias: bias to add
        bias_alpha: alpha to multiply by bias
    '''
    g_out = None

    if s is None:
        s = g.norm(dim=1, keepdim=True)

    if gbar is None:
        gbar = g / s
        
    # apply B
    q = gbar @ B.T

    # analogous "activation" of the inf-width net
    g_out = (F00ReLUsqrt(q, 1, s) * Amult.type_as(q)) @ A

    if bias is not None:
        g_out += (bias * bias_alpha).type_as(q) if bias_alpha else bias.type_as(q)

    return g_out, q

def inf_linear_backward(
        A, 
        Amult, 
        B, 
        grad, 
        g, 
        gbar, 
        s, 
        q=None):
    '''
    Perform the backward pass of an inf-width pi layer, 
    using projected gradient descent in inf-width.

    See the paper for derivation of this function.

    Note there is a way to find this that is much easier using actual autograd,
    but we stick with this for the paper.


    Inputs:
        A: A matrix
        Amult: Amult (stores lr and wd)
        B: B matrix
        g: the incoming inf-width representation of preactivations
        gbar: normed incoming preactivations (optional)
        s: standard deviations of the preactivations (optional)
        q: intermediate value from the forward pass to save on compute (optional)
    '''

    dgamma = grad @ (A.T * Amult.type_as(grad) )
    
    if q is None:
        q = gbar @ B.T

    dalpha11 = F11ReLUsqrt(q, 1, s)
    dalpha02 = F02ReLUsqrt(q, 1, s)

    dbeta11s = (dgamma * dalpha11) @ B
    dbeta02s = contract('bm,bm,br->br', dalpha02, dgamma, g)

    return dbeta11s + dbeta02s



def InfPiLinearReLUFunctionBuilder(layernorm=False, cuda_batch_size=None, return_hidden=False):
    '''
    Dynamically create either a regular InfPiLinearReLUFunction,
    or one that batches operations along an axis to the GPU.
    '''

    class InfPiLinearReLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(
                ctx, 
                g_in, 
                A, 
                Amult, 
                B, 
                A_pi, 
                Amult_pi, 
                B_pi, 
                gbar_in=None, 
                s_in=None, 
                bias=None):
            '''
            Perform the forward pass of an inf-width pi layer,
            storing the appropriate matrices for the custom backward pass.

            Inputs:
                ctx: torch context for saving variables
                g_in: incoming preactivations
                A: A matrix
                Amult: Amult (stores lr and wd)
                B: B matrix
                A_pi: pi matrix to store A's gradient
                Amult_pi: pi matrix to store Amult's gradient
                B_pi: pi matrix to store B's gradient
                gbar_in: normed incoming preactivations (optional)
                s_in: standard deviations of the preactivations (optional)
                bias: bias to add 
            '''
            if s_in is None:
                s_in = g_in.norm(dim=1, keepdim=True)

            if gbar_in is None:
                gbar_in = g_in / s_in

            # no gradient on anything but pi vars
            A.requires_grad = False
            Amult.requires_grad = False
            B.requires_grad = False
            gbar_in.requires_grad = False
            s_in.requires_grad = False

            s = s_in
            if layernorm:
                s = 1
            
            # perform forward pass, return q for backwars storage
            g_out, q_in = inf_linear_forward_return_q(A, Amult, B, g_in, gbar_in, s, bias)

            # save everything
            ctx.save_for_backward(g_in, A, Amult, B, A_pi, Amult_pi, B_pi, gbar_in, s_in, bias, q_in)

            # s_out = g_out.norm(dim=1, keepdim=True)
            # gbar_out = g_out / s_out
            # ctx.mark_non_differentiable(s_out, gbar_out)

            if return_hidden:
                return g_out, gbar_in, q_in, s_in
            return g_out

        @staticmethod
        def backward(ctx, grad_g_out):
            '''
            Perform the backward pass of an inf-width pi layer,
            using projected gradient descent in inf-width.
            '''
            # load saved variables
            g_in, A, Amult, B, A_pi, Amult_pi, B_pi, gbar_in, s_in, bias, q_in = ctx.saved_tensors
            grad_g_in = grad_A_in = grad_Amult_in = grad_B_in = grad_A_pi = grad_Amult_pi = grad_B_pi = grad_gbar_in = grad_s_in = grad_bias_in = None

            s = s_in
            g = g_in
            if layernorm:
                s = 1
                g = gbar_in
            
            # perform backwards function
            grad_g_in = inf_linear_backward(A, Amult, B, grad_g_out, g, gbar_in, s, q=q_in)

            if layernorm:
                drho = torch.einsum('br,br->b', grad_g_in, g)
                grad_g_in -= drho[:, None] * g
                grad_g_in /= s_in
            
            # calculate gradients

            # TODO: when do we need this?
            # if ctx.needs_input_grad[0]: # g_in
            #     raise NotImplementedError()
            #     grad_g_in = dbeta11s + dbeta02s

            if ctx.needs_input_grad[1]: # A
                raise NotImplementedError()

            if ctx.needs_input_grad[2]: # Amult
                raise NotImplementedError()
                
            if ctx.needs_input_grad[3]: # B
                raise NotImplementedError()

            if ctx.needs_input_grad[4]: # A pi
                grad_A_pi = grad_g_out * s

            if ctx.needs_input_grad[5]: # Amult pi
                grad_Amult_pi = torch.ones(s_in.shape[0], dtype=Amult.dtype, device=Amult.device)

            if ctx.needs_input_grad[6]: # B pi
                grad_B_pi = gbar_in

            if ctx.needs_input_grad[7]: # gbar_in
                raise NotImplementedError()

            if ctx.needs_input_grad[8]: # s_in
                raise NotImplementedError()

            if bias is not None and ctx.needs_input_grad[9]: # bias
                grad_bias_in = grad_g_out.sum(dim=0)

            # if q_in is not None and ctx.needs_input_grad[10]: # q_in
                # raise NotImplementedError()

            return grad_g_in, grad_A_in, grad_Amult_in, grad_B_in, grad_A_pi, grad_Amult_pi, grad_B_pi, grad_gbar_in, grad_s_in, grad_bias_in

    class BatchedInfPiLinearReLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(
                ctx, 
                g_in, 
                A, 
                Amult, 
                B, 
                A_pi, 
                Amult_pi, 
                B_pi, 
                gbar_in=None,
                s_in=None, 
                bias=None):
            '''
            Perform the forward pass of an inf-width pi layer,
            storing the appropriate matrices for the custom backward pass
            and appropriately batched to the GPU.

            Inputs:
                ctx: torch context for saving variables
                g_in: incoming preactivations
                A: A matrix
                Amult: Amult (stores lr and wd)
                B: B matrix
                A_pi: pi matrix to store A's gradient
                Amult_pi: pi matrix to store Amult's gradient
                B_pi: pi matrix to store B's gradient
                gbar_in: normed incoming preactivations (optional)
                s_in: standard deviations of the preactivations (optional)
                bias: bias to add 
            '''
            if s_in is None:
                s_in = g_in.norm(dim=1, keepdim=True)

            if gbar_in is None:
                gbar_in = g_in / s_in

            A.requires_grad = False
            Amult.requires_grad = False
            B.requires_grad = False
            gbar_in.requires_grad = False
            s_in.requires_grad = False

            s = s_in
            if layernorm:
                s = 1
            
            # g_out, q = inf_linear_forward_return_q(A, Amult, B, g_in, gbar_in, s_in, bias)
            fun_batcher = FunBatcher(cuda_batch_size)
            g_out = fun_batcher.batch_fun(inf_linear_forward_batched, 0, batched_args=[A, Amult.type_as(A), B], unbatched_args=[g_in, gbar_in, s])
            
            # can't include this in batching otherwixe it gets added many times
            if bias is not None: 
                g_out += bias.cuda()


            ctx.save_for_backward(g_in, A, Amult, B, A_pi, Amult_pi, B_pi, gbar_in, s_in, bias)
            # ctx.save_for_backward(g_in, A, Amult, B, A_pi, Amult_pi, B_pi, gbar_in, s_in, bias)


            # s_out = g_out.norm(dim=1, keepdim=True)
            # gbar_out = g_out / s_out
            # ctx.mark_non_differentiable(s_out, gbar_out)

            return g_out.to(g_in.device)

        @staticmethod
        def backward(ctx, grad_g_out):
            '''
            Perform the backward pass of an inf-width pi layer,
            using projected gradient descent in inf-width
            and appropriately batched to the GPU.
            '''
            g_in, A, Amult, B, A_pi, Amult_pi, B_pi, gbar_in, s_in, bias = ctx.saved_tensors
            grad_g_in = grad_A_in = grad_Amult_in = grad_B_in = grad_A_pi = grad_Amult_pi = grad_B_pi = grad_gbar_in = grad_s_in = grad_bias_in = None

            s = s_in
            g = g_in
            if layernorm:
                s = 1
                g = gbar_in
            
            # grad_g_in = inf_linear_backward(A, Amult, B, grad_g_out, g_in, gbar_in, s_in)
            fun_batcher = FunBatcher(cuda_batch_size)
            grad_g_in = fun_batcher.batch_fun(inf_linear_backward, 0, batched_args=[A, Amult.type_as(A), B], unbatched_args=[grad_g_out, g, gbar_in, s])

            if layernorm:
                drho = torch.einsum('br,br->b', grad_g_in, g)
                grad_g_in -= drho[:, None] * g
                grad_g_in /= s_in

            grad_g_in = grad_g_in.to(g_in.device)

            # if ctx.needs_input_grad[0]: # g_in
            #     grad_g_in = dbeta11s + dbeta02s

            if ctx.needs_input_grad[1]: # A
                raise NotImplementedError()

            if ctx.needs_input_grad[2]: # Amult
                raise NotImplementedError()
                
            if ctx.needs_input_grad[3]: # B
                raise NotImplementedError()

            if ctx.needs_input_grad[4]: # A pi
                grad_A_pi = grad_g_out * s

            if ctx.needs_input_grad[5]: # Amult pi
                grad_Amult_pi = torch.ones(s_in.shape[0], dtype=Amult.dtype, device=Amult.device)

            if ctx.needs_input_grad[6]: # B pi
                grad_B_pi = gbar_in

            if ctx.needs_input_grad[7]: # gbar_in
                raise NotImplementedError()

            if ctx.needs_input_grad[8]: # s_in
                raise NotImplementedError()

            if bias is not None and ctx.needs_input_grad[9]: # bias
                grad_bias_in = grad_g_out.sum(dim=0)

            return grad_g_in, grad_A_in, grad_Amult_in, grad_B_in, grad_A_pi, grad_Amult_pi, grad_B_pi, grad_gbar_in, grad_s_in, grad_bias_in

    if cuda_batch_size is not None:
        if return_hidden:
            raise NotImplementedError("Batched operations not implemented for MAML.")
        return BatchedInfPiLinearReLUFunction
    else:
        return InfPiLinearReLUFunction
