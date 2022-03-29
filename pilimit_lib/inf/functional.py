import torch
import torch.functional
from torch import nn
from inf.math import *

class FunBatcher():
    def __init__(self, cuda_batch_size):
        self.cuda_batch_size = cuda_batch_size

    def batch_fun(self, fun, batch_axis, **kwargs):
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
        self.cuda_batch_size = cuda_batch_size
        self.device_count = torch.cuda.device_count()
        self.devices = list(range(0, self.device_count))

    def batch_fun(self, fun, batch_axis, **kwargs):
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

def inf_linear_forward_return_q(A, Amult, B, g, gbar, s, bias, bias_alpha=None):
    g_out = None

    if s is None:
        s = g.norm(dim=1, keepdim=True)

    if gbar is None:
        gbar = g / s

    q = gbar @ B.T

    # test = torch.einsum("bm, mr -> bmr", F00ReLUsqrt(q, 1, s)* Amult.type_as(q),  A)

    # normed = test[0, :, :].norm(dim=1)
    # print((normed < .1 ).sum()  / q.shape[1], (normed < .01 ).sum() / q.shape[1], (normed < .000001 ).sum() / q.shape[1])
    # idx = normed > .000001
    # g_out = test[:, idx, :].sum(axis=1)
    # g_out = test.sum(axis=1)


    g_out = (F00ReLUsqrt(q, 1, s) * Amult.type_as(q)) @ A

    if bias is not None:
        g_out += (bias * bias_alpha).type_as(q) if bias_alpha else bias.type_as(q)

    return g_out, q

def inf_linear_backward(A, Amult, B, grad, g, gbar, s, q=None):
    dgamma = grad @ (A.T * Amult.type_as(grad) )
    
    if q is None:
        q = gbar @ B.T

    dalpha11 = F11ReLUsqrt(q, 1, s)
    dalpha02 = F02ReLUsqrt(q, 1, s)

    dbeta11s = (dgamma * dalpha11) @ B
    dbeta02s = contract('bm,bm,br->br', dalpha02, dgamma, g)

    return dbeta11s + dbeta02s



def InfPiLinearReLUFunctionBuilder(layernorm=False, cuda_batch_size=None):
    class InfPiLinearReLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, g_in, A, Amult, B, A_pi, Amult_pi, B_pi, gbar_in=None, s_in=None, bias=None):
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
            
            g_out, q_in = inf_linear_forward_return_q(A, Amult, B, g_in, gbar_in, s, bias)


            ctx.save_for_backward(g_in, A, Amult, B, A_pi, Amult_pi, B_pi, gbar_in, s_in, bias, q_in)

            # s_out = g_out.norm(dim=1, keepdim=True)
            # gbar_out = g_out / s_out
            # ctx.mark_non_differentiable(s_out, gbar_out)

            return g_out

        @staticmethod
        def backward(ctx, grad_g_out):
            g_in, A, Amult, B, A_pi, Amult_pi, B_pi, gbar_in, s_in, bias, q_in = ctx.saved_tensors
            grad_g_in = grad_A_in = grad_Amult_in = grad_B_in = grad_A_pi = grad_Amult_pi = grad_B_pi = grad_gbar_in = grad_s_in = grad_bias_in = None

            s = s_in
            g = g_in
            if layernorm:
                s = 1
                g = gbar_in
            
            grad_g_in = inf_linear_backward(A, Amult, B, grad_g_out, g, gbar_in, s, q=q_in)
            # print(grad_g_in)

            if layernorm:
                drho = torch.einsum('br,br->b', grad_g_in, g)
                grad_g_in -= drho[:, None] * g
                grad_g_in /= s_in
            
            # if ctx.needs_input_grad[0]: # g_in
            #     grad_g_in = dbeta11s + dbeta02s

            if ctx.needs_input_grad[1]: # A
                raise NotImplementedError()

            if ctx.needs_input_grad[2]: # Amult
                raise NotImplementedError()
                
            if ctx.needs_input_grad[3]: # B
                raise NotImplementedError()

            if ctx.needs_input_grad[4]: # A pi
                # grad_g_out /= 3
                # print(grad_g_out, grad_g_out.dtype)
                # print(s, s.dtype)
                grad_A_pi = grad_g_out * s

                # print("test", grad_A_pi* 3)
                # 0/0

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
                # print(grad_bias_in.dtype, bias.dtype)
                # grad_bias_in = grad_g_in.sum(dim=0)

            return grad_g_in, grad_A_in, grad_Amult_in, grad_B_in, grad_A_pi, grad_Amult_pi, grad_B_pi, grad_gbar_in, grad_s_in, grad_bias_in

    class BatchedInfPiLinearReLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, g_in, A, Amult, B, A_pi, Amult_pi, B_pi, gbar_in=None, s_in=None, bias=None):
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
        return BatchedInfPiLinearReLUFunction
    else:
        return InfPiLinearReLUFunction
