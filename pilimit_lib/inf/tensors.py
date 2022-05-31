import torch

class FinPiParameter(torch.nn.Parameter):
    def __new__(cls, data, requires_grad=True):
        '''
        A Finite Pi-Parameter.

        This class is almost the same as torch.nn.Parameter,
        but has a function to store omega, gcovinv, and pi_proj.
        '''
        return super().__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad=True):
        super(FinPiParameter, self).__init__()

        self.omega = None
        self.gcovinv = None
        self.pi_proj = None

    def store_gproj_vars(self, omega, gcovinv, pi_proj):
        '''
        A function to store omega, gcovinv, and pi_proj in this variable.
        These variables are used for gradient projection.
        Storing in the parameter allows for easy tracking of these matrices 
        as well as easy saving/loading.

        However, this is kind of a gross solution. It would probably be better to 
        accomplish this in another way (stored in the layers?).

        This will get refactored.
        '''
        self.omega = omega
        self.gcovinv = gcovinv
        self.pi_proj = pi_proj

class InfPiParameter(torch.nn.Parameter):
    def __new__(cls, dyn_data, apply_lr=False, lr_mult=1, requires_grad=True, optim_mode="project"):
        '''
        Create a custom __new__ function which used to be necessary for DynamicTensor
        but isn't now.

        This will get refactored.
        '''
        # return super().__new__(cls, dyn_data.tensor() if isinstance(dyn_data, DynamicTensor) else dyn_data, requires_grad)
        return super().__new__(cls, dyn_data, requires_grad)

    def __init__(
            self, 
            dyn_data, 
            apply_lr=False, 
            lr_mult=1, 
            requires_grad=True, 
            optim_mode="project"):
        '''
        Define an Inf-width Pi Parameter (A, B, or Amult).

        These parameters allow for gradient concatenation instead of accumulation.
        To allow for this, set_pi_size can be used to dynamically size of the pi update
        (which depends on batch size).

        This parameter also allows for gradient accumulation (multiple steps without lr),
        and stores pi_grad_norm for gradient clipping.

        Inputs:
            dyn_data: the expanding tensor to concatenate gradient to
            apply_lr: whether to acually use lr on this param when updatng (only for Amult)
            lr_mult: an lr multiplier for just this parameter
            requires_grad: whether to require grad (required by Parameter)
            optim_mode: which optimization mode to use (only projection for now)
        '''


        self.dyn_data = dyn_data
        self.apply_lr = apply_lr
        self.lr_mult = lr_mult
        self.requires_grad = requires_grad
        super(InfPiParameter, self).__init__()

        if optim_mode not in ["project"]:
            raise ValueError("optim_mode must be 'project'")
            
        self.project()
        
        # necessary for gradient clipping - A and B must be used together to find this value
        # but ultimately it applies to only A
        # this will get refactored
        self.pi_grad_norm = None 
        self.accum_grad = None

    def project(self):
        '''
        Use projection for training.
        This initializes a dummy self.pi.
        '''
        self.optim_mode = "project"

        self.pi = torch.zeros(0, 0, requires_grad=True, device=self.device, dtype=self.dtype)
        
    def cat_grad(self, x, alpha=1):
        '''
        Concate an incoming gradient update x to the inf-width pi-parameter,
        with learning rate multiplier and alpha if appilcable (only Amult).
        '''

        self.data = torch.cat((self.data, alpha*x*self.lr_mult if self.apply_lr else x))
        
    @torch.no_grad()
    def stage_grad(self):
        '''
        Stage gradient updates in self.accum_grad, in case one 
        wishes to use a small batch size and only apply lr every few steps.
        '''
        grad = self.pi.grad.clone()

        if self.accum_grad is None:
            self.accum_grad = grad
        else:
            self.accum_grad = torch.cat((self.accum_grad, grad))
        
        self.pi.grad[:] = 0
        
    @torch.no_grad()
    def unstage_grad(self):
        '''
        Reload the gradient into self.pi.grad.
        '''
        self.set_pi_size(self.accum_grad.shape[0])

        self.pi.grad = self.accum_grad.clone()
        self.accum_grad = None

    @torch.no_grad()
    def set_pi_size(self, batch_shape, dim=0):
        '''
        Reshape self.pi according to batch_shape so that 
        it may store the projected inf-width gradient update.
        '''

        if batch_shape == self.pi.shape[dim]:
            if self.pi.grad is not None: self.pi.grad[:] = 0
            return

        shape = list(self.shape) # retain other dims of var
        shape[dim] = batch_shape

        self.pi = torch.zeros(shape, requires_grad=True, device=self.device, dtype=self.dtype)
        self.pi.retain_grad()

    def checkpoint(self):
        self.checkpoint_size = self.data.shape[0]

    def restore(self):
        self.data = self.data[:self.checkpoint_size]

''' 

# Unnecessary code - may come back to this concept if it's useful

HANDLED_FUNCTIONS = {}
class DynamicTensor(object):
    def __init__(self, r, resizemult=2, initsize=0, initbuffersize=1000, device="cpu", dtype=None):
        self._r = r
        self._resizemult = resizemult
        self._size = initsize
        self.device = device
        self.dtype = dtype
        
        if r is None:
            self._arr = torch.zeros([initbuffersize], device=device, requires_grad=True, dtype=dtype)
        else:
            self._arr = torch.zeros([initbuffersize, r], device=device, requires_grad=True, dtype=dtype)
        # self._arr.retain_grad()

        # self.tensor_arr = self._arr[:self._size]
        # self.tensor_arr.retain_grad()

    def __repr__(self):
        return "DynamicTensor(r={}, m={})".format(self._r, self._size)

    def tensor(self):
        # return self.tensor_arr
        return self._arr[:self._size]
    
    def cat(self, *arrs):
        for arr in arrs:
            self._cat(arr)
        
    def _cat(self, arr):
        0/0
        assert isinstance(arr, torch.Tensor)

        self._arr.requires_grad = False
        size = arr.shape[0]
        
        if self._size + size > len(self._arr):
            if self._r is not None:
                assert arr.shape[1] == self._r
                self._arr.resize_(int(self._resizemult * self._size + size), self._r)
            else:
                assert len(arr.shape) == 1
                self._arr.resize_(int(self._resizemult * self._size + size))

        self._arr[self._size:self._size+size] = arr
        self._size += size

        self._arr.requires_grad = True

    @classmethod
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
                issubclass(t, (torch.Tensor, DynamicTensor))
                for t in types
            ):
            args = [a.tensor() if hasattr(a, 'tensor') else a for a in args]
            return func(*args, **kwargs)
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

'''