import torch

class DynArr():
  def __init__(self, d=None, resizemult=2, initsize=0, initbuffersize=10, device='cpu', **kw):
    '''
    DynArr allows for the "expanding" arrays in the Pi-Net.

    Each training step uses gradient *concatenation* instead of accumulation, 
    meaning all matrices get larger.
    
    DynArr is a simple class to allow this to happen.

    Note that torch is really not a fan of this whole process, hence the custom backwards and
    '''
    self.d = d
    self.device = device
    self.resizemult = resizemult
    if d is not None:
      self.arr = torch.zeros([initbuffersize, d], device=device)
    else:
      self.arr = torch.zeros([initbuffersize], device=device)
    self.size = initsize
  def isempty(self):
    return self.size == 0
  def _cat(self, arr):
    size = arr.shape[0]
    if self.size + size > len(self.arr):
      newsize = int(self.resizemult * self.size + size)
      if self.d is not None:
        assert arr.shape[1] == self.d
        self.arr.resize_(newsize, self.d)
      else:
        assert len(arr.shape) == 1
        self.arr.resize_(newsize)
    self.arr[self.size:self.size+size] = arr
    self.size += size
  def cat(self, *arrs):
    for arr in arrs:
      self._cat(arr)
  def checkpoint(self):
    self._checkpoint = self.size
  def restore(self):
    self.size = self._checkpoint
  @property
  def a(self):
    return self.arr[:self.size]
  def cuda(self):
    self.arr = self.arr.cuda()
    return self
  def cpu(self):
    self.arr = self.arr.cpu()
    return self
  def half(self):
    self.arr = self.arr.half()
    return self
  def float(self):
    self.arr = self.arr.float()
    return self

class CycArr():
  def __init__(self, d=None, maxsize=10000, initsize=0, device='cpu', **kw):
    '''
    Used for some testing (deprecated). 

    Instead of appending to end, write cyclically.
    '''
    assert initsize <= maxsize
    self.size = initsize
    if initsize == maxsize:
      self.end = 0
    else:
      self.end = None
    self.d = d
    self.maxsize = maxsize
    self.device = device
    if d is not None:
      self.arr = torch.zeros([maxsize, d], device=device)
    else:
      self.arr = torch.zeros([maxsize], device=device)

  @property
  def a(self):
    if self.size == self.maxsize:
      return self.arr
    return self.arr[:self.size]
  
  def cuda(self):
    self.arr = self.arr.cuda()
    return self
    
  def half(self):
    self.arr = self.arr.half()
    return self

  def isempty(self):
    return self.size == 0

  def _cat(self, arr):
    size = arr.shape[0]
    if self.size == self.maxsize:
      # cyclic writing
      if self.end + size < self.maxsize:
        self.arr[self.end:self.end+size] = arr
        self.end += size
      else:
        p1size = self.maxsize - self.end
        p2size = size - p1size
        self.arr[self.end:] = arr[:p1size]
        self.arr[:p2size] = arr[p1size:]
        self.end = p2size
    elif self.size + size >= self.maxsize:
      assert size < self.maxsize
      # writing at the end and spill over to beginning
      p1size = self.maxsize - self.size
      p2size = size - p1size
      self.arr[self.size:] = arr[:p1size]
      self.arr[:p2size] = arr[p1size:]
      self.end = p2size
      self.size = self.maxsize
    else:
      # noncyclic writing
      self.arr[self.size:self.size+size] = arr
      self.size += size

  def cat(self, *arrs):
    for arr in arrs:
      self._cat(arr)

