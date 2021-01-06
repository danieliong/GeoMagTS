import numpy as np
from collections import deque

class SparseColumnMatrix(np.ndarray):
    
    def __new__(cls, input_array, active_mask=None):
        obj = np.asarray(input_array).view(cls)
        obj.active_mask = active_mask 
        
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.active_mask = getattr(obj, 'active_mask', None) 
    
    def __array_ufunc__(self, ufunc, method, *inputs, where=True, **kwargs):
        
        masks = []
        
        for input in inputs:
            if isinstance(input, SparseColumnMatrix):
                masks.append(input.active_mask)
            else:
                masks.append(None)

        pass 
    
        
    
    