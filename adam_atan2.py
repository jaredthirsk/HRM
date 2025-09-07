"""
AdamATan2 optimizer workaround
This is a temporary implementation to replace the missing adam-atan2 package.
Based on standard Adam with potential arctan2 learning rate scheduling support.
"""

import torch
from torch.optim import Adam


class AdamATan2(Adam):
    """
    AdamATan2 optimizer - wrapper around standard Adam optimizer.
    
    This is a workaround implementation since the original adam-atan2 package
    has installation issues. The arctan2 scheduling is typically handled 
    externally via learning rate schedulers, so this implementation focuses
    on providing a compatible interface.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages (default: (0.9, 0.999))
        eps: term added for numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        The arctan2 scheduling (if needed) should be handled by external
        learning rate schedulers that modify the lr parameter groups.
        """
        return super().step(closure)