"""
Proper Python implementation of AdamATan2 optimizer
Based on the paper "Scaling Exponents Across Parameterizations and Optimizers" (arXiv:2407.05872)
by Google DeepMind team.

This is a pure Python implementation that matches the algorithmic behavior
of the CUDA-optimized version without requiring compilation.
"""

import torch
from torch.optim import Optimizer
import math


class AdamATan2(Optimizer):
    """
    AdamATan2 optimizer - a numerically stable, scale-invariant version of Adam
    that eliminates the epsilon hyperparameter by using atan2.
    
    The key innovation is replacing:
        param -= lr * exp_avg / (sqrt(exp_avg_sq) + eps)
    with:
        param -= lr * atan2(exp_avg, sqrt(exp_avg_sq))
    
    This eliminates the need for epsilon while maintaining numerical stability.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages (default: (0.9, 0.999))
        weight_decay: weight decay coefficient (default: 0.01)
    
    Reference:
        Scaling Exponents Across Parameterizations and Optimizers
        https://arxiv.org/abs/2407.05872
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(AdamATan2, self).__init__(params, defaults)
        
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamATan2 does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step'].item()
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Weight decay (decoupled, like AdamW)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                # Compute the denominator for atan2
                denom = exp_avg_sq.sqrt() / bias_correction2_sqrt
                
                # Compute step size with bias correction
                step_size = lr / bias_correction1
                
                # AdamATan2 update using atan2 instead of division
                # This is the key innovation: atan2(exp_avg, denom) instead of exp_avg / (denom + eps)
                p.data.add_(torch.atan2(exp_avg, denom), alpha=-step_size)
                
        return loss


# Also create an alias for compatibility
class AdamAtan2(AdamATan2):
    """Alias for AdamATan2 with different capitalization"""
    pass