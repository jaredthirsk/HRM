# Workarounds and Technical Debt

## adam-atan2 Optimizer Issue

### Problem
The `adam-atan2` package fails to install due to a setuptools compatibility issue:
```
TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
```

### Root Cause
- The package's setup.py is incompatible with newer setuptools versions
- The package repository appears to be missing or private on GitHub
- PyPI version 0.0.3 has metadata generation issues

### Attempted Solutions
1. ❌ Direct pip install: `pip3 install adam-atan2`
2. ❌ Install from GitHub: Repository not found
3. ❌ Install with no-build-isolation flag
4. ❌ Upgrade setuptools and retry

### Proposed Workarounds

#### Option 1: Local Implementation (Recommended)
Create a local implementation of AdamATan2 based on the paper/expected behavior:
```python
# models/optimizers/adam_atan2.py
import torch
import math

class AdamATan2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        # Implementation based on standard Adam with atan2 scheduling
        pass
```

#### Option 2: Modify Training Script
Edit `pretrain.py` to use standard Adam optimizer:
```python
# Replace line 19:
# from adam_atan2 import AdamATan2

# With:
from torch.optim import Adam as AdamATan2  # Temporary alias
```

#### Option 3: Manual Package Fix
1. Download the source code
2. Fix the setup.py file
3. Install locally

### Impact on Training
- AdamATan2 provides arctan2 learning rate scheduling
- May affect convergence speed and final performance
- Standard Adam should still work but might need hyperparameter tuning

## FlashAttention Installation

### Status
✅ **RESOLVED** - Implemented fallback attention mechanism

### Problem
FlashAttention packages were not installed, causing import errors during model initialization.

### Solution Implemented
Modified `models/layers.py` to gracefully handle missing FlashAttention:
1. Added import try/catch blocks with fallback flags
2. Implemented `standard_attention()` function as fallback
3. Modified `Attention.forward()` to use standard attention when FlashAttention unavailable

### Impact
- Training will work without FlashAttention
- May use more memory than FlashAttention (but should be fine for single GPU)
- Performance may be slightly slower but functionally equivalent

### Optional Future Installation
```bash
# For Ampere or earlier GPUs
pip3 install flash-attn

# For Hopper GPUs  
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

## Other Potential Issues

### 1. CUDA Version Mismatch
- Current: PyTorch with CUDA 11.8
- May need: CUDA 12.6 for optimal performance
- Workaround: Continue with current setup, monitor for issues

### 2. Weights & Biases Login
- Need to run `wandb login` before training
- Can run offline with `wandb offline` if needed

### 3. OMP_NUM_THREADS
- Must set to prevent CPU oversubscription
- Current setting: `OMP_NUM_THREADS=8`

## Tracking
- [ ] Resolve adam-atan2 issue
- [ ] Install FlashAttention (if GPU supports)
- [ ] Test W&B integration
- [ ] Verify CUDA compatibility
- [ ] Document any additional workarounds needed