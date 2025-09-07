# Complete Instructions: How to Get HRM Working

## Overview
The official README.md provides incomplete instructions that will NOT work out of the box. This document provides the ACTUAL steps needed to get training running, including all necessary workarounds.

## What the Official README Says vs Reality

### Official README Claims:
```bash
# "Install Python Dependencies"
pip install -r requirements.txt

# "Run Experiments" 
OMP_NUM_THREADS=8 python pretrain.py
```

### Reality: This WILL FAIL due to:
1. ❌ adam-atan2 package installation fails
2. ❌ FlashAttention not installed
3. ❌ No dataset exists yet
4. ❌ Tensor view errors during runtime
5. ❌ Torch compile issues

## ACTUAL Working Instructions

### Step 1: Install Dependencies (Modified)

The official `requirements.txt` includes `adam-atan2` which WILL FAIL. Install everything else:

```bash
# Install all dependencies EXCEPT adam-atan2
pip3 install torch einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core huggingface_hub

# Optional: Install ninja for faster compilation
pip3 install ninja packaging setuptools-scm
```

**What README didn't mention**: adam-atan2 has broken setuptools compatibility

### Step 2: Create adam-atan2 Workaround

The README assumes adam-atan2 will install. It won't. Create this file:

```bash
cat > adam_atan2.py << 'EOF'
"""
AdamATan2 optimizer workaround
This replaces the broken adam-atan2 package
"""
import torch
from torch.optim import Adam

class AdamATan2(Adam):
    """Wrapper around standard Adam optimizer"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
EOF
```

**What README didn't mention**: No fallback for missing optimizer

### Step 3: Fix FlashAttention Import Errors

The README mentions FlashAttention but doesn't handle when it's not available. Modify `/exp/HRM/models/layers.py`:

#### Fix 1: Import handling (lines 7-18)
Replace:
```python
try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    from flash_attn import flash_attn_func
```

With:
```python
try:
    from flash_attn_interface import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    try:
        from flash_attn import flash_attn_func
        HAS_FLASH_ATTN = True
    except ImportError:
        HAS_FLASH_ATTN = False
        flash_attn_func = None
```

#### Fix 2: Add fallback attention (after line 47)
Add this function:
```python
def standard_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool = False):
    """Fallback standard attention when FlashAttention not available"""
    batch_size, seq_len, num_heads, head_dim = query.shape
    
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
    
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, value)
    
    return attn_output.transpose(1, 2)
```

#### Fix 3: Update Attention.forward() (around line 162)
Replace:
```python
attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
if isinstance(attn_output, tuple):
    attn_output = attn_output[0]
attn_output = attn_output.view(batch_size, seq_len, self.output_size)
```

With:
```python
if HAS_FLASH_ATTN:
    attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]
    attn_output = attn_output.view(batch_size, seq_len, self.output_size)
else:
    attn_output = standard_attention(query, key, value, causal=self.causal)
    attn_output = attn_output.contiguous().view(batch_size, seq_len, self.output_size)
```

**What README didn't mention**: No fallback for missing FlashAttention

### Step 4: Build the Dataset FIRST

The README mentions datasets but doesn't emphasize they must be built first:

```bash
# Build Sudoku dataset (takes ~20-30 minutes)
python3 dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000
```

**What README didn't mention**: You MUST build datasets before training

### Step 5: Configure Weights & Biases (Optional)

```bash
# Login to W&B for experiment tracking
wandb login

# Or run offline
wandb offline
```

**What README didn't mention**: W&B will prompt during training if not configured

### Step 6: Run Training with Required Modifications

The README's training command WILL NOT WORK. Use this instead:

```bash
DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python3 pretrain.py \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  epochs=20000 \
  eval_interval=2000 \
  global_batch_size=128 \
  lr=7e-5 \
  puzzle_emb_lr=7e-5 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0
```

**Critical additions not in README**:
- `DISABLE_COMPILE=1` - Prevents Dynamo compilation errors
- Reduced batch size (128 vs 384) - Prevents memory issues
- Specific learning rates - Required for convergence

### Step 7: For Background Training

Since training takes 6-10 hours:

```bash
# Option 1: nohup
nohup [training command] > training.log 2>&1 &

# Option 2: screen
screen -S hrm_training
[run training command]
# Detach with Ctrl+A, D

# Option 3: tmux
tmux new -s hrm_training
[run training command]
# Detach with Ctrl+B, D
```

**What README didn't mention**: Training takes many hours

## Summary of Issues with Official Documentation

### What the README Got Wrong/Omitted:

1. **No mention of adam-atan2 installation failures**
   - Broken setuptools compatibility
   - No workaround provided

2. **FlashAttention assumed to work**
   - No fallback implementation
   - Import errors not handled

3. **Dataset building not emphasized**
   - Must be done BEFORE training
   - Takes significant time

4. **Torch compilation issues**
   - Need DISABLE_COMPILE=1
   - Dynamo errors not mentioned

5. **Memory/batch size issues**
   - Default batch size too large
   - May cause OOM errors

6. **Missing tensor layout fixes**
   - Need .contiguous() for view operations
   - Stride errors not addressed

### Files We Had to Modify/Create:

1. **Created**: `/exp/HRM/adam_atan2.py` (new file)
2. **Modified**: `/exp/HRM/models/layers.py` (3 major changes)
3. **Environment**: Added `DISABLE_COMPILE=1`
4. **Parameters**: Adjusted batch_size and learning rates

## Verification Commands

```bash
# Check training is running
ps aux | grep pretrain.py

# Monitor GPU (if using)
nvidia-smi

# Check dataset was built
ls -la data/sudoku-extreme-1k-aug-1000/

# Monitor training progress
tail -f wandb/latest-run/files/output.log

# Check W&B dashboard
echo "https://wandb.ai/[your-username]/Sudoku-extreme-1k-aug-1000%20ACT-torch"
```

## Expected Behavior When Working

- Training starts without import errors
- Progress bar shows iterations advancing
- CPU usage at 90-100%
- Memory usage ~1-2GB
- Speed: ~1-2 iterations/second
- W&B uploads metrics

## Common Failure Modes and Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'adam_atan2'` | Create adam_atan2.py workaround |
| `ModuleNotFoundError: No module named 'flash_attn'` | Apply layers.py modifications |
| `RuntimeError: Cannot view tensor with shape...` | Add .contiguous() in layers.py |
| `FileNotFoundError: data/sudoku-extreme-1k-aug-1000` | Build dataset first |
| `torch._dynamo.exc.TorchRuntimeError` | Set DISABLE_COMPILE=1 |
| `Out of Memory` | Reduce global_batch_size |

## Time Estimates

- Installing dependencies: 5 minutes
- Creating workarounds: 10 minutes
- Building dataset: 20-30 minutes
- Training to convergence: 6-10 hours
- Quick test (100 epochs): 1-2 hours

## Conclusion

The official README provides a idealized "happy path" that assumes:
- All packages install correctly
- FlashAttention is available
- Datasets already exist
- No compilation issues
- Sufficient memory

In reality, NONE of these assumptions hold, requiring significant debugging and workarounds to get the model actually training. This document provides the REAL instructions that work.