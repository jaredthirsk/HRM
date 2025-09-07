# ✅ Working Training Command

## Success! 🎉

Training is now working after fixing the tensor view stride issue in the attention mechanism.

## Working Command

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

## Key Changes Made

### 1. Fixed Tensor View Issue
- Added `.contiguous()` before `.view()` in standard attention
- This resolves stride mismatch errors in tensor reshaping

### 2. Disabled torch.compile
- Set `DISABLE_COMPILE=1` to avoid Dynamo compilation errors
- Model runs in eager mode, slightly slower but stable

### 3. Reduced Batch Size
- Changed from 384 to 128 to be more conservative on memory
- Still sufficient for effective training

## Training Progress

- **Status**: ✅ Running successfully
- **Speed**: ~1.5s per batch  
- **W&B Tracking**: Active
- **Expected Time**: ~6-8 hours for full convergence (20k epochs)

## Monitoring

- Training logs show progress in terminal
- W&B dashboard: https://wandb.ai/jaredthirsk-org/Sudoku-extreme-1k-aug-1000%20ACT-torch
- Look for `eval/exact_accuracy` metric reaching >90%

## For Faster Results (Test Run)

```bash
DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python3 pretrain.py \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  epochs=1000 \
  eval_interval=100 \
  global_batch_size=128 \
  lr=7e-5 \
  puzzle_emb_lr=7e-5 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0
```

This will complete in ~1-2 hours and show if the model is learning effectively.

## Issues Resolved

- ✅ adam-atan2 import (local implementation)
- ✅ FlashAttention import (standard attention fallback)  
- ✅ Tensor view stride errors (contiguous memory layout)
- ✅ torch.compile errors (disabled compilation)

The HRM model is now successfully training on Sudoku puzzles! 🧩