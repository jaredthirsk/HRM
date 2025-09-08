# Training Recommendations for HRM

## Current Issue
The model achieved 0% accuracy after training with overly aggressive optimizations:
- `halt_max_steps` reduced from 16 to 2 (too few computation steps)
- `H_cycles` and `L_cycles` reduced from 2 to 1 (insufficient reasoning iterations)

## Root Cause
Sudoku solving requires multiple reasoning steps. By limiting the model to only 2 ACT steps and 1 cycle, we prevented it from having enough computation to learn the puzzle-solving patterns.

## Recommended Approaches

### Option 1: Balanced Configuration (Recommended)
Use the new `hrm_v1_balanced.yaml` config with moderate settings:
```bash
OMP_NUM_THREADS=8 DISABLE_COMPILE=1 python pretrain.py \
  arch=hrm_v1_balanced \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  epochs=100 \
  eval_interval=20 \
  global_batch_size=256 \
  lr=1e-3
```

**Advantages:**
- 8 ACT steps provides sufficient computation for puzzle solving
- 2x faster than original (8 vs 16 steps)
- Maintains learning capability

### Option 2: Easier Dataset
Build and train on easier puzzles first:
```bash
# Build easier dataset
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-easy-1k \
  --subsample-size 1000 \
  --num-aug 100 \
  --min-difficulty 0

# Train on easier puzzles
OMP_NUM_THREADS=8 DISABLE_COMPILE=1 python pretrain.py \
  arch=hrm_v1_balanced \
  data_path=data/sudoku-easy-1k \
  epochs=50 \
  eval_interval=10
```

### Option 3: Progressive Training
Start with few ACT steps and gradually increase:
```bash
# Stage 1: Quick learning (4 steps)
OMP_NUM_THREADS=8 DISABLE_COMPILE=1 python pretrain.py \
  arch.halt_max_steps=4 \
  epochs=20

# Stage 2: Increase capacity (8 steps)
OMP_NUM_THREADS=8 DISABLE_COMPILE=1 python pretrain.py \
  arch.halt_max_steps=8 \
  epochs=40 \
  checkpoint_path=checkpoints/stage1

# Stage 3: Full capacity (16 steps)
OMP_NUM_THREADS=8 DISABLE_COMPILE=1 python pretrain.py \
  arch.halt_max_steps=16 \
  epochs=40 \
  checkpoint_path=checkpoints/stage2
```

## Key Parameters to Monitor

1. **halt_max_steps**: Controls maximum computation steps
   - 2-4: Too few for Sudoku
   - 8-12: Good balance
   - 16: Original, most capable but slow

2. **H_cycles/L_cycles**: Reasoning iterations
   - Must be at least 2 for complex puzzles
   - Higher values = better reasoning but slower

3. **Learning rate**: 
   - Start with 1e-3
   - Reduce if loss explodes
   - Use warmup for stability

4. **Batch size**:
   - Larger batches (256-512) for stable training
   - Smaller batches train faster but may be unstable

## Expected Training Times

With optimized attention (F.scaled_dot_product_attention):
- halt_max_steps=4: ~30 min for 100 epochs
- halt_max_steps=8: ~1 hour for 100 epochs  
- halt_max_steps=16: ~2 hours for 100 epochs

## Success Indicators

Watch for these metrics during training:
- Loss decreasing consistently
- Q-values converging (indicates ACT learning)
- Validation accuracy > 0% after 20 epochs
- Puzzle embeddings showing variance (not stuck at initialization)

## Debugging Tips

If training fails to improve:
1. Check if puzzle embeddings are updating (should have non-zero gradients)
2. Verify ACT is using variable steps (not always max steps)
3. Ensure loss is not NaN or exploding
4. Try reducing learning rate or increasing warmup steps

## Next Steps

1. Start with Option 1 (balanced config) on existing dataset
2. If no improvement after 20 epochs, try Option 2 (easier dataset)
3. Consider Option 3 for production training (progressive difficulty)