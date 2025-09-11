# 9x9 Sudoku Training Plan - Attempt 1

## Configuration
```bash
DISABLE_COMPILE=1 OMP_NUM_THREADS=4 python3 pretrain.py \
  arch=hrm_v1_balanced \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  epochs=300 \
  eval_interval=30 \
  global_batch_size=27 \
  lr=2e-5 \
  lr_warmup_steps=2000 \
  weight_decay=0.02 \
  lr_min_ratio=0.1
```

## Training Parameters
- **Model**: HRM v1 Balanced (512 hidden, 8 ACT steps, 4 H/L layers)
- **Dataset**: sudoku-extreme-1k-aug-1000 (1M total puzzles)
- **Epochs**: 300 (11,111 total iterations)
- **Batch Size**: 27 (3×9, natural fit for 9x9)
- **Learning Rate**: 2e-5 with 2000 step warmup
- **Regularization**: weight_decay=0.02 (2x stronger than 6x6)

## Current Progress
- **Started**: Training in progress
- **Speed**: ~3.0 it/s (observed from logs)
- **Estimated Time**: ~1 hour total (11,111 iterations)
- **Checkpoints**: Every 30 epochs (1,111 iterations)

## Complexity Jump: 6x6 → 9x9

| Aspect | 6x6 | 9x9 | Scale Factor |
|--------|-----|-----|--------------|
| Cells | 36 | 81 | 2.25x |
| Vocab Size | 8 | 11 | 1.4x |
| Constraints | 18 | 27 | 1.5x |
| Box Shape | 3×2 | 3×3 | Different |
| Typical Blanks | 14 | 40-60 | 3-4x |
| Solution Space | ~10^9 | ~10^20 | Exponential |

## Model Scaling

Based on successful progression:
- **4x4**: 3.4M params → 100% accuracy
- **6x6**: 20.2M params → 100% accuracy
- **9x9**: ~27M params (balanced config) → Target >80%

## Dataset Analysis

**Current Dataset** (sudoku-extreme-1k-aug-1000):
- 1,000 base puzzles from "extreme" difficulty
- 1,000 augmentations per puzzle
- Total: 1,000,000 training examples
- Should be sufficient based on 6x6 success with 16K

**Data/Parameter Ratio**:
- 6x6: 16K puzzles / 20M params = 0.8K per M
- 9x9: 1M puzzles / 27M params = 37K per M ✅
- Much better ratio than 6x6!

## Expected Challenges

1. **Constraint Complexity**
   - 3×3 boxes more complex than 6x6's 3×2
   - More interdependencies between regions
   - Longer logical chains required

2. **ACT Utilization**
   - 8 steps may be insufficient for hardest puzzles
   - Model might need all steps for complex reasoning
   - Unlike 6x6 which solved in 1 step

3. **Generalization**
   - "Extreme" difficulty puzzles are adversarial
   - May require multiple reasoning strategies
   - Test set likely has very hard edge cases

## Success Metrics

### Minimum Success (>70% test accuracy)
- Demonstrates learning on full-scale Sudoku
- Shows architecture scales to real complexity
- Validates approach for logical reasoning

### Good Success (>85% test accuracy)
- Competitive with specialized Sudoku solvers
- Strong generalization to unseen puzzles
- Efficient ACT usage

### Excellent Success (>95% test accuracy)
- Near-perfect logical reasoning
- Matches 6x6 performance at larger scale
- Ready for production use

## Checkpoints to Monitor

1. **Step 1,111** (Epoch 30): Early learning check
2. **Step 3,333** (Epoch 90): Mid-training performance
3. **Step 5,555** (Epoch 150): Convergence check
4. **Step 7,777** (Epoch 210): Late-stage refinement
5. **Step 11,111** (Epoch 300): Final model

## Analysis Points

### Training Metrics to Watch
- **Loss Convergence**: Should decrease smoothly
- **Train/Test Gap**: Monitor for overfitting
- **ACT Steps**: Average should adapt to puzzle difficulty
- **Q-Learning**: Halt accuracy should improve

### Solving Behavior to Analyze
- Steps needed for different difficulty levels
- Error patterns (systematic vs random)
- Self-correction capability
- Constraint satisfaction order

## Risk Factors

1. **Insufficient Data**: 1M might not cover all patterns
2. **Model Capacity**: 27M params might be too small
3. **Training Time**: 300 epochs might not be enough
4. **Extreme Difficulty**: Dataset might be too hard

## Contingency Plans

If accuracy plateaus below 70%:
1. Generate easier dataset (mixed difficulties)
2. Increase model size (full HRM config)
3. Extend training to 500 epochs
4. Adjust learning rate schedule

## Comparison to Previous Results

| Model | Dataset Size | Parameters | Epochs | Test Accuracy |
|-------|-------------|------------|--------|---------------|
| 4x4 | 10K | 3.4M | 50 | 100% |
| 6x6 v1 | 5K | 10.6M | 100 | 89% |
| 6x6 v2 | 20K | 20.2M | 200 | 100% |
| 9x9 v1 | 1M | ~27M | 300 | **TBD** |

## Next Steps After Training

1. **Visualize** solving patterns on test puzzles
2. **Analyze** error types if accuracy < 100%
3. **Compare** ACT usage between difficulties
4. **Document** insights for scaling further

## Notes

- Training started with balanced config (more conservative than full HRM)
- Using strongest regularization yet (weight_decay=0.02)
- Warmup steps increased to handle larger model
- Batch size 27 chosen for GPU memory and 9x9 alignment

This represents the ultimate test of HRM's logical reasoning capability on standard Sudoku complexity.