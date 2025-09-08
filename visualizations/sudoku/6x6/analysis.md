# 6x6 Sudoku Model Analysis

## Training Metrics Summary
- **Model Size**: 10.6M parameters (3x larger than 4x4, 2.5x smaller than 9x9)
- **Final Accuracy**: 100% on training metrics
- **Loss**: 0.016 (very low, indicating good convergence)
- **ACT Steps**: Average 4.75 steps (configured max: 6)
- **Training Time**: ~22 minutes for 100 epochs
- **Q-Learning**: Successfully learned halting (100% q_halt_accuracy)

## Evaluation Results

### Surprising Discovery: Training vs Test Gap
Despite 100% training accuracy, the model shows **88.9% accuracy on test puzzles**. This reveals:

1. **Overfitting on Training Data**: The model memorized training puzzles rather than learning general solving rules
2. **Consistent Error Pattern**: Makes the same 1-2 errors repeatedly across steps
3. **No Self-Correction**: Once an error is made in step 1, it persists through all 6 steps

### Detailed Behavior Analysis

From the text visualization:
- **Step 1**: Model attempts to solve all blanks, gets 9/12 correct (75%)
- **Steps 2-6**: Predictions stabilize at 8/12 correct, with one cell changing
- **Error Types**: 
  - Row 1, Col 1: Predicts 4, correct is 3
  - Row 2, Col 1: Predicts 3→4 (unstable), correct is 4
  - Row 2, Col 4: Predicts 4→3 (unstable), correct is 3

### ACT Mechanism Observations
- **Uses all 6 steps**: Unlike 4x4 which solved in 1 step
- **No improvement across steps**: Additional computation doesn't fix errors
- **Evaluation forces max steps**: Same issue as 4x4 model

## Comparison Across Puzzle Sizes

| Metric | 4x4 | 6x6 | 9x9 |
|--------|-----|-----|-----|
| Parameters | 3.4M | 10.6M | 27M |
| Cells | 16 | 36 | 81 |
| Train Accuracy | 100% | 100% | TBD |
| Test Accuracy | 100% | ~89% | TBD |
| ACT Steps Used | 4/4 | 6/6 | TBD |
| Solving Pattern | Instant | Partial | TBD |

## Key Insights

### Why 6x6 is Harder Than Expected
1. **Complexity Jump**: 36 cells vs 16 is a 2.25x increase
2. **Constraint Interactions**: 6x6 uses 2x3 rectangular boxes, different from square boxes
3. **Limited Training Data**: Only 5000 puzzles may be insufficient

### Model Limitations Revealed
1. **Memorization vs Reasoning**: High train accuracy with lower test accuracy suggests memorization
2. **No Error Recovery**: Model can't correct mistakes in later steps
3. **ACT Not Learning Properly**: Should halt earlier if predictions don't change

## Recommendations

### Immediate Fixes
1. **Increase Training Data**: Generate 20,000+ puzzles
2. **Add Data Augmentation**: Rotate, reflect, and relabel puzzles
3. **Longer Training**: 200+ epochs with learning rate scheduling
4. **Regularization**: Add dropout or weight decay to prevent overfitting

### Architecture Improvements
1. **Increase Model Capacity**: Try hidden_size=448 or add more layers
2. **Better Positional Encoding**: 6x6's rectangular boxes may need special handling
3. **Multi-Scale Reasoning**: Add connections between H and L modules at multiple points

### Training Strategy
```bash
# Recommended training command for better 6x6 performance
DISABLE_COMPILE=1 OMP_NUM_THREADS=4 python3 pretrain.py \
  arch.hidden_size=448 \
  arch.H_layers=4 \
  arch.L_layers=4 \
  arch.halt_max_steps=8 \
  data_path=data/sudoku-6x6-large \
  epochs=200 \
  eval_interval=25 \
  global_batch_size=32 \
  lr=3e-5 \
  weight_decay=0.01
```

## Conclusion

The 6x6 results reveal that HRM can struggle with intermediate complexity puzzles when training data is limited. While the architecture works perfectly on simple 4x4 puzzles, the jump to 6x6 exposes:
- Tendency to memorize rather than reason
- Inability to self-correct errors
- Need for more sophisticated training strategies

This is valuable insight for scaling to full 9x9 Sudoku, suggesting we need:
- Much larger datasets (100K+ puzzles)
- Progressive curriculum (train on easier puzzles first)
- Better regularization techniques