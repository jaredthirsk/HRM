# 6x6 Sudoku - Attempt 2 Training Plan

## Training Configuration
- **Dataset**: data/sudoku-6x6-large (20,000 train, 4,000 test puzzles)
- **Model**: 448 hidden, 4 layers H/L, 8 ACT steps
- **Training**: 200 epochs, batch_size 32, lr=3e-5, weight_decay=0.01
- **Total iterations**: 100,000

## Improvements from Attempt 1
| Aspect | Attempt 1 | Attempt 2 | Improvement |
|--------|-----------|-----------|-------------|
| Training puzzles | 4,000 | 16,000 | 4x more data |
| Test puzzles | 1,000 | 4,000 | 4x more validation |
| Epochs | 100 | 200 | 2x longer training |
| Model size | 384 hidden, 3 layers | 448 hidden, 4 layers | 30% larger |
| Regularization | None | weight_decay=0.01 | Prevents overfitting |
| Blanks per puzzle | 12 | 14 | Slightly harder |

## Expected Outcomes
- **Target**: >95% test accuracy (vs ~89% in attempt 1)
- **Training time**: ~5 hours
- **Key metric to watch**: Test accuracy should stay close to train accuracy

## Analysis Points
1. **Generalization**: With 4x more data, should see less overfitting
2. **ACT efficiency**: May learn to use fewer than 8 steps
3. **Error patterns**: Check if errors are random or systematic
4. **Convergence**: Loss should decrease smoothly with regularization

## Checkpoints to Analyze
- Step 12,500 (epoch 25)
- Step 25,000 (epoch 50)
- Step 50,000 (epoch 100)
- Step 75,000 (epoch 150)
- Step 100,000 (epoch 200)

## Success Criteria
- Test accuracy > 95%
- Train/test gap < 5%
- Consistent solving across different puzzle patterns
- ACT steps adapt based on puzzle difficulty