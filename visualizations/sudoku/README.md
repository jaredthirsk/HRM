# Sudoku Model Visualizations

This directory contains visualizations of how the HRM model solves Sudoku puzzles of various sizes.

## How to Create Visualizations

### Prerequisites
```bash
pip3 install matplotlib
```

### Generate Image Visualizations

Creates PNG images showing the model's solving process step-by-step:

```bash
# For 4x4 Sudoku
python3 visualize_predictions.py \
  --checkpoint "checkpoints/Sudoku-4x4 ACT-torch/[model_name]/step_[N]" \
  --data-path data/sudoku-4x4 \
  --num-puzzles 3 \
  --save-dir visualizations/sudoku/4x4

# For 6x6 Sudoku  
python3 visualize_predictions.py \
  --checkpoint "checkpoints/Sudoku-6x6 ACT-torch/[model_name]/step_[N]" \
  --data-path data/sudoku-6x6 \
  --num-puzzles 3 \
  --save-dir visualizations/sudoku/6x6

# For 9x9 Sudoku
python3 visualize_predictions.py \
  --checkpoint "checkpoints/Sudoku-9x9 ACT-torch/[model_name]/step_[N]" \
  --data-path data/sudoku-extreme-1k-aug-1000 \
  --num-puzzles 3 \
  --save-dir visualizations/sudoku/9x9
```

### Generate Text Visualizations

Creates console-friendly text output showing solving process:

```bash
# For 4x4 Sudoku
python3 visualize_text.py \
  --checkpoint "checkpoints/Sudoku-4x4 ACT-torch/[model_name]/step_[N]" \
  --data-path data/sudoku-4x4 \
  --num-puzzles 2

# For 6x6 Sudoku
python3 visualize_text.py \
  --checkpoint "checkpoints/Sudoku-6x6 ACT-torch/[model_name]/step_[N]" \
  --data-path data/sudoku-6x6 \
  --num-puzzles 2
```

## Visualization Features

### Image Visualizations (`visualize_predictions.py`)
- **Input Puzzle**: Shows given numbers in black
- **Step-by-step Predictions**: Shows predicted numbers in blue
- **Confidence Levels**: Green background intensity indicates prediction confidence
- **Final Solution**: Shows the correct solution for comparison
- Each puzzle generates a multi-panel PNG showing the complete solving process

### Text Visualizations (`visualize_text.py`)
- **Console Output**: ASCII art representation of puzzles
- **Color Coding**: Blue text for predicted values
- **Accuracy Metrics**: Shows per-step accuracy and filled cells count
- **Compact Format**: Easy to read in terminal or logs

## Directory Structure
```
visualizations/sudoku/
├── README.md          # This file
├── 4x4/              # 4x4 Sudoku visualizations
│   ├── analysis.md   # Analysis of model behavior
│   ├── puzzle_1.png  # Example solved puzzles
│   ├── puzzle_2.png
│   └── puzzle_3.png
├── 6x6/              # 6x6 Sudoku visualizations
│   └── ...
└── 9x9/              # 9x9 Sudoku visualizations
    └── ...
```

## Interpreting Results

### Perfect Solution (4x4 Example)
- Model solves all cells in first step
- Continues for configured ACT steps (redundant computation)
- 100% accuracy achieved

### Partial Solution
- Model may solve some cells per step
- ACT mechanism determines when to stop
- Later steps refine uncertain predictions

### Failed Solution
- Incorrect predictions shown in visualization
- Helps identify where model struggles
- Useful for debugging training issues

## Key Findings

From 4x4 analysis:
- Model achieves 100% accuracy but doesn't halt early during evaluation
- ACT mechanism forces maximum steps in eval mode
- Represents 75% wasted computation that could be optimized

## Tips

1. **Use MPLBACKEND=Agg** for headless environments:
   ```bash
   MPLBACKEND=Agg python3 visualize_predictions.py ...
   ```

2. **Find checkpoints** to visualize:
   ```bash
   find checkpoints -name "step_*" | grep -i "sudoku"
   ```

3. **Adjust number of puzzles** with `--num-puzzles N` (default: 3 for images, 2 for text)

4. **Custom save locations** with `--save-dir path/to/dir`