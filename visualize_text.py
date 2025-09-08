#!/usr/bin/env python3
"""Simple text visualization of model solving Sudoku puzzles."""

import os
import torch
import numpy as np
from pathlib import Path
from visualize_predictions import load_checkpoint, load_test_puzzles, puzzle_to_grid


def text_grid(grid, highlight_predicted=None):
    """Create text representation of Sudoku grid."""
    size = len(grid)
    box_size = 2 if size == 4 else 3
    
    lines = []
    for i in range(size):
        if i > 0 and i % box_size == 0:
            lines.append("-" * (size * 2 + box_size - 1))
        
        row = []
        for j in range(size):
            if j > 0 and j % box_size == 0:
                row.append("|")
            
            val = grid[i, j]
            if val == 0:
                row.append(".")
            else:
                # Highlight predicted values
                if highlight_predicted is not None and highlight_predicted[i, j]:
                    row.append(f"\033[94m{int(val)}\033[0m")  # Blue
                else:
                    row.append(str(int(val)))
        
        lines.append(" ".join(row))
    
    return "\n".join(lines)


def solve_and_display(model, puzzle, solution, is_4x4=True):
    """Show step-by-step solving process."""
    size = 4 if is_4x4 else 9
    
    print("\n" + "="*50)
    print("ORIGINAL PUZZLE:")
    original_grid = puzzle_to_grid(puzzle, is_4x4)
    print(text_grid(original_grid))
    
    with torch.no_grad():
        batch = {
            "inputs": torch.tensor(puzzle).unsqueeze(0),
            "labels": torch.tensor(solution).unsqueeze(0),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.int32)
        }
        
        carry = model.initial_carry(batch)
        
        step = 0
        while True:
            carry, _, metrics, preds, finished = model(carry=carry, batch=batch, return_keys=["logits"])
            step += 1
            
            if "logits" in preds:
                predictions = torch.argmax(preds["logits"][0], dim=-1).cpu().float().numpy()
                pred_grid = puzzle_to_grid(predictions, is_4x4)
                
                # Create highlight mask for predicted values
                highlight = (original_grid == 0) & (pred_grid > 0)
                
                print(f"\nSTEP {step}:")
                print(text_grid(pred_grid, highlight))
                
                # Check accuracy
                accuracy = np.mean(predictions == solution)
                correct_fills = np.sum((original_grid == 0) & (pred_grid == puzzle_to_grid(solution, is_4x4)))
                total_blanks = np.sum(original_grid == 0)
                
                print(f"Accuracy: {accuracy:.1%} | Filled: {correct_fills}/{total_blanks}")
            
            if finished or step >= 10:
                break
    
    print("\nCORRECT SOLUTION:")
    solution_grid = puzzle_to_grid(solution, is_4x4)
    print(text_grid(solution_grid))
    
    # Final comparison
    is_correct = np.all(pred_grid == solution_grid)
    print(f"\n✓ SOLVED CORRECTLY!" if is_correct else "\n✗ INCORRECT SOLUTION")
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--num-puzzles", type=int, default=2)
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, is_4x4 = load_checkpoint(args.checkpoint)
    
    # Load test puzzles
    puzzles, solutions = load_test_puzzles(args.data_path, args.num_puzzles)
    
    # Display each puzzle
    for idx, (puzzle, solution) in enumerate(zip(puzzles, solutions)):
        print(f"\n\n{'='*20} PUZZLE {idx + 1} {'='*20}")
        solve_and_display(model, puzzle, solution, is_4x4)


if __name__ == "__main__":
    main()