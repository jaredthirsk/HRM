#!/usr/bin/env python3
"""Visualize how the model solves Sudoku puzzles."""

import os
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Import model components
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.losses import ACTLossHead


def load_checkpoint(checkpoint_path: str) -> torch.nn.Module:
    """Load a trained model checkpoint."""
    # Load config from checkpoint directory
    config_path = Path(checkpoint_path).parent / "all_config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config for 4x4 model
        config = {
            'arch': {
                'name': 'hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1',
                'loss': {'name': 'losses@ACTLossHead', 'loss_type': 'stablemax_cross_entropy'},
                'halt_exploration_prob': 0.1,
                'halt_max_steps': 4,
                'H_cycles': 2,
                'L_cycles': 2,
                'H_layers': 2,
                'L_layers': 2,
                'hidden_size': 256,
                'num_heads': 4,
                'expansion': 4,
                'puzzle_emb_ndim': 256,
                'pos_encodings': 'rope'
            }
        }
    
    # Determine if 4x4 or 9x9 based on checkpoint
    is_4x4 = '4x4' in checkpoint_path or config['arch'].get('hidden_size', 512) == 256
    
    model_config = {
        **config['arch'],
        'batch_size': 1,
        'vocab_size': 6 if is_4x4 else 11,  # 4x4: PAD + blank + 1-4, 9x9: PAD + blank + 1-9
        'seq_len': 16 if is_4x4 else 81,
        'num_puzzle_identifiers': 1,
        'causal': False
    }
    
    # Create model
    model = HierarchicalReasoningModel_ACTV1(model_config)
    model = ACTLossHead(model, loss_type='stablemax_cross_entropy')
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, is_4x4


def load_test_puzzles(data_path: str, num_puzzles: int = 5):
    """Load test puzzles from dataset."""
    test_dir = Path(data_path) / "test"
    
    inputs = np.load(test_dir / "all__inputs.npy")
    labels = np.load(test_dir / "all__labels.npy")
    
    # Sample random puzzles
    indices = np.random.choice(len(inputs), min(num_puzzles, len(inputs)), replace=False)
    
    return inputs[indices], labels[indices]


def puzzle_to_grid(puzzle: np.ndarray, is_4x4: bool = True):
    """Convert flat puzzle array to 2D grid."""
    size = 4 if is_4x4 else 9
    # Subtract 1 to convert from vocab (PAD=0, blank=1, digits=2+) back to standard (blank=0, digits=1+)
    grid = (puzzle - 1).reshape(size, size)
    grid[grid < 0] = 0  # PAD becomes blank
    return grid


def visualize_puzzle_solution(model: torch.nn.Module, puzzle: np.ndarray, solution: np.ndarray, is_4x4: bool = True):
    """Visualize how the model solves a single puzzle."""
    size = 4 if is_4x4 else 9
    
    # Prepare batch
    with torch.no_grad():
        batch = {
            "inputs": torch.tensor(puzzle).unsqueeze(0),
            "labels": torch.tensor(solution).unsqueeze(0),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.int32)
        }
        
        # Get initial carry
        carry = model.initial_carry(batch)
        
        # Collect predictions at each step
        all_predictions = []
        all_confidences = []
        steps_taken = 0
        
        while True:
            carry, _, metrics, preds, finished = model(carry=carry, batch=batch, return_keys=["logits"])
            
            if "logits" in preds:
                logits = preds["logits"][0]  # Remove batch dimension
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                confidence = torch.max(probs, dim=-1)[0]
                
                all_predictions.append(predictions.cpu().float().numpy())
                all_confidences.append(confidence.cpu().float().numpy())
            
            steps_taken += 1
            if finished or steps_taken >= 16:
                break
    
    # Create visualization
    fig = plt.figure(figsize=(16, 4) if is_4x4 else (20, 5))
    
    # Original puzzle
    ax1 = plt.subplot(1, steps_taken + 2, 1)
    plot_sudoku_grid(puzzle_to_grid(puzzle, is_4x4), puzzle_to_grid(puzzle, is_4x4), 
                     title="Input Puzzle", is_4x4=is_4x4, ax=ax1)
    
    # Predictions at each step
    for step in range(steps_taken):
        ax = plt.subplot(1, steps_taken + 2, step + 2)
        pred_grid = puzzle_to_grid(all_predictions[step], is_4x4)
        conf_grid = all_confidences[step].reshape(size, size)
        plot_sudoku_grid(puzzle_to_grid(puzzle, is_4x4), pred_grid, 
                        confidence=conf_grid, 
                        title=f"Step {step+1}", 
                        is_4x4=is_4x4, ax=ax)
    
    # Final/correct solution
    ax_last = plt.subplot(1, steps_taken + 2, steps_taken + 2)
    plot_sudoku_grid(puzzle_to_grid(solution, is_4x4), puzzle_to_grid(solution, is_4x4), 
                    title="Correct Solution", is_4x4=is_4x4, ax=ax_last)
    
    plt.suptitle(f"Model Solving Process ({steps_taken} steps)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, steps_taken


def plot_sudoku_grid(original: np.ndarray, predicted: np.ndarray, 
                     confidence: Optional[np.ndarray] = None,
                     title: str = "", is_4x4: bool = True, ax=None):
    """Plot a single Sudoku grid with original and predicted values."""
    if ax is None:
        ax = plt.gca()
    
    size = 4 if is_4x4 else 9
    box_size = 2 if is_4x4 else 3
    
    # Clear axis
    ax.clear()
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10)
    
    # Draw grid lines
    for i in range(size + 1):
        lw = 2 if i % box_size == 0 else 0.5
        ax.axhline(i, color='black', linewidth=lw)
        ax.axvline(i, color='black', linewidth=lw)
    
    # Fill cells
    for i in range(size):
        for j in range(size):
            x, y = j, size - i - 1
            
            # Background color based on confidence
            if confidence is not None:
                alpha = confidence[i, j] * 0.3
                color = 'green' if predicted[i, j] > 0 else 'white'
                rect = patches.Rectangle((x, y), 1, 1, 
                                        linewidth=0, 
                                        facecolor=color, 
                                        alpha=alpha)
                ax.add_patch(rect)
            
            # Number display
            if original[i, j] > 0:  # Given number
                ax.text(x + 0.5, y + 0.5, str(int(original[i, j])),
                       ha='center', va='center', fontsize=14, 
                       fontweight='bold', color='black')
            elif predicted[i, j] > 0:  # Predicted number
                color = 'blue'
                if confidence is not None and confidence[i, j] < 0.5:
                    color = 'red'  # Low confidence
                ax.text(x + 0.5, y + 0.5, str(int(predicted[i, j])),
                       ha='center', va='center', fontsize=12,
                       color=color)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--num-puzzles", type=int, default=3, help="Number of puzzles to visualize")
    parser.add_argument("--save-dir", type=str, default="visualizations", help="Directory to save images")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, is_4x4 = load_checkpoint(args.checkpoint)
    
    # Load test puzzles
    print(f"Loading test puzzles from {args.data_path}...")
    puzzles, solutions = load_test_puzzles(args.data_path, args.num_puzzles)
    
    # Visualize each puzzle
    for idx, (puzzle, solution) in enumerate(zip(puzzles, solutions)):
        print(f"\nVisualizing puzzle {idx + 1}/{len(puzzles)}...")
        
        fig, steps = visualize_puzzle_solution(model, puzzle, solution, is_4x4)
        
        # Check accuracy
        with torch.no_grad():
            batch = {
                "inputs": torch.tensor(puzzle).unsqueeze(0),
                "labels": torch.tensor(solution).unsqueeze(0),
                "puzzle_identifiers": torch.zeros(1, dtype=torch.int32)
            }
            carry = model.initial_carry(batch)
            
            # Run until completion
            while True:
                carry, _, _, preds, finished = model(carry=carry, batch=batch, return_keys=["logits"])
                if finished:
                    break
            
            if "logits" in preds:
                final_pred = torch.argmax(preds["logits"][0], dim=-1).cpu().numpy()
                accuracy = np.mean(final_pred == solution)
                print(f"  Accuracy: {accuracy:.1%}")
                print(f"  Steps taken: {steps}")
        
        # Save figure
        save_path = os.path.join(args.save_dir, f"puzzle_{idx+1}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        
        plt.close(fig)
    
    print(f"\nAll visualizations saved to {args.save_dir}/")


if __name__ == "__main__":
    main()