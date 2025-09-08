#!/usr/bin/env python3
"""Build a 6x6 Sudoku dataset - intermediate complexity between 4x4 and 9x9."""

import os
import json
import numpy as np
from typing import Tuple
from tqdm import tqdm
from common import PuzzleDatasetMetadata


def generate_6x6_sudoku(num_blanks: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 6x6 Sudoku puzzle with solution.
    
    6x6 Sudoku has:
    - 6 rows, 6 columns
    - 6 2x3 rectangular regions
    - Numbers 1-6
    """
    # Start with a valid 6x6 solution
    # The pattern ensures all constraints are met
    base = np.array([
        [1, 2, 3, 4, 5, 6],
        [4, 5, 6, 1, 2, 3],
        [2, 3, 4, 5, 6, 1],
        [5, 6, 1, 2, 3, 4],
        [3, 4, 5, 6, 1, 2],
        [6, 1, 2, 3, 4, 5]
    ])
    
    # Shuffle rows within 2-row bands (maintaining 2x3 box constraints)
    for band_start in [0, 2, 4]:
        if np.random.rand() > 0.5:
            base[[band_start, band_start+1]] = base[[band_start+1, band_start]]
    
    # Shuffle columns within 3-column bands
    for band_start in [0, 3]:
        cols = [band_start, band_start+1, band_start+2]
        np.random.shuffle(cols)
        base[:, [band_start, band_start+1, band_start+2]] = base[:, cols]
    
    # Shuffle the 2-row bands themselves
    band_order = np.random.permutation(3)
    new_base = np.vstack([base[b*2:(b+1)*2] for b in band_order])
    base = new_base
    
    # Shuffle the 3-column bands
    band_order = np.random.permutation(2)
    new_base = np.hstack([base[:, b*3:(b+1)*3] for b in band_order])
    base = new_base
    
    # Shuffle digits (relabel numbers)
    perm = np.random.permutation(6) + 1
    solution = perm[base - 1]
    
    # Create puzzle by removing cells
    puzzle = solution.copy()
    blank_positions = np.random.choice(36, size=min(num_blanks, 35), replace=False)
    puzzle.flat[blank_positions] = 0
    
    return puzzle, solution


def verify_6x6_solution(puzzle: np.ndarray, solution: np.ndarray) -> bool:
    """Verify that a 6x6 Sudoku solution is valid."""
    # Check rows
    for row in solution:
        if len(set(row)) != 6 or min(row) != 1 or max(row) != 6:
            return False
    
    # Check columns
    for col in solution.T:
        if len(set(col)) != 6 or min(col) != 1 or max(col) != 6:
            return False
    
    # Check 2x3 boxes
    for box_row in range(3):
        for box_col in range(2):
            box = solution[box_row*2:(box_row+1)*2, box_col*3:(box_col+1)*3].flatten()
            if len(set(box)) != 6 or min(box) != 1 or max(box) != 6:
                return False
    
    # Check puzzle matches solution at non-blank positions
    non_blank = puzzle > 0
    if not np.all(puzzle[non_blank] == solution[non_blank]):
        return False
    
    return True


def build_6x6_dataset(num_puzzles: int = 5000, num_blanks: int = 12, output_dir: str = "data/sudoku-6x6"):
    """Build a 6x6 Sudoku dataset."""
    
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # 80% train, 20% test split
    train_size = int(num_puzzles * 0.8)
    
    for split, size in [("train", train_size), ("test", num_puzzles - train_size)]:
        print(f"Generating {size} {split} puzzles...")
        
        inputs = []
        labels = []
        puzzle_indices = [0]
        group_indices = [0]
        puzzle_identifiers = []
        
        # Generate puzzles with verification
        generated = 0
        attempts = 0
        pbar = tqdm(total=size)
        
        while generated < size:
            puzzle, solution = generate_6x6_sudoku(num_blanks)
            attempts += 1
            
            # Verify the puzzle is valid
            if verify_6x6_solution(puzzle, solution):
                # Flatten to 1D (36 cells for 6x6)
                inputs.append(puzzle.flatten())
                labels.append(solution.flatten())
                puzzle_identifiers.append(0)
                puzzle_indices.append(generated + 1)
                group_indices.append(generated + 1)
                generated += 1
                pbar.update(1)
            
            if attempts > size * 10:
                print(f"Warning: High rejection rate. Generated {generated}/{size} after {attempts} attempts")
                break
        
        pbar.close()
        print(f"Generated {generated} valid puzzles from {attempts} attempts")
        
        # Convert to numpy arrays with correct dtype
        # Add 1 to account for PAD token (0 -> PAD, 1 -> blank, 2-7 -> digits 1-6)
        inputs_arr = np.array(inputs, dtype=np.int32) + 1
        labels_arr = np.array(labels, dtype=np.int32) + 1
        
        # Save arrays
        save_dir = os.path.join(output_dir, split)
        np.save(os.path.join(save_dir, "all__inputs.npy"), inputs_arr)
        np.save(os.path.join(save_dir, "all__labels.npy"), labels_arr)
        np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), np.array(puzzle_indices, dtype=np.int32))
        np.save(os.path.join(save_dir, "all__group_indices.npy"), np.array(group_indices, dtype=np.int32))
        np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), np.array(puzzle_identifiers, dtype=np.int32))
        
        # Save metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=36,  # 6x6 = 36 cells
            vocab_size=8,  # PAD + blank + digits 1-6
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=generated,
            mean_puzzle_examples=1,
            sets=["all"]
        )
        
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
    
    # Save identifiers
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    print(f"\nDataset saved to {output_dir}")
    print(f"Puzzle size: 6x6 (36 cells)")
    print(f"Vocab size: 8 (PAD + blank + 1-6)")
    print(f"Number of blanks: {num_blanks}/36")
    print(f"Difficulty: Intermediate (between 4x4 and 9x9)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-puzzles", type=int, default=5000, 
                       help="Number of puzzles to generate")
    parser.add_argument("--num-blanks", type=int, default=12, 
                       help="Number of blank cells per puzzle (1-35)")
    parser.add_argument("--output-dir", type=str, default="data/sudoku-6x6", 
                       help="Output directory")
    
    args = parser.parse_args()
    build_6x6_dataset(args.num_puzzles, args.num_blanks, args.output_dir)