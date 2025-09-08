#!/usr/bin/env python3
"""Build a simple 4x4 Sudoku dataset for faster training and testing."""

import os
import json
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from common import PuzzleDatasetMetadata


def generate_4x4_sudoku(num_blanks: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 4x4 Sudoku puzzle with solution."""
    # Start with a valid 4x4 solution
    base = np.array([
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1]
    ])
    
    # Shuffle rows within bands
    for band in [0, 2]:
        idx = [band, band+1]
        np.random.shuffle(idx)
        base[[band, band+1]] = base[idx]
    
    # Shuffle columns within bands  
    for band in [0, 2]:
        idx = [band, band+1]
        np.random.shuffle(idx)
        base[:, [band, band+1]] = base[:, idx]
    
    # Shuffle digits
    perm = np.random.permutation(4) + 1
    solution = perm[base - 1]
    
    # Create puzzle by removing cells
    puzzle = solution.copy()
    blank_positions = np.random.choice(16, size=num_blanks, replace=False)
    puzzle.flat[blank_positions] = 0
    
    return puzzle, solution


def build_4x4_dataset(num_puzzles: int = 1000, num_blanks: int = 6, output_dir: str = "data/sudoku-4x4"):
    """Build a 4x4 Sudoku dataset."""
    
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
        
        for i in tqdm(range(size)):
            puzzle, solution = generate_4x4_sudoku(num_blanks)
            
            # Flatten to 1D (16 cells for 4x4)
            inputs.append(puzzle.flatten())
            labels.append(solution.flatten())
            puzzle_identifiers.append(0)
            puzzle_indices.append(i + 1)
            group_indices.append(i + 1)
        
        # Convert to numpy arrays with correct dtype
        # Add 1 to account for PAD token (0 -> PAD, 1 -> blank, 2-5 -> digits 1-4)
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
            seq_len=16,  # 4x4 = 16 cells
            vocab_size=6,  # PAD + blank + digits 1-4
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=size,
            mean_puzzle_examples=1,
            sets=["all"]
        )
        
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
    
    # Save identifiers
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    print(f"Dataset saved to {output_dir}")
    print(f"Puzzle size: 4x4 (16 cells)")
    print(f"Vocab size: 6 (PAD + blank + 1-4)")
    print(f"Number of blanks: {num_blanks}/16")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-puzzles", type=int, default=10000, help="Number of puzzles to generate")
    parser.add_argument("--num-blanks", type=int, default=6, help="Number of blank cells per puzzle (1-15)")
    parser.add_argument("--output-dir", type=str, default="data/sudoku-4x4", help="Output directory")
    
    args = parser.parse_args()
    build_4x4_dataset(args.num_puzzles, args.num_blanks, args.output_dir)