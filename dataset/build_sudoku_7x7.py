#!/usr/bin/env python3
"""Build a 7x7 Sudoku dataset - unique challenge with irregular regions."""

import os
import json
import numpy as np
from typing import Tuple, List
from tqdm import tqdm
from common import PuzzleDatasetMetadata


# Define the 7 irregular regions for 7x7 Sudoku (each region has exactly 7 cells)
REGIONS_7x7 = [
    # Region 0 (top-left L-shape)
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)],
    # Region 1 (top-right)
    [(0, 3), (0, 4), (0, 5), (0, 6), (1, 4), (1, 5), (1, 6)],
    # Region 2 (left-middle)
    [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3), (4, 3)],
    # Region 3 (center)
    [(2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 4)],
    # Region 4 (left-bottom)
    [(3, 0), (3, 1), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1)],
    # Region 5 (bottom-middle)
    [(4, 5), (4, 6), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)],
    # Region 6 (bottom)
    [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)]
]


def get_region_map_7x7() -> np.ndarray:
    """Create a 7x7 grid showing which region each cell belongs to."""
    region_map = np.zeros((7, 7), dtype=int)
    for region_id, cells in enumerate(REGIONS_7x7):
        for row, col in cells:
            region_map[row, col] = region_id
    return region_map


def is_valid_7x7(grid: np.ndarray, region_map: np.ndarray) -> bool:
    """Check if a 7x7 grid satisfies all Sudoku constraints."""
    # Check rows
    for row in grid:
        if len(set(row[row > 0])) != len(row[row > 0]):
            return False
    
    # Check columns
    for col in grid.T:
        if len(set(col[col > 0])) != len(col[col > 0]):
            return False
    
    # Check regions
    for region_id in range(7):
        region_cells = []
        for row, col in REGIONS_7x7[region_id]:
            if grid[row, col] > 0:
                region_cells.append(grid[row, col])
        if len(set(region_cells)) != len(region_cells):
            return False
    
    return True


def generate_7x7_sudoku_backtrack(num_blanks: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a 7x7 Sudoku using backtracking."""
    region_map = get_region_map_7x7()
    
    def solve(grid):
        """Backtracking solver."""
        for row in range(7):
            for col in range(7):
                if grid[row, col] == 0:
                    # Try digits 1-7 in random order
                    for num in np.random.permutation(7) + 1:
                        # Check row
                        if num in grid[row, :]:
                            continue
                        # Check column
                        if num in grid[:, col]:
                            continue
                        # Check region
                        region_id = region_map[row, col]
                        region_vals = [grid[r, c] for r, c in REGIONS_7x7[region_id]]
                        if num in region_vals:
                            continue
                        
                        # Place number and recurse
                        grid[row, col] = num
                        if solve(grid):
                            return True
                        grid[row, col] = 0
                    
                    return False
        return True
    
    # Generate complete solution
    solution = np.zeros((7, 7), dtype=int)
    
    # Seed with a few random values to ensure variety
    for _ in range(3):
        row, col = np.random.randint(0, 7, 2)
        val = np.random.randint(1, 8)
        solution[row, col] = val
        if not is_valid_7x7(solution, region_map):
            solution[row, col] = 0
    
    # Solve to get complete grid
    if not solve(solution):
        # If failed, try with empty grid
        solution = np.zeros((7, 7), dtype=int)
        solve(solution)
    
    # Create puzzle by removing cells
    puzzle = solution.copy()
    blank_positions = np.random.choice(49, size=min(num_blanks, 48), replace=False)
    puzzle.flat[blank_positions] = 0
    
    return puzzle, solution


def visualize_7x7_regions():
    """Print ASCII visualization of 7x7 regions."""
    region_map = get_region_map_7x7()
    symbols = ['█', '▓', '▒', '░', '▄', '▀', '■']
    
    print("\n7x7 Sudoku Regions:")
    print("=" * 15)
    for row in range(7):
        for col in range(7):
            print(symbols[region_map[row, col]], end=" ")
        print()
    print("=" * 15)
    print("Each symbol represents a different region\n")


def build_7x7_dataset(num_puzzles: int = 3000, num_blanks: int = 20, output_dir: str = "data/sudoku-7x7"):
    """Build a 7x7 Sudoku dataset."""
    
    # Show region structure
    visualize_7x7_regions()
    
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # 80% train, 20% test split
    train_size = int(num_puzzles * 0.8)
    
    region_map = get_region_map_7x7()
    
    for split, size in [("train", train_size), ("test", num_puzzles - train_size)]:
        print(f"Generating {size} {split} puzzles...")
        
        inputs = []
        labels = []
        puzzle_indices = [0]
        group_indices = [0]
        puzzle_identifiers = []
        
        # Generate puzzles
        generated = 0
        attempts = 0
        max_attempts = size * 20
        pbar = tqdm(total=size)
        
        while generated < size and attempts < max_attempts:
            attempts += 1
            
            try:
                puzzle, solution = generate_7x7_sudoku_backtrack(num_blanks)
                
                # Verify the solution is complete and valid
                if np.all(solution > 0) and is_valid_7x7(solution, region_map):
                    # Verify puzzle matches solution at non-blank positions
                    non_blank = puzzle > 0
                    if np.all(puzzle[non_blank] == solution[non_blank]):
                        inputs.append(puzzle.flatten())
                        labels.append(solution.flatten())
                        puzzle_identifiers.append(0)
                        puzzle_indices.append(generated + 1)
                        group_indices.append(generated + 1)
                        generated += 1
                        pbar.update(1)
            except:
                # Backtracking might fail occasionally, just try again
                continue
        
        pbar.close()
        print(f"Generated {generated} valid puzzles from {attempts} attempts")
        
        if generated == 0:
            print("Failed to generate any puzzles! Check the algorithm.")
            return
        
        # Convert to numpy arrays with correct dtype
        # Add 1 to account for PAD token (0 -> PAD, 1 -> blank, 2-8 -> digits 1-7)
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
            seq_len=49,  # 7x7 = 49 cells
            vocab_size=9,  # PAD + blank + digits 1-7
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
    
    # Save region map for reference
    np.save(os.path.join(output_dir, "region_map.npy"), region_map)
    
    print(f"\nDataset saved to {output_dir}")
    print(f"Puzzle size: 7x7 (49 cells)")
    print(f"Vocab size: 9 (PAD + blank + 1-7)")
    print(f"Number of blanks: {num_blanks}/49")
    print(f"Difficulty: Challenging (irregular regions)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-puzzles", type=int, default=3000, 
                       help="Number of puzzles to generate")
    parser.add_argument("--num-blanks", type=int, default=20, 
                       help="Number of blank cells per puzzle (1-48)")
    parser.add_argument("--output-dir", type=str, default="data/sudoku-7x7", 
                       help="Output directory")
    
    args = parser.parse_args()
    build_7x7_dataset(args.num_puzzles, args.num_blanks, args.output_dir)