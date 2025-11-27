import numpy as np
import os
import sys

# Add current directory to path to import solvers
sys.path.append(os.getcwd())

from solvers import Z3PredecessorFinder, SATPredecessorFinder, ILPPredecessorFinder
from minimizing_solvers import Z3MinimizerSolver, SATMinimizerSolver, ILPMinimizerSolver

def load_pattern(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    grid = []
    for line in lines:
        if line.strip():
            row = [int(x) for x in line.strip().split(',')]
            grid.append(row)
    
    width = len(grid[0])
    height = len(grid)
    ## add 4 padding to the grid
    new_grid = [[0] * (width + 8) for _ in range(height + 8)]
    for i in range(height):
        for j in range(width):
            new_grid[i + 4][j + 4] = grid[i][j]
    return np.array(new_grid)

def print_grid(grid):
    for row in grid:
        print(''.join(['#' if x else '.' for x in row]))

def main():
    pattern_path = 'alphabet/a.txt'
    if not os.path.exists(pattern_path):
        print(f"Error: {pattern_path} not found.")
        return

    print(f"Loading pattern from {pattern_path}...")
    target_grid = load_pattern(pattern_path)

    print("Target Pattern:")
    print_grid(target_grid)
    print("-" * 20)

    solvers = [
        ("Z3 Standard", Z3PredecessorFinder()),
        ("SAT Standard", SATPredecessorFinder()),
        ("ILP Standard", ILPPredecessorFinder()),
        ("Z3 Minimizer", Z3MinimizerSolver()),
        ("SAT Minimizer", SATMinimizerSolver()),
        ("ILP Minimizer", ILPMinimizerSolver())
    ]

    for name, solver in solvers:
        print(f"\nRunning {name}...")
        try:
            # For minimizers, we just call find_previous without extra args for now to get *a* predecessor
            # The user asked for "previous state", not specifically minimized, but using the *solvers*
            # However, minimizing solvers might need to be called differently if we want them to actually minimize.
            # But the interface seems compatible for a basic find_previous.
            # Let's check if they have specific requirements.
            # Z3MinimizerSolver.find_previous(self, grid, seed=None, exclude=None, max_alive=None)
            # It seems compatible.
            
            result = solver.find_previous(target_grid)
            
            
            if result is not None:
                print(f"{name} found a predecessor:")
                print_grid(result)
            else:
                print(f"{name} found NO predecessor.")
        except Exception as e:
            print(f"{name} failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
