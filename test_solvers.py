import numpy as np
import time
from py_z3.classes import GameOfLifeGrid
from solvers import Z3PredecessorFinder, SATPredecessorFinder, ILPPredecessorFinder

def test_solver(solver, name):
    print(f"Testing {name}...")
    
    # Test 1: Blinker
    # Target (vertical blinker)
    # . . . . .
    # . . # . .
    # . . # . .
    # . . # . .
    # . . . . .
    target = np.zeros((5, 5), dtype=int)
    target[1:4, 2] = 1
    
    start = time.time()
    pred = solver.find_previous(target)
    end = time.time()
    
    if pred is None:
        print(f"  FAILED: No predecessor found. ({end-start:.4f}s)")
        return
        
    # Verify
    grid = GameOfLifeGrid(grid=pred)
    grid.compute_next()
    if np.array_equal(grid.next_grid, target):
        print(f"  PASSED: Predecessor found and verified. ({end-start:.4f}s)")
    else:
        print(f"  FAILED: Predecessor found but invalid. ({end-start:.4f}s)")
        # print("  Predecessor:")
        # print(pred)
        # print("  Evolved:")
        # print(grid.next_grid)

def main():
    solvers = [
        (Z3PredecessorFinder(), "Z3 Solver"),
        (SATPredecessorFinder(), "SAT Solver (Kissat)"),
        (ILPPredecessorFinder(), "ILP Solver (Scipy)")
    ]
    
    for solver, name in solvers:
        try:
            test_solver(solver, name)
        except Exception as e:
            print(f"  ERROR: {name} raised exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
