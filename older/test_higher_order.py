import numpy as np
import time
from py_z3.classes import GameOfLifeGrid
from solvers import SATPredecessorFinder
from higher_order_solvers import NaiveHigherOrderSolver

def test_higher_order():
    print("Testing NaiveHigherOrderSolver with SATPredecessorFinder...")
    
    finder = SATPredecessorFinder()
    solver = NaiveHigherOrderSolver(finder)
    
    # Test 1: Blinker (Period 2)
    # If we go back 10 steps, we should find a valid chain.
    # Note: Naive solver might drift into non-blinker states that eventually evolve to blinker,
    # but for a blinker, the immediate predecessor is usually the other phase of the blinker.
    
    target = np.zeros((10, 10), dtype=int)
    # Vertical blinker at (5,5)
    target[4:7, 5] = 1
    
    print("Target (Blinker Phase A):")
    # print(target)
    
    steps = 10
    start = time.time()
    path = solver.solve(target, steps)
    end = time.time()
    
    if path is None:
        print(f"FAILED: Could not find path of length {steps}.")
    else:
        print(f"SUCCESS: Found path of length {len(path)-1} (requested {steps}) in {end-start:.4f}s.")
        
        # Verify the chain forward
        valid = True
        for i in range(len(path) - 1):
            current = path[i]
            next_expected = path[i+1]
            
            grid = GameOfLifeGrid(grid=current)
            grid.compute_next()
            
            if not np.array_equal(grid.next_grid, next_expected):
                print(f"  Mismatch at step {i} -> {i+1}")
                valid = False
                break
        
        if valid:
            print("  Forward verification PASSED.")
        else:
            print("  Forward verification FAILED.")

    # Test 2: Glider (Period 4, moves)
    # Glider at t=0
    # . # .
    # . . #
    # # # #
    # (This is one phase)
    
    glider = np.zeros((10, 10), dtype=int)
    glider[1, 2] = 1
    glider[2, 3] = 1
    glider[3, 1:4] = 1
    
    # Let's evolve it 4 steps to get a target, then try to backtrack 4 steps.
    g = GameOfLifeGrid(grid=glider)
    for _ in range(4):
        g.advance()
    target_glider = g.grid
    
    print("\nTarget (Glider at t+4):")
    
    steps = 4
    start = time.time()
    path_glider = solver.solve(target_glider, steps)
    end = time.time()
    
    if path_glider is None:
        print(f"FAILED: Could not backtrack glider {steps} steps.")
    else:
        print(f"SUCCESS: Found glider path of length {steps} in {end-start:.4f}s.")
        # Verify
        valid = True
        for i in range(len(path_glider) - 1):
            current = path_glider[i]
            next_expected = path_glider[i+1]
            grid = GameOfLifeGrid(grid=current)
            grid.compute_next()
            if not np.array_equal(grid.next_grid, next_expected):
                valid = False
                break
        if valid:
            print("  Forward verification PASSED.")
        else:
            print("  Forward verification FAILED.")

if __name__ == "__main__":
    test_higher_order()
