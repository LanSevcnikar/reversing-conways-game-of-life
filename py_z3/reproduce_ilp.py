import numpy as np
from minimizing_solvers import ILPMinimizerSolver

def test_ilp_minimization():
    solver = ILPMinimizerSolver()
    
    # Test Case 1: Empty Grid
    # Target: 5x5 empty
    # Expected Predecessor: 5x5 empty (0 live cells)
    print("Test Case 1: Empty Grid (Default behavior - should be optimized)")
    target = np.zeros((5, 5), dtype=int)
    pred = solver.find_previous(target)
    if pred is None:
        print("FAILED: No predecessor found")
    else:
        count = np.sum(pred)
        print(f"Predecessor found with {count} live cells")
        if count == 0:
            print("SUCCESS")
        else:
            print(f"FAILED: Non-minimal solution found: {count} cells.")

    # Test Case 2: Blinker (Vertical)
    print("\nTest Case 2: Blinker (Default behavior - should be optimized)")
    target = np.zeros((5, 5), dtype=int)
    target[1, 2] = 1
    target[2, 2] = 1
    target[3, 2] = 1
    
    pred = solver.find_previous(target)
    if pred is None:
        print("FAILED: No predecessor found")
    else:
        count = np.sum(pred)
        print(f"Predecessor found with {count} live cells")
        if count <= 3:
            print("SUCCESS")
        else:
            print(f"FAILED: Non-minimal solution found: {count} cells.")

    # Test Case 3: Block
    print("\nTest Case 3: Block (Default behavior - should be optimized)")
    target = np.zeros((5, 5), dtype=int)
    target[1, 1] = 1
    target[1, 2] = 1
    target[2, 1] = 1
    target[2, 2] = 1
    
    pred = solver.find_previous(target)
    if pred is None:
        print("FAILED: No predecessor found")
    else:
        count = np.sum(pred)
        print(f"Predecessor found with {count} live cells")
        if count <= 4:
            print("SUCCESS")
        else:
            print(f"FAILED: Non-minimal solution found: {count} cells.")

if __name__ == "__main__":
    test_ilp_minimization()
