import numpy as np
import time
import random
from py_z3.classes import GameOfLifeGrid
from solvers import Z3PredecessorFinder, SATPredecessorFinder, ILPPredecessorFinder
from generate_alphabet import GenerateText

def test_multi_solution():
    print("Testing Multi-Solution Generation (Target: 64 solutions per letter)")
    
    solvers = [
        ("SAT", SATPredecessorFinder()),
        ("Z3", Z3PredecessorFinder()),
        ("ILP", ILPPredecessorFinder())
    ]
    
    gen = GenerateText()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # letters = "A" # For quick testing
    
    target_solutions = 64
    
    results = {}
    
    for name, solver in solvers:
        print(f"\n--- Solver: {name} ---")
        results[name] = {}
        
        total_start = time.time()
        
        for char in letters:
            grid = gen.text_to_grid(char)
            found_solutions = []
            
            start_time = time.time()
            
            for i in range(target_solutions):
                # Use a random seed for diversity, though exclude should force difference anyway
                seed = random.randint(0, 1000000)
                
                sol = solver.find_previous(grid, seed=seed, exclude=found_solutions)
                
                if sol is not None:
                    found_solutions.append(sol)
                else:
                    # No more solutions found
                    break
            
            elapsed = time.time() - start_time
            count = len(found_solutions)
            results[name][char] = count
            
            print(f"Letter {char}: Found {count}/{target_solutions} solutions in {elapsed:.4f}s")
            
            # Verify solutions are distinct (sanity check)
            # (exclude logic should guarantee this, but good to check)
            # And verify they are valid predecessors
            for idx, sol in enumerate(found_solutions):
                g = GameOfLifeGrid(grid=sol)
                g.compute_next()
                if not np.array_equal(g.next_grid, grid):
                    print(f"  ERROR: Solution {idx} is invalid!")
        
        total_elapsed = time.time() - total_start
        print(f"{name} Total Time: {total_elapsed:.2f}s")

    print("\n--- Summary ---")
    print("Letter | SAT | Z3  | ILP")
    print("-------|-----|-----|-----")
    for char in letters:
        s = results["SAT"].get(char, 0)
        z = results["Z3"].get(char, 0)
        i = results["ILP"].get(char, 0)
        print(f"   {char}   | {s:3d} | {z:3d} | {i:3d}")

if __name__ == "__main__":
    test_multi_solution()
