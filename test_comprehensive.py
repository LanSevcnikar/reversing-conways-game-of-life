import numpy as np
import time
import sys
from py_z3.classes import GameOfLifeGrid
from solvers import Z3PredecessorFinder, SATPredecessorFinder, ILPPredecessorFinder
from generate_alphabet import GenerateText

def test_solvers_comprehensive():
    gen = GenerateText()
    solvers = [
        (Z3PredecessorFinder(), "Z3"),
        (SATPredecessorFinder(), "SAT"),
        (ILPPredecessorFinder(), "ILP")
    ]
    
    letters = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    
    results = {name: {'found': 0, 'failed': 0, 'invalid': 0, 'time': 0.0} for _, name in solvers}
    
    print(f"{'Letter':<8} | {'Z3':<10} | {'SAT':<10} | {'ILP':<10}")
    print("-" * 46)
    
    for char in letters:
        # Create grid for letter with padding
        target = gen.text_to_grid(char, padding=5)
        
        row_str = f"{char:<8} | "
        
        for solver, name in solvers:
            start = time.time()
            try:
                pred = solver.find_previous(target)
                end = time.time()
                duration = end - start
                results[name]['time'] += duration
                
                if pred is None:
                    row_str += f"{'None':<10} | "
                    results[name]['failed'] += 1
                else:
                    # Verify
                    grid = GameOfLifeGrid(grid=pred)
                    grid.compute_next()
                    if np.array_equal(grid.next_grid, target):
                        row_str += f"{'PASS':<10} | "
                        results[name]['found'] += 1
                    else:
                        row_str += f"{'FAIL':<10} | "
                        results[name]['invalid'] += 1
            except Exception as e:
                row_str += f"{'ERR':<10} | "
                # print(f"\nError in {name} for {char}: {e}")
        
        print(row_str)
        
    print("\nSummary:")
    for name, stats in results.items():
        print(f"{name}: Found={stats['found']}, Failed={stats['failed']}, Invalid={stats['invalid']}, Total Time={stats['time']:.2f}s")

if __name__ == "__main__":
    test_solvers_comprehensive()
