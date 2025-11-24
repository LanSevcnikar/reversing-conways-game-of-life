import numpy as np
from typing import List, Optional
import random
from py_z3.classes import HigherOrderSolver, PredecessorFinder

class NaiveHigherOrderSolver(HigherOrderSolver):
    def __init__(self, finder: PredecessorFinder, max_retries: int = 10):
        super().__init__(finder)
        self.max_retries = max_retries

    def solve(self, target_grid: np.ndarray, steps: int) -> Optional[List[np.ndarray]]:
        """
        Backtracks 'steps' times using randomized DFS.
        If a path leads to a dead end, it backtracks and tries a different predecessor
        by excluding the one that failed.
        """
        
        # Stack stores: (current_grid, list_of_excluded_predecessors)
        # Initial state
        stack = [(target_grid, [])]
        
        while stack:
            if len(stack) == steps + 1:
                # Found full path
                return [s[0] for s in stack][::-1]
                
            current_grid, excluded = stack[-1]
            depth = len(stack) - 1
            
            # Try to find a predecessor
            # We use a random seed for diversity
            seed = random.randint(0, 100000)
            
            # print(f"Depth {depth}: searching... (excluded {len(excluded)})")
            prev = self.finder.find_previous(current_grid, seed=seed, exclude=excluded)
            
            if prev is not None:
                # Found one, push to stack
                print(f"Depth {depth}: Found predecessor!")
                stack.append((prev, []))
            else:
                # Dead end. Backtrack.
                print(f"Depth {depth}: Dead end. Backtracking...")
                
                if len(stack) == 1:
                    # We exhausted options for the target itself? 
                    # Or we just failed to find *any* predecessor for target with current exclusions.
                    # If we can't find predecessor for target, we are done (fail).
                    return None
                    
                failed_grid, _ = stack.pop()
                parent_grid, parent_excluded = stack[-1]
                
                # Add the failed grid to exclusion list
                parent_excluded.append(failed_grid)
                
                # Optimization: Limit retries per level to avoid infinite local minima search
                if len(parent_excluded) > 100:
                     print(f"Depth {depth-1}: Max retries reached. Backtracking further.")
                     if len(stack) == 1:
                         return None
                     stack.pop()
                     grandparent_grid, grandparent_excluded = stack[-1]
                     grandparent_excluded.append(parent_grid)
                
        return None
