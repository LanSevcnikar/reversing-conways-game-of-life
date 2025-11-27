import numpy as np
from typing import List, Optional, Tuple
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


class NaiveDFSSolver(HigherOrderSolver):
    """
    A naive DFS solver that exhaustively backtracks through the search space.
    Uses depth-first search to explore all possible predecessor paths,
    backtracking when hitting dead ends and trying alternative predecessors.
    """
    
    def __init__(self, finder: PredecessorFinder, max_attempts_per_level: int = 1000, verbose: bool = True):
        """
        Args:
            finder: The PredecessorFinder to use for finding single-step predecessors
            max_attempts_per_level: Maximum number of different predecessors to try at each level
            verbose: Whether to print progress information
        """
        super().__init__(finder)
        self.max_attempts_per_level = max_attempts_per_level
        self.verbose = verbose
        self.stats = {
            'total_attempts': 0,
            'backtracks': 0,
            'max_depth_reached': 0
        }
    
    def solve(self, target_grid: np.ndarray, steps: int) -> Optional[List[np.ndarray]]:
        """
        Find a sequence of grids going back 'steps' generations using DFS with backtracking.
        
        Args:
            target_grid: The final grid state we want to reach
            steps: Number of generations to go back
            
        Returns:
            List of grids [g_0, g_1, ..., g_steps] where g_steps == target_grid,
            or None if no solution found
        """
        if steps == 0:
            return [target_grid]
        
        # Reset statistics
        self.stats = {
            'total_attempts': 0,
            'backtracks': 0,
            'max_depth_reached': 0
        }
        
        if self.verbose:
            print(f"\n=== Starting DFS search for {steps} steps back ===")
        
        # Stack contains tuples of (grid, excluded_predecessors, attempt_count)
        stack: List[Tuple[np.ndarray, List[np.ndarray], int]] = [(target_grid, [], 0)]
        
        iteration = 0
        while stack:
            iteration += 1
            
            # Check if we've found a complete path
            current_depth = len(stack) - 1
            if current_depth > self.stats['max_depth_reached']:
                self.stats['max_depth_reached'] = current_depth
            
            if len(stack) == steps + 1:
                # Success! Extract the path
                path = [item[0] for item in stack]
                if self.verbose:
                    print(f"\n✓ Solution found after {iteration} iterations!")
                    print(f"  Total attempts: {self.stats['total_attempts']}")
                    print(f"  Backtracks: {self.stats['backtracks']}")
                return path[::-1]  # Reverse to get chronological order
            
            # Get current state
            current_grid, excluded, attempt_count = stack[-1]
            depth = len(stack) - 1
            
            # Check if we've exceeded max attempts at this level
            if attempt_count >= self.max_attempts_per_level:
                if self.verbose:
                    print(f"Depth {depth}: Max attempts ({self.max_attempts_per_level}) reached, backtracking...")
                
                if len(stack) == 1:
                    # Can't backtrack further
                    if self.verbose:
                        print("\n✗ No solution found - exhausted all possibilities")
                    return None
                
                # Backtrack
                self.stats['backtracks'] += 1
                failed_grid, _, _ = stack.pop()
                
                # Add failed grid to parent's exclusion list
                parent_grid, parent_excluded, parent_attempts = stack[-1]
                parent_excluded.append(failed_grid)
                stack[-1] = (parent_grid, parent_excluded, parent_attempts + 1)
                continue
            
            # Try to find a new predecessor
            self.stats['total_attempts'] += 1
            
            # Use different seeds for diversity
            seed = random.randint(0, 1000000)
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Depth {depth}/{steps}, Attempts at level: {attempt_count}, Excluded: {len(excluded)}")
            
            predecessor = self.finder.find_previous(current_grid, seed=seed, exclude=excluded)
            
            if predecessor is not None:
                # Found a predecessor, go deeper
                if self.verbose and depth % 5 == 0:
                    print(f"Depth {depth}: Found predecessor (attempt {attempt_count + 1})")
                
                stack.append((predecessor, [], 0))
            else:
                # No predecessor found with current exclusions
                if self.verbose:
                    print(f"Depth {depth}: No more predecessors available (tried {attempt_count + 1} alternatives)")
                
                if len(stack) == 1:
                    # Can't backtrack from root
                    if self.verbose:
                        print("\n✗ No solution found - no valid predecessors for target")
                    return None
                
                # Backtrack
                self.stats['backtracks'] += 1
                failed_grid, _, _ = stack.pop()
                
                # Add to parent's exclusion list and increment parent's attempt count
                parent_grid, parent_excluded, parent_attempts = stack[-1]
                parent_excluded.append(failed_grid)
                stack[-1] = (parent_grid, parent_excluded, parent_attempts + 1)
        
        # Stack is empty without finding solution
        if self.verbose:
            print("\n✗ No solution found - search space exhausted")
        return None
    
    def get_stats(self) -> dict:
        """Return statistics from the last solve() call"""
        return self.stats.copy()
