import numpy as np
import os
import subprocess
import tempfile
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
from py_z3.classes import PredecessorFinder
from z3 import Solver, Int, Sum, If, Or, And, sat, Not, is_true, set_param, Bool
from scipy.optimize import milp, LinearConstraint, Bounds


class SATPredecessorFinder(PredecessorFinder):
    def __init__(self, kissat_path="./kissat/kissat-master/build/kissat"):
        self.kissat_path = kissat_path
        self._cache_clauses()

    def _cache_clauses(self):
        # Pre-compute clauses for a generic cell and its 8 neighbors.
        # Variables: 0 (self), 1..8 (neighbors)
        # We generate clauses that FORBID invalid states for Target=0 and Target=1.
        
        self.clauses_for_0 = [] # Clauses to enforce if target is 0
        self.clauses_for_1 = [] # Clauses to enforce if target is 1
        
        # Iterate all 2^9 = 512 states
        # State bits: [self, n1, n2, ..., n8]
        for i in range(512):
            current_alive = (i >> 0) & 1
            neigh_alive_count = sum((i >> j) & 1 for j in range(1, 9))
            
            next_alive = 0
            if current_alive == 1:
                if neigh_alive_count in [2, 3]:
                    next_alive = 1
            else:
                if neigh_alive_count == 3:
                    next_alive = 1
            
            # If this state leads to 1, it must be forbidden if target is 0.
            # If this state leads to 0, it must be forbidden if target is 1.
            
            # Clause to forbid state 'i':
            # If bit j is 1, literal is -var_j
            # If bit j is 0, literal is var_j
            # Vars are 0..8
            clause = []
            for bit in range(9):
                if (i >> bit) & 1:
                    clause.append(-(bit)) # We use 0-based relative index here, will offset later
                else:
                    clause.append(bit)
            
            if next_alive == 1:
                self.clauses_for_0.append(clause)
            else:
                self.clauses_for_1.append(clause)

    def find_previous(self, grid: np.ndarray, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        return self._find_previous_optimized(grid, seed, exclude)

    def _find_previous_optimized(self, grid, seed, exclude):
        rows, cols = grid.shape
        def var_idx(r, c): return r * cols + c + 1
        
        clauses = []
        
        # Re-generate templates with (index, sign)
        # index 0..8. sign True if negated.
        if not hasattr(self, 'tmpl_0'):
            self.tmpl_0 = []
            self.tmpl_1 = []
            for i in range(512):
                current_alive = (i >> 0) & 1
                neigh_alive_count = sum((i >> j) & 1 for j in range(1, 9))
                next_alive = 1 if (current_alive and neigh_alive_count in [2,3]) or (not current_alive and neigh_alive_count==3) else 0
                
                # Clause to forbid this state
                # list of (bit_index, must_be_negated)
                # If bit is 1 in state, we need NOT var -> must_be_negated=True
                clause_spec = []
                for bit in range(9):
                    is_set = (i >> bit) & 1
                    clause_spec.append((bit, is_set == 1))
                
                if next_alive == 1:
                    self.tmpl_0.append(clause_spec)
                else:
                    self.tmpl_1.append(clause_spec)

        for r in range(rows):
            for c in range(cols):
                neigh_coords = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = (r + dr) % rows, (c + dc) % cols
                        neigh_coords.append((nr, nc))
                
                my_var = var_idx(r, c)
                neigh_vars = [var_idx(nr, nc) for nr, nc in neigh_coords]
                var_map = [my_var] + neigh_vars
                
                target_state = grid[r, c]
                template = self.tmpl_1 if target_state == 1 else self.tmpl_0
                
                for spec in template:
                    clause = []
                    for idx, must_be_negated in spec:
                        abs_var = var_map[idx]
                        if must_be_negated:
                            clause.append(-abs_var)
                        else:
                            clause.append(abs_var)
                    clauses.append(clause)
                    
        # Exclude constraints
        if exclude:
            for ex_grid in exclude:
                # Clause: OR (var != val)
                # If val=1, literal is -var
                # If val=0, literal is var
                clause = []
                for r in range(rows):
                    for c in range(cols):
                        idx = var_idx(r, c)
                        if ex_grid[r, c] == 1:
                            clause.append(-idx)
                        else:
                            clause.append(idx)
                clauses.append(clause)

        num_vars = rows * cols
        cnf_content = f"p cnf {num_vars} {len(clauses)}\n"
        for cl in clauses:
            cnf_content += " ".join(map(str, cl)) + " 0\n"
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as tmp_cnf:
            tmp_cnf.write(cnf_content)
            tmp_cnf_path = tmp_cnf.name
            
        try:
            cmd = [self.kissat_path]
            if seed is not None:
                cmd.append(f"--seed={seed}")
            cmd.append(tmp_cnf_path)
            
            res = subprocess.run(cmd, capture_output=True, text=True)
            
            if res.returncode == 10:
                model = {}
                for line in res.stdout.splitlines():
                    if line.startswith('v'):
                        parts = line.split()[1:]
                        for p in parts:
                            val = int(p)
                            if val == 0: break
                            model[abs(val)] = 1 if val > 0 else 0
                result_grid = np.zeros((rows, cols), dtype=int)
                for r in range(rows):
                    for c in range(cols):
                        idx = var_idx(r, c)
                        result_grid[r, c] = model.get(idx, 0)
                return result_grid
            else:
                return None
        finally:
            if os.path.exists(tmp_cnf_path):
                os.remove(tmp_cnf_path)
