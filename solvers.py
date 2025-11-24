import numpy as np
import os
import subprocess
import tempfile
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
from py_z3.classes import PredecessorFinder
from z3 import Solver, Int, Sum, If, Or, And, sat
from scipy.optimize import milp, LinearConstraint, Bounds

class Z3PredecessorFinder(PredecessorFinder):
    def find_previous(self, grid: np.ndarray, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        rows, cols = grid.shape
        s = Solver()
        
        if seed is not None:
            from z3 import set_param
            set_param('smt.random_seed', seed)
            set_param('sat.random_seed', seed)
        
        # Variables: Bool
        from z3 import Bool
        vars_grid = [[Bool(f"c_{r}_{c}") for c in range(cols)] for r in range(rows)]
        
        # Neighbors helper (Wrapping)
        def get_neighs(r, c):
            ns = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = (r + dr) % rows, (c + dc) % cols
                    ns.append(vars_grid[nr][nc])
            return ns

        # Rules
        for r in range(rows):
            for c in range(cols):
                v = vars_grid[r][c]
                neighs = get_neighs(r, c)
                
                # Sum of neighbors (convert Bool to Int for summing)
                # Z3 optimizes Sum([If(b,1,0)...]) well
                s_sum = Sum([If(n, 1, 0) for n in neighs])
                
                # Logic:
                # Next=1 <==> (v AND (s==2 OR s==3)) OR (NOT v AND s==3)
                # Simplified: (s==3) OR (v AND s==2)
                
                is_alive_next = Or(
                    s_sum == 3,
                    And(v, s_sum == 2)
                )
                
                if grid[r, c] == 1:
                    s.add(is_alive_next)
                else:
                    s.add(z3_not(is_alive_next))
                    
        # Exclude constraints
        if exclude:
            for ex_grid in exclude:
                # Add constraint: NOT (all cells match ex_grid)
                # Equivalent to: OR (at least one cell differs)
                differs = []
                for r in range(rows):
                    for c in range(cols):
                        if ex_grid[r, c] == 1:
                            differs.append(z3_not(vars_grid[r][c])) # Must be 0 to differ
                        else:
                            differs.append(vars_grid[r][c]) # Must be 1 to differ
                s.add(Or(differs))

        if s.check() == sat:
            m = s.model()
            res = np.zeros((rows, cols), dtype=int)
            for r in range(rows):
                for c in range(cols):
                    # is_true returns True if the boolean is true in the model
                    if is_true(m[vars_grid[r][c]]):
                        res[r, c] = 1
            return res
        return None

def z3_not(expr):
    from z3 import Not
    return Not(expr)

def is_true(val):
    from z3 import is_true
    return is_true(val)


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


class ILPPredecessorFinder(PredecessorFinder):
    def find_previous(self, grid: np.ndarray, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        # ILP doesn't easily support seed/exclude in this simple wrapper without adding constraints.
        # For exclude, we'd need to add a constraint that sum(diffs) >= 1.
        # |x - y| = x(1-y) + y(1-x).
        # sum(x_i * (1-ex_i) + (1-x_i) * ex_i) >= 1
        # This is linearizable.
        # But for now, let's just ignore seed/exclude for ILP or implement basic support.
        
        rows, cols = grid.shape
        N = rows * cols
        
        x_map = {}
        u_map = {}
        var_count = 0
        
        for r in range(rows):
            for c in range(cols):
                x_map[(r,c)] = var_count
                var_count += 1
                
        dead_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r,c] == 0:
                    u_map[(r,c)] = [var_count]
                    var_count += 1
                    dead_cells.append((r,c))
                    
        c_vec = np.zeros(var_count)
        integrality = np.ones(var_count)
        b_l = np.zeros(var_count)
        b_u = np.ones(var_count)
        
        def get_neigh_indices(r, c):
            idx = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr==0 and dc==0: continue
                    nr, nc = (r+dr)%rows, (c+dc)%cols
                    idx.append(x_map[(nr,nc)])
            return idx

        M = 9
        
        A_rows = []
        b_l_rows = []
        b_u_rows = []
        
        def add_constr(coeffs, lb, ub):
            row = np.zeros(var_count)
            for idx, val in coeffs.items():
                row[idx] = val
            A_rows.append(row)
            b_l_rows.append(lb)
            b_u_rows.append(ub)
            
        for r in range(rows):
            for c in range(cols):
                neighs = get_neigh_indices(r, c)
                
                if grid[r,c] == 1:
                    # 2 <= n <= 3
                    coeffs = {ni: 1 for ni in neighs}
                    add_constr(coeffs, 2, 3)
                    
                    # x + n >= 3
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[x_map[(r,c)]] = 1
                    add_constr(coeffs, 3, M + 1)
                else:
                    # Dead cell logic (Optimized)
                    # y=0 implies:
                    # 1. n != 3
                    # 2. if n=2 then x=0
                    
                    # We use 1 binary aux var u:
                    # u=0 => n <= 2
                    # u=1 => n >= 4
                    
                    # Constraints:
                    # n <= 2 + M*u
                    # n >= 4 - M*(1-u)
                    # n + x <= 2 + M*u  (Enforces n=2 => x=0 when u=0)
                    
                    u = u_map[(r,c)][0] # We only need 1 var now
                    
                    # n <= 2 + M*u => n - M*u <= 2
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[u] = -M
                    add_constr(coeffs, -np.inf, 2)
                    
                    # n >= 4 - M(1-u) => n - M*u >= 4 - M
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[u] = -M
                    add_constr(coeffs, 4 - M, np.inf)
                    
                    # n + x <= 2 + M*u => n + x - M*u <= 2
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[x_map[(r,c)]] = 1
                    coeffs[u] = -M
                    add_constr(coeffs, -np.inf, 2)

        # Exclude constraints
        if exclude:
            for ex_grid in exclude:
                # sum(x_i if ex_i=0 else (1-x_i)) >= 1
                # sum(x_i * (1-ex_i) - x_i * ex_i) >= 1 - sum(ex_i)
                # coeffs[x_i] = 1 if ex_i=0 else -1
                # lb = 1 - sum(ex_i)
                coeffs = {}
                sum_ex = 0
                for r in range(rows):
                    for c in range(cols):
                        idx = x_map[(r,c)]
                        if ex_grid[r,c] == 1:
                            coeffs[idx] = -1
                            sum_ex += 1
                        else:
                            coeffs[idx] = 1
                add_constr(coeffs, 1 - sum_ex, np.inf)

        res = milp(c=c_vec, integrality=integrality, bounds=Bounds(b_l, b_u), 
                   constraints=LinearConstraint(A_rows, b_l_rows, b_u_rows))
                   
        if res.success:
            sol_grid = np.zeros((rows, cols), dtype=int)
            for r in range(rows):
                for c in range(cols):
                    idx = x_map[(r,c)]
                    if res.x[idx] > 0.5:
                        sol_grid[r,c] = 1
            return sol_grid
        else:
            return None
