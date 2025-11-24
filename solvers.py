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
    def find_previous(self, grid: np.ndarray) -> Optional[np.ndarray]:
        rows, cols = grid.shape
        s = Solver()
        
        # Variables: 0 or 1
        vars_grid = [[Int(f"c_{r}_{c}") for c in range(cols)] for r in range(rows)]
        
        for r in range(rows):
            for c in range(cols):
                v = vars_grid[r][c]
                s.add(v >= 0, v <= 1)
                
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
                s_sum = Sum(*neighs)
                
                is_alive_next = Or(
                    And(v == 1, Or(s_sum == 2, s_sum == 3)),
                    And(v == 0, s_sum == 3)
                )
                
                if grid[r, c] == 1:
                    s.add(is_alive_next)
                else:
                    s.add(z3_not(is_alive_next))

        if s.check() == sat:
            m = s.model()
            res = np.zeros((rows, cols), dtype=int)
            for r in range(rows):
                for c in range(cols):
                    res[r, c] = m[vars_grid[r][c]].as_long()
            return res
        return None

def z3_not(expr):
    from z3 import Not
    return Not(expr)


class SATPredecessorFinder(PredecessorFinder):
    def __init__(self, kissat_path="./kissat/kissat-master/build/kissat"):
        self.kissat_path = kissat_path

    def find_previous(self, grid: np.ndarray) -> Optional[np.ndarray]:
        rows, cols = grid.shape
        
        def var_idx(r, c):
            return r * cols + c + 1
            
        clauses = []
        
        for r in range(rows):
            for c in range(cols):
                # Identify neighbors (Wrapping)
                neigh_coords = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = (r + dr) % rows, (c + dc) % cols
                        neigh_coords.append((nr, nc))
                
                my_var = var_idx(r, c)
                neigh_vars = [var_idx(nr, nc) for nr, nc in neigh_coords]
                
                k = 1 + len(neigh_vars)
                target_state = grid[r, c]
                
                for i in range(1 << k):
                    current_alive = (i >> 0) & 1
                    neigh_alive_count = sum((i >> j) & 1 for j in range(1, k))
                    
                    next_alive = 0
                    if current_alive == 1:
                        if neigh_alive_count in [2, 3]:
                            next_alive = 1
                    else:
                        if neigh_alive_count == 3:
                            next_alive = 1
                            
                    if next_alive != target_state:
                        clause = []
                        if (i >> 0) & 1:
                            clause.append(-my_var)
                        else:
                            clause.append(my_var)
                            
                        for idx, nv in enumerate(neigh_vars):
                            if (i >> (idx + 1)) & 1:
                                clause.append(-nv)
                            else:
                                clause.append(nv)
                        clauses.append(clause)

        num_vars = rows * cols
        cnf_content = f"p cnf {num_vars} {len(clauses)}\n"
        for cl in clauses:
            cnf_content += " ".join(map(str, cl)) + " 0\n"
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as tmp_cnf:
            tmp_cnf.write(cnf_content)
            tmp_cnf_path = tmp_cnf.name
            
        try:
            res = subprocess.run([self.kissat_path, tmp_cnf_path], capture_output=True, text=True)
            
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
    def find_previous(self, grid: np.ndarray) -> Optional[np.ndarray]:
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

