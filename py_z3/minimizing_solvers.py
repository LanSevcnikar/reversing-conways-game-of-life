import numpy as np
import os
import subprocess
import tempfile
from typing import Optional, List, Tuple
from z3 import Solver, Int, Sum, If, Or, And, sat, Not, is_true, set_param, Bool
from scipy.optimize import milp, LinearConstraint, Bounds
from solvers import Z3PredecessorFinder, SATPredecessorFinder, ILPPredecessorFinder, z3_not, z3_is_true

class Z3MinimizerSolver(Z3PredecessorFinder):
    def find_previous(self, grid: np.ndarray, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None, max_alive: Optional[int] = None) -> Optional[np.ndarray]:
        # Reuse parent logic but add max_alive constraint
        # Since parent logic is inside a method, we can't easily inject constraints without copy-paste or refactoring parent to be more modular.
        # Given the request to "implement all 3 again but this time with also an objective", I will copy-paste and modify.
        # Alternatively, I can copy the parent code here.
        
        rows, cols = grid.shape
        s = Solver()
        
        if seed is not None:
            set_param('smt.random_seed', seed)
            set_param('sat.random_seed', seed)
        
        vars_grid = [[Bool(f"c_{r}_{c}") for c in range(cols)] for r in range(rows)]
        
        def get_neighs(r, c):
            ns = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = (r + dr) % rows, (c + dc) % cols
                    ns.append(vars_grid[nr][nc])
            return ns

        for r in range(rows):
            for c in range(cols):
                v = vars_grid[r][c]
                neighs = get_neighs(r, c)
                s_sum = Sum([If(n, 1, 0) for n in neighs])
                is_alive_next = Or(s_sum == 3, And(v, s_sum == 2))
                
                if grid[r, c] == 1:
                    s.add(is_alive_next)
                else:
                    s.add(z3_not(is_alive_next))
                    
        if exclude:
            for ex_grid in exclude:
                differs = []
                for r in range(rows):
                    for c in range(cols):
                        if ex_grid[r, c] == 1:
                            differs.append(z3_not(vars_grid[r][c]))
                        else:
                            differs.append(vars_grid[r][c])
                s.add(Or(differs))

        # Max alive constraint
        if max_alive is not None:
            all_cells = []
            for r in range(rows):
                for c in range(cols):
                    all_cells.append(If(vars_grid[r][c], 1, 0))
            s.add(Sum(all_cells) <= max_alive)

        if s.check() == sat:
            m = s.model()
            res = np.zeros((rows, cols), dtype=int)
            for r in range(rows):
                for c in range(cols):
                    if z3_is_true(m[vars_grid[r][c]]):
                        res[r, c] = 1
            return res
        return None

    def find_n_previous(self, grid: np.ndarray, n: int, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None, minimize: bool = False) -> List[np.ndarray]:
        if not minimize:
            return super().find_n_previous(grid, n, seed, exclude)
        
        # Minimization logic (Binary Search)
        rows, cols = grid.shape
        total_cells = rows * cols
        
        if self.find_previous(grid, seed=seed) is None:
            return []
            
        low = 0
        high = total_cells
        min_k = high
        found_any = False
        
        while low <= high:
            mid = (low + high) // 2
            sol = self.find_previous(grid, seed=seed, max_alive=mid)
            if sol is not None:
                min_k = mid
                high = mid - 1
                found_any = True
            else:
                low = mid + 1
        
        if not found_any:
            return []
            
        target_k = min_k + 2
        
        found_solutions = []
        current_exclude = list(exclude) if exclude else []
        
        for i in range(n):
            current_seed = (seed + i) if seed is not None else i
            sol = self.find_previous(grid, seed=current_seed, exclude=current_exclude, max_alive=target_k)
            if sol is not None:
                found_solutions.append(sol)
                current_exclude.append(sol)
            else:
                break
        return found_solutions


class SATMinimizerSolver(SATPredecessorFinder):
    def _add_cardinality_constraint(self, clauses: List[List[int]], vars_list: List[int], k: int, num_vars_start: int) -> int:
        n = len(vars_list)
        if k >= n: return num_vars_start
        if k < 0:
            clauses.append([0])
            return num_vars_start

        s_vars = {}
        current_var = num_vars_start

        def get_s(i, j):
            if (i, j) not in s_vars:
                nonlocal current_var
                current_var += 1
                s_vars[(i, j)] = current_var
            return s_vars[(i, j)]

        # Sequential Counter Encoding
        if k >= 1:
            clauses.append([-vars_list[0], get_s(1, 1)])
            
        for i in range(1, n):
            x_val = vars_list[i]
            # -s_{i-1, j} \/ s_{i, j}
            for j in range(1, k + 1):
                if i > 1:
                    clauses.append([-get_s(i - 1, j), get_s(i, j)])
            
            # -x_i \/ s_{i, 1}
            if k >= 1:
                clauses.append([-x_val, get_s(i, 1)])
            
            # -x_i \/ -s_{i-1, j-1} \/ s_{i, j}
            for j in range(2, k + 1):
                if i > 1:
                    clauses.append([-x_val, -get_s(i - 1, j - 1), get_s(i, j)])
            
            # Restriction: sum <= k => forbid s_{i-1, k} if x_i is true
            if i > 1:
                clauses.append([-x_val, -get_s(i - 1, k)])
        
        if k == 0:
            for x in vars_list:
                clauses.append([-x])
                
        return current_var

    def find_previous(self, grid: np.ndarray, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None, max_alive: Optional[int] = None) -> Optional[np.ndarray]:
        if max_alive is None:
            return super().find_previous(grid, seed, exclude)
            
        rows, cols = grid.shape
        def var_idx(r, c): return r * cols + c + 1
        
        clauses = []
        
        # Reuse template generation logic
        if not hasattr(self, 'tmpl_0'):
            self._cache_clauses()
            self.tmpl_0 = []
            self.tmpl_1 = []
            for i in range(512):
                current_alive = (i >> 0) & 1
                neigh_alive_count = sum((i >> j) & 1 for j in range(1, 9))
                next_alive = 1 if (current_alive and neigh_alive_count in [2,3]) or (not current_alive and neigh_alive_count==3) else 0
                
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
                    
        if exclude:
            for ex_grid in exclude:
                clause = []
                for r in range(rows):
                    for c in range(cols):
                        idx = var_idx(r, c)
                        if ex_grid[r, c] == 1:
                            clause.append(-idx)
                        else:
                            clause.append(idx)
                clauses.append(clause)

        all_vars = [var_idx(r, c) for r in range(rows) for c in range(cols)]
        num_vars = rows * cols
        
        num_vars = self._add_cardinality_constraint(clauses, all_vars, max_alive, num_vars)
        
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

    def find_n_previous(self, grid: np.ndarray, n: int, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None, minimize: bool = False) -> List[np.ndarray]:
        if not minimize:
            return super().find_n_previous(grid, n, seed, exclude)
            
        rows, cols = grid.shape
        total_cells = rows * cols
        
        if self.find_previous(grid, seed=seed) is None:
            return []
            
        low = 0
        high = total_cells
        min_k = high
        found_any = False
        
        while low <= high:
            mid = (low + high) // 2
            sol = self.find_previous(grid, seed=seed, max_alive=mid)
            if sol is not None:
                min_k = mid
                high = mid - 1
                found_any = True
            else:
                low = mid + 1
                
        if not found_any:
            return []
            
        target_k = min_k + 2
        
        found_solutions = []
        current_exclude = list(exclude) if exclude else []
        
        for i in range(n):
            current_seed = (seed + i) if seed is not None else i
            sol = self.find_previous(grid, seed=current_seed, exclude=current_exclude, max_alive=target_k)
            if sol is not None:
                found_solutions.append(sol)
                current_exclude.append(sol)
            else:
                break
        return found_solutions


class ILPMinimizerSolver(ILPPredecessorFinder):
    def find_previous(self, grid: np.ndarray, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None, max_alive: Optional[int] = None, optimize: bool = True) -> Optional[np.ndarray]:
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
        if optimize:
            for r in range(rows):
                for c in range(cols):
                    c_vec[x_map[(r,c)]] = 1.0
        
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
                    coeffs = {ni: 1 for ni in neighs}
                    add_constr(coeffs, 2, 3)
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[x_map[(r,c)]] = 1
                    add_constr(coeffs, 3, M + 1)
                else:
                    u = u_map[(r,c)][0]
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[u] = -M
                    add_constr(coeffs, -np.inf, 2)
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[u] = -M
                    add_constr(coeffs, 4 - M, np.inf)
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[x_map[(r,c)]] = 1
                    coeffs[u] = -M
                    add_constr(coeffs, -np.inf, 2)

        if exclude:
            for ex_grid in exclude:
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

        if max_alive is not None:
            coeffs = {}
            for r in range(rows):
                for c in range(cols):
                    coeffs[x_map[(r,c)]] = 1
            add_constr(coeffs, -np.inf, max_alive)

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

    def find_n_previous(self, grid: np.ndarray, n: int, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None, minimize: bool = False) -> List[np.ndarray]:
        if not minimize:
            return super().find_n_previous(grid, n, seed, exclude)
            
        found_solutions = []
        current_exclude = list(exclude) if exclude else []
        target_max_alive = None
        
        min_sol = self.find_previous(grid, seed=seed, exclude=current_exclude, optimize=True)
        
        if min_sol is None:
            return []
            
        min_alive = np.sum(min_sol)
        target_max_alive = min_alive + 2
        
        found_solutions.append(min_sol)
        current_exclude.append(min_sol)
        
        for _ in range(n - 1):
            sol = self.find_previous(grid, seed=seed, exclude=current_exclude, max_alive=target_max_alive)
            if sol is not None:
                found_solutions.append(sol)
                current_exclude.append(sol)
            else:
                break
        return found_solutions


import numpy as np
import os
import subprocess
import tempfile
from typing import Optional, List, Tuple
from z3 import Solver, Int, Sum, If, Or, And, sat, Not, is_true, set_param, Bool
from scipy.optimize import milp, LinearConstraint, Bounds
from solvers import Z3PredecessorFinder, SATPredecessorFinder, ILPPredecessorFinder, z3_not, z3_is_true

class Z3MinimizerSolver(Z3PredecessorFinder):
    def find_previous(self, grid: np.ndarray, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None, max_alive: Optional[int] = None) -> Optional[np.ndarray]:
        # Reuse parent logic but add max_alive constraint
        # Since parent logic is inside a method, we can't easily inject constraints without copy-paste or refactoring parent to be more modular.
        # Given the request to "implement all 3 again but this time with also an objective", I will copy-paste and modify.
        # Alternatively, I can copy the parent code here.
        
        rows, cols = grid.shape
        s = Solver()
        
        if seed is not None:
            set_param('smt.random_seed', seed)
            set_param('sat.random_seed', seed)
        
        vars_grid = [[Bool(f"c_{r}_{c}") for c in range(cols)] for r in range(rows)]
        
        def get_neighs(r, c):
            ns = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = (r + dr) % rows, (c + dc) % cols
                    ns.append(vars_grid[nr][nc])
            return ns

        for r in range(rows):
            for c in range(cols):
                v = vars_grid[r][c]
                neighs = get_neighs(r, c)
                s_sum = Sum([If(n, 1, 0) for n in neighs])
                is_alive_next = Or(s_sum == 3, And(v, s_sum == 2))
                
                if grid[r, c] == 1:
                    s.add(is_alive_next)
                else:
                    s.add(z3_not(is_alive_next))
                    
        if exclude:
            for ex_grid in exclude:
                differs = []
                for r in range(rows):
                    for c in range(cols):
                        if ex_grid[r, c] == 1:
                            differs.append(z3_not(vars_grid[r][c]))
                        else:
                            differs.append(vars_grid[r][c])
                s.add(Or(differs))

        # Max alive constraint
        if max_alive is not None:
            all_cells = []
            for r in range(rows):
                for c in range(cols):
                    all_cells.append(If(vars_grid[r][c], 1, 0))
            s.add(Sum(all_cells) <= max_alive)

        if s.check() == sat:
            m = s.model()
            res = np.zeros((rows, cols), dtype=int)
            for r in range(rows):
                for c in range(cols):
                    if z3_is_true(m[vars_grid[r][c]]):
                        res[r, c] = 1
            return res
        return None

    def find_n_previous(self, grid: np.ndarray, n: int, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None, minimize: bool = False) -> List[np.ndarray]:
        if not minimize:
            return super().find_n_previous(grid, n, seed, exclude)
        
        # Minimization logic (Binary Search)
        rows, cols = grid.shape
        total_cells = rows * cols
        
        if self.find_previous(grid, seed=seed) is None:
            return []
            
        low = 0
        high = total_cells
        min_k = high
        found_any = False
        
        while low <= high:
            mid = (low + high) // 2
            sol = self.find_previous(grid, seed=seed, max_alive=mid)
            if sol is not None:
                min_k = mid
                high = mid - 1
                found_any = True
            else:
                low = mid + 1
        
        if not found_any:
            return []
            
        target_k = min_k + 2
        
        found_solutions = []
        current_exclude = list(exclude) if exclude else []
        
        for i in range(n):
            current_seed = (seed + i) if seed is not None else i
            sol = self.find_previous(grid, seed=current_seed, exclude=current_exclude, max_alive=target_k)
            if sol is not None:
                found_solutions.append(sol)
                current_exclude.append(sol)
            else:
                break
        return found_solutions

class ILPMinimizerSolver(ILPPredecessorFinder):
    def find_previous(self, grid: np.ndarray, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None, max_alive: Optional[int] = None, optimize: bool = False) -> Optional[np.ndarray]:
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
                    var_count += 1
                    dead_cells.append((r,c))
                    
        c_vec = np.zeros(var_count)
        if optimize:
            for r in range(rows):
                for c in range(cols):
                    c_vec[x_map[(r,c)]] = 1.0
        
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
                    coeffs = {ni: 1 for ni in neighs}
                    add_constr(coeffs, 2, 3)
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[x_map[(r,c)]] = 1
                    add_constr(coeffs, 3, M + 1)
                else:
                    u = u_map[(r,c)][0]
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[u] = -M
                    add_constr(coeffs, -np.inf, 2)
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[u] = -M
                    add_constr(coeffs, 4 - M, np.inf)
                    coeffs = {ni: 1 for ni in neighs}
                    coeffs[x_map[(r,c)]] = 1
                    coeffs[u] = -M
                    add_constr(coeffs, -np.inf, 2)

        if exclude:
            for ex_grid in exclude:
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

        if max_alive is not None:
            coeffs = {}
            for r in range(rows):
                for c in range(cols):
                    coeffs[x_map[(r,c)]] = 1
            add_constr(coeffs, -np.inf, max_alive)

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

    def find_n_previous(self, grid: np.ndarray, n: int, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None, minimize: bool = False) -> List[np.ndarray]:
        if not minimize:
            return super().find_n_previous(grid, n, seed, exclude)
            
        found_solutions = []
        current_exclude = list(exclude) if exclude else []
        target_max_alive = None
        
        min_sol = self.find_previous(grid, seed=seed, exclude=current_exclude, optimize=True)
        
        if min_sol is None:
            return []
            
        min_alive = np.sum(min_sol)
        target_max_alive = min_alive + 2
        
        found_solutions.append(min_sol)
        current_exclude.append(min_sol)
        
        for _ in range(n - 1):
            sol = self.find_previous(grid, seed=seed, exclude=current_exclude, max_alive=target_max_alive)
            if sol is not None:
                found_solutions.append(sol)
                current_exclude.append(sol)
            else:
                break
        return found_solutions