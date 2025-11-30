from ABCPreviousFinder import PreviousFinder
from typing import Optional, List
import numpy as np
import pulp


class ILPMinPreviousFinder(PreviousFinder):
    def __init__(self, solver: Optional[pulp.LpSolver] = None):
        self.solver = solver

    @staticmethod
    def _gol_result(prev_center: int, neighbor_count: int) -> int:
        """Return Game of Life result (0/1) given prev_center and neighbor_count."""
        if prev_center == 1:
            return 1 if neighbor_count in (2, 3) else 0
        else:
            return 1 if neighbor_count == 3 else 0

    def find_previous(self, grid: np.ndarray, seed: Optional[int] = None,
                      exclude: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        """
        grid : 2D numpy array of {0,1} representing the *current* Game of Life state.
        Returns a 2D numpy array of {0,1} representing a minimal-alive previous state,
        or None if no previous state exists (MILP infeasible).
        """
        rows, cols = grid.shape
        N = rows * cols

        # Create MILP
        prob = pulp.LpProblem("GOL_previous_min_alive", pulp.LpMinimize)

        # Variables: x_{i,j} previous-state alive (binary)
        x = {
            (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
            for i in range(rows) for j in range(cols)
        }

        # s_var_{i,j} integer 0..8 representing neighbor count around previous state (does not include center)
        s_var = {
            (i, j): pulp.LpVariable(f"s_{i}_{j}", lowBound=0, upBound=8, cat="Integer")
            for i in range(rows) for j in range(cols)
        }

        # For each cell: allowed (p,s) pairs that produce observed current value
        # We'll create binary w_{i,j,p,s} only for allowed pairs, force sum w = 1,
        # link p and s via w, and link s_var to neighbor sums.
        w_vars = {}  # keys: (i,j,p,s) -> binary variable

        for i in range(rows):
            for j in range(cols):
                c = int(grid[i, j])
                allowed_pairs = []
                for p in (0, 1):
                    for s in range(9):
                        if self._gol_result(p, s) == c:
                            allowed_pairs.append((p, s))

                # We must have at least one allowed pair; if none, infeasible right away
                if len(allowed_pairs) == 0:
                    return None

                # Create w variables for each allowed pair and force exactly one to be selected
                w_list = []
                for (p, s) in allowed_pairs:
                    var = pulp.LpVariable(f"w_{i}_{j}_p{p}_s{s}", lowBound=0, upBound=1, cat="Binary")
                    w_vars[(i, j, p, s)] = var
                    w_list.append(var)

                # Exactly one (p,s) combination must be chosen for this cell
                prob += (pulp.lpSum(w_list) == 1), f"one_pair_{i}_{j}"

                # Link previous center p to w: p_var == sum_{s} w_{1,s}
                # But we already have x[i,j] as the previous center variable; enforce equality:
                # x_{i,j} == sum_{s} w_{1,s}
                prob += (x[(i, j)] == pulp.lpSum(
                    w_vars[(i, j, 1, s)] for (pp, s) in [(1, s) for (_, _, pp, s) in [(i, j, 1, s) for s in range(9)]] if (i, j, 1, s) in w_vars
                )), f"link_center_{i}_{j}"

                # Link s_var to the chosen s: s_var == sum_{(p,s)} s * w_{p,s}
                prob += (s_var[(i, j)] == pulp.lpSum(
                    s * w_vars[(i, j, p, s)] for (p, s) in [(p, s) for (p, s) in [(p, s) for p in (0, 1) for s in range(9)] if (i, j, p, s) in w_vars]
                )), f"link_sval_{i}_{j}"

        # Link s_var to neighbor sums: s_var_{i,j} == sum of x over neighbors (8-neighborhood, no wrapping)
        for i in range(rows):
            for j in range(cols):
                neighbor_sum = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbor_sum.append(x[(ni, nj)])
                # neighbor_sum may have length 0..8
                prob += (s_var[(i, j)] == pulp.lpSum(neighbor_sum)), f"neighbors_sum_{i}_{j}"

        # Exclude specific previous states if provided (optional)
        # --- Exclusion constraints ---
        if exclude:
            for k, E in enumerate(exclude):
                diff = []
                for i in range(rows):
                    for j in range(cols):
                        if int(E[i, j]) == 1:
                            diff.append(1 - x[(i, j)])
                        else:
                            diff.append(x[(i, j)])
                prob += pulp.lpSum(diff) >= 1

        # ------------------------------------------------------------------
        # 🎯 Objective with SEED support
        # ------------------------------------------------------------------

        # Primary objective: minimize alive cells
        alive_term = pulp.lpSum(x.values())

        # Secondary objective: deterministic randomized tie break
        if seed is not None:
            rng = np.random.default_rng(seed)
            noise = {(i, j): rng.uniform(0.0, 1e-3) for i in range(rows) for j in range(cols)}

            tie_break = pulp.lpSum(noise[(i, j)] * x[(i, j)]
                                   for i in range(rows) for j in range(cols))

            # Total objective = minimize alive count, then noise
            prob += alive_term + tie_break

        else:
            prob += alive_term

        # ------------------------------------------------------------------

        solver = self.solver or pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] not in ("Optimal", "Feasible"):
            return None

        # Read solution
        prev = np.zeros_like(grid, dtype=int)
        for i in range(rows):
            for j in range(cols):
                prev[i, j] = int(pulp.value(x[(i, j)]) > 0.5)

        return prev