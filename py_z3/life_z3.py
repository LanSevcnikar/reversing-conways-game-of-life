"""
life_smt.py

Find preimages (possible previous generations) for a given Game of Life board
using Z3 (SMT). Uses integer variables 0/1 for cells and constrains each cell
according to the Life rule using sums of neighbors.

Requirements:
    pip install z3-solver

Functions:
    find_preimages(next_grid, max_solutions=10, wrap=False) -> list of 2D lists (0/1)
Example usage in __main__ demonstrates a blinker and a random small board.
"""

from z3 import Solver, Int, Sum, If, Or, And, sat
import random
import itertools


def _mk_vars(rows, cols, name="c"):
    """Create a rows x cols matrix of Int Z3 variables (0 or 1)."""
    return [[Int(f"{name}_{r}_{c}") for c in range(cols)] for r in range(rows)]


def _neigh_coords(r, c, rows, cols, wrap):
    """Yield neighbor coordinates (8 neighbors) for cell (r,c)."""
    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dr, dc in deltas:
        rr, cc = r + dr, c + dc
        if wrap:
            rr %= rows
            cc %= cols
            yield rr, cc
        else:
            if 0 <= rr < rows and 0 <= cc < cols:
                yield rr, cc
            # else: outside -> considered 0 (we handle by not including)


def _life_next_expr(cell_var, neighbor_vars):
    """
    Return an Int expression that evaluates to 1 if the given cell_var and
    neighbor_vars produce a live cell in the next generation, else 0.
    Life rules:
      - If cell==1 and (sum == 2 or sum == 3) => next = 1
      - If cell==0 and (sum == 3) => next = 1
      - otherwise next = 0
    """
    s = Sum(*neighbor_vars)  # sum of neighbor Ints
    stay_alive = And(cell_var == 1, Or(s == 2, s == 3))
    become_alive = And(cell_var == 0, s == 3)
    return If(Or(stay_alive, become_alive), 1, 0)


def find_preimages(next_grid, max_solutions=10, wrap=False):
    """
    next_grid: list of lists of 0/1 ints (rows x cols) representing the known board at t+1.
    max_solutions: maximum number of preimages to return (use None for "all" but beware explosion).
    wrap: if True, uses toroidal wrapping; if False, outside cells are considered dead (0).
    Returns: list of preimage grids (each is rows x cols list of 0/1)
    """
    rows = len(next_grid)
    cols = len(next_grid[0]) if rows > 0 else 0

    # Create variables for t (unknown)
    vars_grid = _mk_vars(rows, cols, name="x")

    s = Solver()

    # Constrain all vars to be 0 or 1
    for r in range(rows):
        for c in range(cols):
            v = vars_grid[r][c]
            s.add(v >= 0, v <= 1)

    # For each cell, build the next-expression from current variables and assert it equals next_grid[r][c]
    for r in range(rows):
        for c in range(cols):
            v = vars_grid[r][c]
            neighs = []
            # If non-wrapping, neighbors outside bounds are simply absent and thus treated as 0.
            for nr, nc in _neigh_coords(r, c, rows, cols, wrap):
                neighs.append(vars_grid[nr][nc])
            # For cells near edges with no wrapping, neighbors list might be shorter; Sum handles that.
            next_expr = _life_next_expr(v, neighs)
            s.add(next_expr == (1 if next_grid[r][c] else 0))

    solutions = []
    found = 0
    while s.check() == sat:
        m = s.model()
        # Extract model
        sol = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                val = m[vars_grid[r][c]].as_long()
                sol[r][c] = 1 if val != 0 else 0
        solutions.append(sol)
        found += 1
        if max_solutions is not None and found >= max_solutions:
            break
        # Block the current model to find a different one next iteration
        s.add(
            Or(
                *[
                    vars_grid[r][c] != sol[r][c]
                    for r in range(rows)
                    for c in range(cols)
                ]
            )
        )
    return solutions


def print_grid(g):
    for row in g:
        print("".join("#" if v else "." for v in row))


def time_function(f, *args, **kwargs):
    import time

    start = time.time()
    res = f(*args, **kwargs)
    end = time.time()
    return res, end - start


if __name__ == "__main__":
    # Example 1: Blinker (period-2 oscillator)
    # A blinker horizontal (t+1):
    # . . . . .
    # . . . . .
    # . # # # .
    # . . . . .
    # . . . . .
    blinker_next = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    FMF = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    print("Finding preimages for blinker (5x5), non-wrapping:")

    pres, time = time_function(find_preimages, FMF, max_solutions=100, wrap=False)
    print(f"Solved in {time:.4f} seconds.")
    print(f"Found {len(pres)} preimage(s). Showing up to 5:")
    for i, p in enumerate(pres[:5], 1):
        print(f"\nPreimage #{i}:")
        print_grid(p)

    # Example 2: Random small board
    print("\nRandom 4x4 example (wrap=True):")
    rnd_next = [[random.choice([0, 1]) for _ in range(4)] for __ in range(4)]
    print("Next generation (target):")
    print_grid(rnd_next)
    pres2 = find_preimages(rnd_next, max_solutions=5, wrap=True)
    print(f"Found {len(pres2)} preimage(s) (wrap=True).")
    for i, p in enumerate(pres2, 1):
        print(f"\nPreimage #{i}:")
        print_grid(p)
