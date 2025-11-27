import sys
import os

# Add root directory to sys.path to allow imports from py_z3 package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_z3.generate_small_alphabet import GenerateSmallText
from py_z3.classes import GameOfLifeGrid
from py_z3.minimizing_solvers import SATMinimizerSolver as SATSolver
from py_z3.minimizing_solvers import ILPMinimizerSolver as ILPSolver

import time

text_generator = GenerateSmallText()
grid = text_generator.text_to_grid("FMF", padding=5)
number_of_ones = grid.sum()
# Fix: Pass grid as keyword argument, otherwise it's interpreted as width
gol_grid = GameOfLifeGrid(grid=grid)
gol_grid.pretty_print()
print(number_of_ones)
solver = SATSolver()
ILPS = ILPSolver()

number_of_steps = 4

for i in range(number_of_steps):
    start = time.time()
    prev = ILPS.find_previous(grid, optimize=True)
    end = time.time()
    gol_grid = GameOfLifeGrid(grid=prev)
    gol_grid.pretty_print()
    number_of_ones = prev.sum()
    print(number_of_ones)
    print("Time taken:", end - start)
    grid = prev

gol = GameOfLifeGrid(grid=grid)
for i in range(number_of_steps):
    gol.pretty_print()
    gol.compute_next()

"""
a = 1
b = 500

# bijection 
while(a <= b):
    m = int((b + a) / 2)
    print("Checking max_alive =", m, "with a =", a, "and b =", b)
    start = time.time()
    prev = solver.find_previous(grid, max_alive=m)
    end = time.time()
    if prev is not None:
        print("Found previous with max_alive =", m)
        print(prev)
        b = m - 1
    else:
        print("No previous found with max_alive =", m)
        a = m + 1
    print("Time taken:", end - start)
    
    
    
"""