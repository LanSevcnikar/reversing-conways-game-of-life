#!/usr/bin/env python3
"""Quick test of the visualizer with a simple pattern."""

import numpy as np
from py_z3.classes import GameOfLifeGrid, GameOfLifeVisualizer

# Create a simple blinker pattern
grid = np.zeros((10, 10), dtype=int)
grid[1, 2] = 1
grid[2, 3] = 1
grid[3, 1:4] = 1  # This creates the bottom row of the glider

print("Testing GameOfLifeVisualizer")
print("Initial grid:")
print(grid)
print("\nThis will show a blinker pattern oscillating.")
print("The animation will loop with a 2-second pause between loops.")
print("Close the window when done.\n")

# Create game and visualizer
game = GameOfLifeGrid(grid=grid)
visualizer = GameOfLifeVisualizer(
    game, 
    interval=100,      # 500ms between frames
    loop_pause=1000    # 2 second pause when loop repeats
)

# Visualize 6 steps (enough to see the oscillation)
visualizer.visualize(steps=20, title="Blinker Pattern - Test")
