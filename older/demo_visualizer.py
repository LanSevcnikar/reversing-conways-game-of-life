#!/usr/bin/env python3
"""
Demo script for GameOfLifeVisualizer.

This script demonstrates how to use the visualizer to animate
Conway's Game of Life patterns.
"""

import numpy as np
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from py_z3.classes import GameOfLifeGrid, GameOfLifeVisualizer
from generate_alphabet import GenerateText
from solvers import SATPredecessorFinder


def demo_simple_pattern():
    """Demo with a simple oscillator pattern (blinker)."""
    print("Demo 1: Blinker Pattern")
    print("=" * 50)
    
    # Create a blinker pattern (3 horizontal cells)
    grid = np.zeros((10, 10), dtype=int)
    grid[4, 4:7] = 1  # Horizontal line
    
    print("Initial grid:")
    print(grid)
    
    # Create game and visualizer
    game = GameOfLifeGrid(grid=grid)
    visualizer = GameOfLifeVisualizer(game)
    
    # Visualize 10 steps
    print("\nVisualizing 10 steps (close window to continue)...")
    visualizer.visualize(steps=10, title="Blinker Pattern (Click Next/Prev)")


def demo_glider():
    """Demo with a glider pattern."""
    print("\nDemo 2: Glider Pattern")
    print("=" * 50)
    
    # Create a glider
    grid = np.zeros((20, 20), dtype=int)
    glider = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    grid[5:8, 5:8] = glider
    
    print("Initial grid with glider:")
    print(grid)
    
    # Create game and visualizer
    game = GameOfLifeGrid(grid=grid)
    visualizer = GameOfLifeVisualizer(game)
    
    # Visualize 20 steps
    print("\nVisualizing 20 steps (close window to continue)...")
    visualizer.visualize(steps=20, title="Glider Pattern")


def demo_alphabet_letter():
    """Demo with an alphabet letter."""
    print("\nDemo 3: Alphabet Letter 'A'")
    print("=" * 50)
    
    # Generate letter A
    gen = GenerateText()
    grid = gen.text_to_grid('A')
    
    print(f"Grid size: {grid.shape}")
    print("Initial grid:")
    print(grid)
    
    # Create game and visualizer
    game = GameOfLifeGrid(grid=grid)
    visualizer = GameOfLifeVisualizer(game)
    
    # Visualize 15 steps
    print("\nVisualizing 15 steps (close window to continue)...")
    visualizer.visualize(steps=15, title="Letter 'A' Evolution")


def demo_from_sequence():
    """Demo visualizing a pre-computed sequence (e.g., from a solver)."""
    print("\nDemo 4: Pre-computed Sequence")
    print("=" * 50)
    
    # Create a simple pattern
    grid1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    # Manually create a sequence (in practice, this could come from a solver)
    sequence = []
    game = GameOfLifeGrid(grid=grid1)
    sequence.append(game.grid.copy())
    
    for _ in range(5):
        game.compute_next()
        game.advance()
        sequence.append(game.grid.copy())
    
    print(f"Created sequence of {len(sequence)} grids")
    
    # Visualize the sequence
    # Note: we still need a GameOfLifeGrid instance, but it won't be used
    dummy_game = GameOfLifeGrid(grid=grid1)
    visualizer = GameOfLifeVisualizer(dummy_game)
    
    print("\nVisualizing pre-computed sequence (close window to continue)...")
    visualizer.visualize_from_sequence(sequence, title="Pre-computed Sequence")


def demo_interactive_solver():
    """Demo with interactive solver (finding predecessors)."""
    print("\nDemo 5: Interactive Solver (Reverse Game of Life)")
    print("=" * 50)
    
    # Generate letter A
    gen = GenerateText()
    grid = gen.text_to_grid('A')
    
    print("Initial grid (Letter A). You can click 'Previous' to find predecessors!")
    
    # Create solver
    print("Initializing SAT Solver...")
    solver = SATPredecessorFinder()
    
    # Create game and visualizer with solver
    game = GameOfLifeGrid(grid=grid)
    visualizer = GameOfLifeVisualizer(game, solver=solver)
    
    print("\nVisualizing... Click 'Previous' to go back in time!")
    visualizer.visualize(steps=0, title="Interactive Solver - Click Previous!")


def demo_save_animation():
    """Demo saving an animation to a file."""
    print("\nDemo 6: Save Animation to GIF")
    print("=" * 50)
    
    # Create a glider
    grid = np.zeros((15, 15), dtype=int)
    glider = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    grid[5:8, 5:8] = glider
    
    game = GameOfLifeGrid(grid=grid)
    visualizer = GameOfLifeVisualizer(game)
    
    output_file = os.path.join(os.getcwd(), "glider_animation.gif")
    
    print(f"Saving animation to {output_file}...")
    try:
        visualizer.save_animation(output_file, steps=15, fps=2, dpi=80)
        print("✓ Animation saved successfully!")
    except Exception as e:
        print(f"✗ Failed to save animation: {e}")
        print("  (This requires pillow to be installed: pip install pillow)")


if __name__ == "__main__":
    print("GameOfLifeVisualizer Demo")
    print("=" * 50)
    print("\nThis demo will show several visualization examples.")
    print("Close each window to proceed to the next demo.\n")
    
    # Run demos
    demo_simple_pattern()
    demo_glider()
    demo_alphabet_letter()
    demo_from_sequence()
    demo_interactive_solver()
    demo_save_animation()
    
    print("\n" + "=" * 50)
    print("All demos completed!")
    print("=" * 50)
