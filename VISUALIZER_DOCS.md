# GameOfLifeVisualizer Documentation

## Overview

The `GameOfLifeVisualizer` class provides a visual animation tool for Conway's Game of Life using matplotlib. It can display looping animations with customizable timing and supports both forward evolution and visualization of pre-computed sequences.

## Features

- **Visual Animation**: Uses matplotlib to create smooth, visual animations
- **Looping**: Automatically loops through the sequence with a configurable pause between loops
- **Customizable Timing**: Control frame rate and loop pause duration
- **Pre-computed Sequences**: Can visualize sequences from solvers (e.g., backtracking results)
- **Save to File**: Export animations as GIF or MP4 files

## Basic Usage

### 1. Simple Evolution Visualization

```python
import numpy as np
from py_z3.classes import GameOfLifeGrid, GameOfLifeVisualizer

# Create a grid with a pattern
grid = np.zeros((10, 10), dtype=int)
grid[4, 4:7] = 1  # Horizontal line (blinker)

# Create game and visualizer
game = GameOfLifeGrid(grid=grid)
visualizer = GameOfLifeVisualizer(
    game, 
    interval=500,      # 500ms between frames
    loop_pause=2000    # 2 second pause when loop repeats
)

# Visualize 10 steps
visualizer.visualize(steps=10, title="My Pattern")
```

### 2. Visualize Pre-computed Sequence

This is useful for visualizing results from backtracking solvers:

```python
from py_z3.classes import GameOfLifeGrid, GameOfLifeVisualizer
from higher_order_solvers import NaiveDFSSolver
from solvers import SATPredecessorFinder

# Get a sequence from a solver
target_grid = ...  # your target grid
finder = SATPredecessorFinder()
solver = NaiveDFSSolver(finder)
sequence = solver.solve(target_grid, steps=5)

if sequence:
    # Visualize the sequence
    dummy_game = GameOfLifeGrid(grid=sequence[0])
    visualizer = GameOfLifeVisualizer(dummy_game, interval=400, loop_pause=2500)
    visualizer.visualize_from_sequence(sequence, title="Backtracking Solution")
```

### 3. Save Animation to File

```python
from py_z3.classes import GameOfLifeGrid, GameOfLifeVisualizer

game = GameOfLifeGrid(grid=my_grid)
visualizer = GameOfLifeVisualizer(game)

# Save as GIF
visualizer.save_animation("output.gif", steps=20, fps=2, dpi=100)

# Save as MP4 (requires ffmpeg)
visualizer.save_animation("output.mp4", steps=20, fps=5, dpi=100)
```

## Constructor Parameters

```python
GameOfLifeVisualizer(game_grid, interval=200, loop_pause=2000)
```

- **game_grid**: `GameOfLifeGrid` instance to visualize
- **interval**: Time between frames in milliseconds (default: 200ms)
- **loop_pause**: Additional pause when loop repeats in milliseconds (default: 2000ms)

## Methods

### `visualize(steps, title="Conway's Game of Life")`

Evolve the game forward for `steps` generations and display the animation.

**Parameters:**
- `steps` (int): Number of steps to evolve
- `title` (str): Title for the visualization window

### `visualize_from_sequence(grids, title="Conway's Game of Life")`

Visualize a pre-computed sequence of grids.

**Parameters:**
- `grids` (List[np.ndarray]): List of grid states to visualize
- `title` (str): Title for the visualization window

### `save_animation(filename, steps, fps=5, dpi=100)`

Save the animation to a file.

**Parameters:**
- `filename` (str): Output filename (e.g., 'game.gif' or 'game.mp4')
- `steps` (int): Number of steps to include
- `fps` (int): Frames per second (default: 5)
- `dpi` (int): Dots per inch for output quality (default: 100)

**Requirements:**
- For GIF: `pillow` package (`pip install pillow`)
- For MP4: `ffmpeg` installed on system

## Examples

### Example 1: Glider Pattern

```python
import numpy as np
from py_z3.classes import GameOfLifeGrid, GameOfLifeVisualizer

# Create a glider
grid = np.zeros((20, 20), dtype=int)
glider = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
])
grid[5:8, 5:8] = glider

game = GameOfLifeGrid(grid=grid)
visualizer = GameOfLifeVisualizer(game, interval=300, loop_pause=1500)
visualizer.visualize(steps=20, title="Glider Pattern")
```

### Example 2: Alphabet Letter

```python
from py_z3.classes import GameOfLifeGrid, GameOfLifeVisualizer
from generate_alphabet import GenerateText

gen = GenerateText()
grid = gen.text_to_grid('A')

game = GameOfLifeGrid(grid=grid)
visualizer = GameOfLifeVisualizer(game, interval=400, loop_pause=2500)
visualizer.visualize(steps=15, title="Letter 'A' Evolution")
```

### Example 3: Visualize Backtracking Solution

```python
from py_z3.classes import GameOfLifeGrid, GameOfLifeVisualizer
from higher_order_solvers import NaiveDFSSolver
from solvers import SATPredecessorFinder
from generate_alphabet import GenerateText

# Get target grid
gen = GenerateText()
target = gen.text_to_grid('B')

# Find predecessors
finder = SATPredecessorFinder()
solver = NaiveDFSSolver(finder, verbose=False)
sequence = solver.solve(target, steps=3)

if sequence:
    # Visualize the backtracking solution
    dummy_game = GameOfLifeGrid(grid=sequence[0])
    visualizer = GameOfLifeVisualizer(dummy_game, interval=600, loop_pause=3000)
    visualizer.visualize_from_sequence(
        sequence, 
        title="Backtracking Solution: 3 Steps to 'B'"
    )
else:
    print("No solution found")
```

## Tips

1. **Adjust timing**: Use shorter `interval` for faster animations, longer for easier viewing
2. **Loop pause**: Set `loop_pause` longer to clearly see when the animation restarts
3. **Performance**: For large grids or many steps, the animation may be slower
4. **Closing**: Close the matplotlib window to end the visualization
5. **Interactive**: The matplotlib window is interactive - you can zoom, pan, etc.

## Demo Scripts

Run the included demo scripts to see examples:

```bash
# Simple quick test
python test_visualizer_simple.py

# Comprehensive demos
python demo_visualizer.py
```
