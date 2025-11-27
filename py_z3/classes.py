import time
from typing import List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class GameOfLifeGrid:
    def __init__(self, width=None, length=None, grid=None):
        if grid is not None:
            self.grid = grid
            self.length, self.width = grid.shape
        else:
            self.width = width
            self.length = length
            self.grid = np.random.randint(2, size=(length, width))
        self.next_grid = None

    def compute_next(self):
        g = self.grid
        n = sum(
            np.roll(np.roll(g, i, 0), j, 1)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            if (i, j) != (0, 0)
        )
        self.next_grid = ((n == 3) | ((g == 1) & (n == 2))).astype(int)

    def advance(self):
        if self.next_grid is None:
            self.compute_next()
        self.grid = self.next_grid
        self.next_grid = None

    def is_same(self, other) -> bool:
        return np.array_equal(self.grid, other.grid)

    def is_stable(self) -> bool:
        if self.next_grid is None:
            self.compute_next()
        return np.array_equal(self.grid, self.next_grid)

    def is_same_as_max(self, other) -> bool:
        """
        Check if this grid's max (stable) form equals another grid,
        without storing the intermediate grids in memory.
        """
        current = self.grid.copy()
        next_ = None
        while True:
            n = sum(
                np.roll(np.roll(current, i, 0), j, 1)
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                if (i, j) != (0, 0)
            )
            next_ = ((n == 3) | ((current == 1) & (n == 2))).astype(int)
            if np.array_equal(current, next_):
                break
            current = next_
        return np.array_equal(current, other.grid)

    def pretty_print(self):
        """
        Print the grid in a pretty format using unicode block characters.
        """
        # Top border
        print("┌" + "─" * (self.width * 2) + "┐")
        
        for row in self.grid:
            print("│", end="")
            for cell in row:
                if cell == 1:
                    print("██", end="")
                else:
                    print("  ", end="")
            print("│")
            
        # Bottom border
        print("└" + "─" * (self.width * 2) + "┘")


class PredecessorFinder(ABC):
    """
    Abstract base class for finding the previous (parent) grid
    that could have evolved into a given Game of Life grid.
    """

    @abstractmethod
    def find_previous(self, grid: np.ndarray, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        """
        Given a grid (numpy 2D array), find a previous grid that could have produced it.
        
        Args:
            grid: The target grid.
            seed: Optional integer seed for randomization.
            exclude: Optional list of grids to exclude from the search.
            
        Returns:
            A numpy array of the same shape, or None if no predecessor found.
        """
        pass

    def find_n_previous(self, grid: np.ndarray, n: int, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None, minimize: bool = False) -> List[np.ndarray]:
        """
        Given a grid (numpy 2D array), attempt to find the n previous grids
        that could have produced it according to the Game of Life rules.
        
        Args:
            grid: The target grid.
            n: Number of previous grids to find.
            seed: Optional integer seed for randomization.
            exclude: Optional list of grids to exclude from the search.
            minimize: Whether to try to minimize the number of alive cells (default False).
            
        Returns:
            A list of numpy arrays of the same shape.
        """
        found_solutions = []
        current_exclude = list(exclude) if exclude else []
        
        for i in range(n):
            # If minimize is True, subclasses should handle it or this default implementation 
            # will just find *any* solution.
            # Ideally, minimizing solvers override this method.
            current_seed = (seed + i) if seed is not None else None
            sol = self.find_previous(grid, seed=current_seed, exclude=current_exclude)
            if sol is not None:
                found_solutions.append(sol)
                current_exclude.append(sol)
            else:
                break
        return found_solutions


class HigherOrderSolver(ABC):
    """
    Abstract base class for finding ancestors N steps back in time.
    """
    
    def __init__(self, finder: PredecessorFinder):
        self.finder = finder

    @abstractmethod
    def solve(self, target_grid: np.ndarray, steps: int) -> List[np.ndarray]:
        """
        Find a sequence of grids [g_0, g_1, ..., g_steps] such that:
        g_steps == target_grid
        g_{i+1} is the next generation of g_i
        
        Returns the list of grids in chronological order (ancestor first).
        Returns None (or raises exception?) if not found.
        Let's say it returns None if it fails.
        """
        pass


class TimeEvaluator:
    """
    Evaluates the time performance of a PredecessorFinder implementation
    over a given set of test grids.
    """

    def __init__(self, finder_class):
        """
        finder_class: a subclass or instance of PredecessorFinder.
        """
        # Accept either a class or an instance
        self.finder = finder_class() if callable(finder_class) else finder_class

    def evaluate(self, test_grids: List[np.ndarray]) -> dict:
        """
        Runs the finder on all given grids and measures execution times.

        Returns a dictionary with min, max, avg, and per-grid timings.
        """
        times: List[float] = []

        for grid in test_grids:
            start = time.perf_counter()
            _ = self.finder.find_previous(grid)
            end = time.perf_counter()
            times.append(end - start)

        return {
            "count": len(times),
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / len(times),
            "times": times,
        }


class GameOfLifeVisualizer:
    """
    Visualizes the evolution of a Game of Life grid over time using matplotlib.
    Supports interactive exploration with Next/Previous buttons if a solver is provided.
    """
    
    def __init__(self, game_grid: GameOfLifeGrid, solver: Optional[PredecessorFinder] = None, interval: float = 200, loop_pause: float = 2000):
        """
        Initialize the visualizer.
        
        Args:
            game_grid: GameOfLifeGrid instance to visualize
            solver: Optional PredecessorFinder instance for finding previous states
            interval: Time between frames in milliseconds (default: 200ms) - used for auto-play (if implemented)
            loop_pause: Additional pause when loop repeats in milliseconds (default: 2000ms)
        """
        self.game_grid = game_grid
        self.solver = solver
        self.interval = interval
        self.loop_pause = loop_pause
        self.fig = None
        self.ax = None
        self.im = None
        self.step_text = None
        self.grids_sequence = [self.game_grid.grid.copy()]
        self.current_idx = 0
        
        # Buttons
        self.btn_prev = None
        self.btn_next = None
        self.ax_prev = None
        self.ax_next = None

    def visualize(self, steps: int = 0, title: str = "Conway's Game of Life"):
        """
        Start the visualization.
        
        Args:
            steps: Number of initial steps to pre-calculate (optional)
            title: Title for the visualization window
        """
        # Pre-calculate steps if requested
        if steps > 0:
            for _ in range(steps):
                self.game_grid.compute_next()
                self.game_grid.advance()
                self.grids_sequence.append(self.game_grid.grid.copy())
        
        self._setup_plot(title)
        plt.show()

    def visualize_from_sequence(self, grids: List[np.ndarray], title: str = "Conway's Game of Life"):
        """
        Visualize a pre-computed sequence of grids.
        
        Args:
            grids: List of grid states to visualize
            title: Title for the visualization window
        """
        self.grids_sequence = grids
        self.current_idx = 0
        self._setup_plot(title)
        plt.show()
    
    def _setup_plot(self, title: str):
        """Setup the matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.2) # Make room for buttons
        
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Initial image
        self.im = self.ax.imshow(
            self.grids_sequence[self.current_idx], 
            cmap='binary', 
            interpolation='nearest',
            vmin=0,
            vmax=1
        )
        
        # Add a text element to show the current step
        self.step_text = self.ax.text(
            0.02, 0.98, f'Step: {self.current_idx}', 
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # Add Buttons
        from matplotlib.widgets import Button
        
        self.ax_prev = plt.axes([0.3, 0.05, 0.15, 0.075])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_prev.on_clicked(self._on_prev)
        
        self.ax_next = plt.axes([0.55, 0.05, 0.15, 0.075])
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_next.on_clicked(self._on_next)

    def _update_display(self):
        """Update the plot with the current grid."""
        self.im.set_array(self.grids_sequence[self.current_idx])
        self.step_text.set_text(f'Step: {self.current_idx}')
        self.fig.canvas.draw_idle()

    def _on_next(self, event):
        """Handle Next button click."""
        if self.current_idx < len(self.grids_sequence) - 1:
            self.current_idx += 1
        else:
            # Calculate next step
            # We need to use the last grid in the sequence
            last_grid = self.grids_sequence[-1]
            temp_game = GameOfLifeGrid(grid=last_grid)
            temp_game.compute_next()
            temp_game.advance()
            new_grid = temp_game.grid.copy()
            self.grids_sequence.append(new_grid)
            self.current_idx += 1
        
        self._update_display()

    def _on_prev(self, event):
        """Handle Previous button click."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self._update_display()
        elif self.solver:
            # Try to find a predecessor
            print("Finding predecessor...")
            current_grid = self.grids_sequence[0]
            prev_grid = self.solver.find_previous(current_grid)
            
            if prev_grid is not None:
                print("Predecessor found!")
                self.grids_sequence.insert(0, prev_grid)
                # current_idx stays 0, but now points to the new previous grid
                self._update_display()
            else:
                print("No predecessor found.")
        else:
            print("No previous history and no solver provided.")

    def save_animation(self, filename: str, steps: int, fps: int = 5, dpi: int = 100):
        """
        Save the animation to a file (requires ffmpeg or pillow).
        This method creates a non-interactive animation for saving.
        """
        # Generate sequence if needed
        temp_seq = [self.game_grid.grid.copy()]
        temp_game = GameOfLifeGrid(grid=self.game_grid.grid.copy())
        for _ in range(steps):
            temp_game.compute_next()
            temp_game.advance()
            temp_seq.append(temp_game.grid.copy())
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(temp_seq[0], cmap='binary', interpolation='nearest', vmin=0, vmax=1)
        
        def update(frame):
            im.set_array(temp_seq[frame])
            return [im]

        anim = FuncAnimation(
            fig,
            update,
            frames=len(temp_seq),
            interval=1000/fps,
            blit=True
        )
        
        anim.save(filename, writer='pillow' if filename.endswith('.gif') else 'ffmpeg', fps=fps, dpi=dpi)
        print(f"Animation saved to {filename}")
        plt.close(fig)

