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


class PredecessorFinder(ABC):
    """
    Abstract base class for finding the previous (parent) grid
    that could have evolved into a given Game of Life grid.
    """

    @abstractmethod
    def find_previous(self, grid: np.ndarray, seed: Optional[int] = None, exclude: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        """
        Given a grid (numpy 2D array), attempt to find a previous grid
        that could have produced it according to the Game of Life rules.
        
        Args:
            grid: The target grid.
            seed: Optional integer seed for randomization.
            exclude: Optional list of grids to exclude from the search (to avoid cycles or visited states).
            
        Returns:
            A numpy array of the same shape or None if no valid predecessor exists.
        """
        pass


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
    Visualizes the evolution of a Game of Life grid over time using matplotlib animation.
    Can display a looping animation of n steps with customizable timing.
    """
    
    def __init__(self, game_grid: GameOfLifeGrid, interval: float = 200, loop_pause: float = 2000):
        """
        Initialize the visualizer.
        
        Args:
            game_grid: GameOfLifeGrid instance to visualize
            interval: Time between frames in milliseconds (default: 200ms)
            loop_pause: Additional pause when loop repeats in milliseconds (default: 2000ms)
        """
        self.game_grid = game_grid
        self.interval = interval
        self.loop_pause = loop_pause
        self.fig = None
        self.ax = None
        self.im = None
        self.animation = None
        self.grids_sequence = []
        self.current_frame = 0
        
    def generate_sequence(self, steps: int) -> List[np.ndarray]:
        """
        Generate a sequence of grids by evolving the game for n steps.
        
        Args:
            steps: Number of steps to evolve
            
        Returns:
            List of grid states [initial, step1, step2, ..., step_n]
        """
        sequence = [self.game_grid.grid.copy()]
        
        for _ in range(steps):
            self.game_grid.compute_next()
            self.game_grid.advance()
            sequence.append(self.game_grid.grid.copy())
        
        return sequence
    
    def visualize_from_sequence(self, grids: List[np.ndarray], title: str = "Conway's Game of Life"):
        """
        Visualize a pre-computed sequence of grids.
        
        Args:
            grids: List of grid states to visualize
            title: Title for the visualization window
        """
        self.grids_sequence = grids
        self._setup_plot(title)
        self._create_animation()
        plt.show()
    
    def visualize(self, steps: int, title: str = "Conway's Game of Life"):
        """
        Visualize the evolution of the game for n steps with looping.
        
        Args:
            steps: Number of steps to evolve and display
            title: Title for the visualization window
        """
        # Generate the sequence
        self.grids_sequence = self.generate_sequence(steps)
        
        # Setup and show
        self._setup_plot(title)
        self._create_animation()
        plt.show()
    
    def _setup_plot(self, title: str):
        """Setup the matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Initial image
        self.im = self.ax.imshow(
            self.grids_sequence[0], 
            cmap='binary', 
            interpolation='nearest',
            vmin=0,
            vmax=1
        )
        
        # Add a text element to show the current step
        self.step_text = self.ax.text(
            0.02, 0.98, '', 
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    def _update_frame(self, frame_num: int):
        """
        Update function for animation.
        
        Args:
            frame_num: Current frame number from FuncAnimation
            
        Returns:
            Updated artists
        """
        # Calculate which grid to show (with looping)
        grid_idx = frame_num % len(self.grids_sequence)
        
        # Update the image
        self.im.set_array(self.grids_sequence[grid_idx])
        
        # Update the step counter
        self.step_text.set_text(f'Step: {grid_idx}/{len(self.grids_sequence)-1}')
        
        return [self.im, self.step_text]
    
    def _frame_interval(self, frame_num: int) -> float:
        """
        Calculate the interval for the current frame.
        Adds extra pause when the loop repeats.
        
        Args:
            frame_num: Current frame number
            
        Returns:
            Interval in milliseconds
        """
        # Check if we're at the end of a loop (about to restart)
        if frame_num > 0 and frame_num % len(self.grids_sequence) == 0:
            return self.loop_pause
        return self.interval
    
    def _create_animation(self):
        """Create the animation object."""
        # We'll use a simple approach: repeat the sequence indefinitely
        # and handle the pause manually by adjusting intervals
        
        # Create animation that loops indefinitely
        self.animation = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self._frame_generator(),
            interval=self.interval,
            blit=True,
            repeat=True,
            cache_frame_data=False
        )
    
    def _frame_generator(self):
        """
        Generator that yields frame numbers and handles loop pausing.
        """
        frame = 0
        while True:
            yield frame
            frame += 1
            
            # Add pause at the end of each loop
            if frame % len(self.grids_sequence) == 0:
                # Pause by yielding the same frame multiple times
                pause_frames = int(self.loop_pause / self.interval)
                for _ in range(pause_frames):
                    yield frame - 1
    
    def save_animation(self, filename: str, steps: int, fps: int = 5, dpi: int = 100):
        """
        Save the animation to a file (requires ffmpeg or pillow).
        
        Args:
            filename: Output filename (e.g., 'game.gif' or 'game.mp4')
            steps: Number of steps to include in the saved animation
            fps: Frames per second
            dpi: Dots per inch for the output
        """
        # Generate sequence if not already done
        if not self.grids_sequence:
            self.grids_sequence = self.generate_sequence(steps)
        
        # Setup plot
        self._setup_plot("Conway's Game of Life")
        
        # Create a finite animation for saving
        anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=len(self.grids_sequence),
            interval=1000/fps,
            blit=True,
            repeat=True
        )
        
        # Save
        anim.save(filename, writer='pillow' if filename.endswith('.gif') else 'ffmpeg', fps=fps, dpi=dpi)
        print(f"Animation saved to {filename}")

