import time
from typing import List
from abc import ABC, abstractmethod
import numpy as np


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
    def find_previous(self, grid: np.ndarray) -> np.ndarray:
        """
        Given a grid (numpy 2D array), attempt to find a previous grid
        that could have produced it according to the Game of Life rules.
        Must return a numpy array of the same shape or None if no valid predecessor exists.
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
