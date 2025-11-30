from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np

class PreviousFinder(ABC):
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
