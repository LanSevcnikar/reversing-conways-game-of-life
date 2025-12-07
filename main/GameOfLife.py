from ABCPreviousFinder import PreviousFinder

import numpy as np
import os

class GameOfLife:
    def __init__(self, grid: np.ndarray = None, path_and_hash: str = None,  depth: int = None):
        if grid is not None:
            self.grid = grid
        elif path_and_hash is not None:
            self.load_from_file(path_and_hash)
        else:
            raise ValueError("Either grid or path_and_hash must be provided")

        self.height = self.grid.shape[0]
        self.width = self.grid.shape[1]
    
        self.hash = hash(tuple(map(tuple, self.grid)))
        self.next = None
        self.previous = []
        if(depth is not None):
            self.depth = depth

    def get_hash(self):
        """
            Returns the hash of the current game of life
        """
        return self.hash

    def pretty_print(self):
        """
        Print the grid in a pretty format using unicode block characters.
        At the bottom, print the number of live cells as well as the dimentions of the smallest bounding box.
        """
        
        print("┌" + "─" * (self.width * 2) + "┐")
        for row in self.grid:
            print("│", end="")
            for cell in row:
                if cell == 1:
                    print("██", end="")
                else:
                    print("  ", end="")
            print("│")
        print("└" + "─" * (self.width * 2) + "┘")
            
        live_cells = np.sum(self.grid)
        bounding_box_top = np.min(np.where(self.grid)[0])
        bounding_box_bottom = np.max(np.where(self.grid)[0])
        bounding_box_left = np.min(np.where(self.grid)[1])
        bounding_box_right = np.max(np.where(self.grid)[1])
        bounding_box_height = bounding_box_bottom - bounding_box_top + 1
        bounding_box_width = bounding_box_right - bounding_box_left + 1
        print(f"Hash: {self.hash}")
        print(f"Depth: {self.depth}")
        print(f"Live cells: {live_cells}")
        print(f"Bounding box of size {bounding_box_height} x {bounding_box_width}")

    def save_to_file(self, path: str):
        """
            Save everything to the file. Format consists of:
            - hash
            - grid
            File name is path/{hash}.txt
        """
        filename = os.path.join(path, f"{self.hash}.txt")
        with open(filename, "w") as f:
            f.write(f"{self.hash}\n")
            f.write(f"{self.depth}\n")
            for row in self.grid:
                f.write(" ".join(map(str, row)) + "\n")

    def load_from_file(self, path_and_hash: str):
        """
        Load everything from the file. Format consists of:
        - hash
        - grid
        """
        filename = path_and_hash
        with open(filename, "r") as f:
            self.hash = int(f.readline())
            self.depth = int(f.readline())
            temp_grid = []
            for line in f:
                temp_grid.append(list(map(int, line.strip().split())))
            self.grid = np.array(temp_grid)

    def get_next_state(self) -> "GameOfLife":
        """
        Get the next state of the grid.
        This could be sped up but it seems like this really will never be a bottleneck.
        """
        next_grid = np.zeros_like(self.grid)

        for r in range(self.height):
            for c in range(self.width):
                live_neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue

                        nr, nc = r + dr, c + dc

                        if 0 <= nr < self.height and 0 <= nc < self.width:
                            live_neighbors += self.grid[nr, nc]

                current_cell_state = self.grid[r, c]

                if current_cell_state == 1:  
                    if live_neighbors < 2 or live_neighbors > 3:
                        next_grid[r, c] = 0
                    else:
                        next_grid[r, c] = 1
                else:  
                    if live_neighbors == 3:
                        next_grid[r, c] = 1
                    else:
                        next_grid[r, c] = 0
        
        return GameOfLife(depth=self.depth - 1, grid=next_grid)


    def get_previous_state(self, previous_finder: PreviousFinder) -> "GameOfLife":
        """
        Get the previous state of the grid.
        """
        seed = np.random.randint(0, 1_000_000)
        previous_grid = previous_finder.find_previous(self.grid, seed)
        if(previous_grid is None):
            return None
        return GameOfLife(depth=self.depth + 1, grid=previous_grid)
