import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from time import sleep

from GameOfLife import GameOfLife
import numpy as np
from ILPMinPreviousFinder import ILPMinPreviousFinder as Solver

lock = threading.Lock()   
GOLS_DIR = "./gols"
NUM_THREADS = 1

def get_random_file(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    if not files:
        return None
    return os.path.join(folder, random.choice(files))


def worker(worker_id: int):
    solver = Solver()
    while True:
        # Safely select a file
        with lock:
            file_path = get_random_file(GOLS_DIR)

        if file_path is None:
            print(f"[Worker {worker_id}] No files found. Sleeping...")
            sleep(2)
            continue

        try:
            print(f"[Worker {worker_id}] Processing {file_path}...")
            gol = GameOfLife(path_and_hash=file_path)
            prev = gol.get_previous_state(solver)
            if prev is None:
                print(f"[Worker {worker_id}] No previous state possible.")
                continue
            else:
                print(f"[Worker {worker_id}] Found previous state (at depth: {prev.depth}) and hash: {prev.hash}.")

            prev.save_to_file(GOLS_DIR)

        except Exception as e:
            sleep(2)
            print(f"[Worker {worker_id}] Error processing {file_path}: {e}")


def main():
    if(True):
        for file in os.listdir(GOLS_DIR):
            os.remove(os.path.join(GOLS_DIR, file))
        og_grid = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        gol = GameOfLife(depth=0, grid=og_grid)
        gol.pretty_print()
        gol.save_to_file(GOLS_DIR)
        gol.save_to_file(GOLS_DIR)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for i in range(NUM_THREADS):
            executor.submit(worker, i)


if __name__ == "__main__":
    main()
