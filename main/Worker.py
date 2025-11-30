import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from time import sleep

GOLS_DIR = "./gols"
NUM_THREADS = 4

lock = threading.Lock()   # for safe file selection


def get_random_file(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    if not files:
        return None
    return os.path.join(folder, random.choice(files))


def worker(worker_id: int):
    while True:
        with lock:
            file_path = get_random_file(GOLS_DIR)

        if file_path is None:
            print(f"[Worker {worker_id}] No files found. Sleeping...")
            sleep(2)
            continue

        try:
            gol = GameOfLife(file_path)

            prev = gol.get_previous_state()
            if prev is None:
                print(f"[Worker {worker_id}] No previous state possible.")
                continue

            filename = f"prev_{worker_id}_{random.randint(1, 1_000_000)}.txt"
            save_path = os.path.join(OUTPUT_DIR, filename)

            prev.save_to_file(save_path)
            print(f"[Worker {worker_id}] Saved {filename}")

        except Exception as e:
            print(f"[Worker {worker_id}] Error processing {file_path}: {e}")

