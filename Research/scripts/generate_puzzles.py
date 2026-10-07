"""Generate a dataset of unique Queens puzzles for training the queen model."""
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent
ROOT = RESEARCH.parent
sys.path.insert(0, str(ROOT / "Tool"))  # queens_research's __init__ pulls in queens.drawer
sys.path.insert(0, str(RESEARCH))

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from queens_research import Grid

M, SPEED, STRETCH, N = 7, 3, 5, 15000
OUT = ROOT / "data" / "queens_7x7_speed3_stretch5.npz"


def one_puzzle(_):
    # One unique puzzle (the argument is just a counter we ignore)
    return Grid(M, speed=SPEED, stretch=STRETCH).as_arrays()


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with Pool() as pool:  # one worker per CPU core
        pairs = list(tqdm(pool.imap_unordered(one_puzzle, range(N), chunksize=20), total=N))
    zones = np.array([z for z, q in pairs])
    queens = np.array([q for z, q in pairs])
    np.savez_compressed(OUT, zones=zones, queens=queens)
    print("Saved", N, "puzzles to", OUT)
