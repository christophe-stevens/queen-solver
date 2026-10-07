"""Render puzzles of random sizes: one PNG + one JSON per puzzle, and one combined .npz."""
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent
ROOT = RESEARCH.parent
sys.path.insert(0, str(ROOT / "Tool"))
sys.path.insert(0, str(RESEARCH))

import json
import random
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from queens_research.grid import Grid
from queens.drawer import GridDrawer
from queens.config import SIZES, MAX_ZONES

OUT = Path("D:/GAME_AI/QUEEN/")
TOTAL, PIXELS = 3300, 280  # number of puzzles; training image width and height
UNIQUE = False             # True: only puzzles with exactly one solution (much slower)
SAVE_SOLVED = False        # True: also save a picture with the queens, for checking by eye


def one_puzzle(i):
    # Puzzle i with a random size; saves its image and JSON, returns arrays for the .npz
    m = random.choice(SIZES)
    g = Grid(m, speed=random.uniform(1, 5), stretch=random.uniform(0, 5), unique=UNIQUE)
    zones, queens = g.as_arrays()
    drawer = GridDrawer(g)
    base = OUT / f"puzzle_{i}"
    unsolved = drawer.draw(show_queens=False)
    unsolved.save(f"{base}.png")
    if SAVE_SOLVED:
        drawer.draw(show_queens=True).save(f"{base}_solved.png")
    info = {"size": m, "zones": zones.tolist(), "queens": queens.tolist()}
    Path(f"{base}.json").write_text(json.dumps(info))
    return np.array(unsolved.resize((PIXELS, PIXELS))), zones, queens, m


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool() as pool:  # one worker per CPU core
        results = list(tqdm(pool.imap(one_puzzle, range(TOTAL), chunksize=10), total=TOTAL))

    # Pad every grid to MAX_ZONES x MAX_ZONES so all puzzles fit in one array
    zones = np.full((TOTAL, MAX_ZONES, MAX_ZONES), -1)  # -1 = outside the grid
    queens = np.zeros((TOTAL, MAX_ZONES, MAX_ZONES), dtype=int)
    for k, (_, z, q, m) in enumerate(results):
        zones[k, :m, :m], queens[k, :m, :m] = z, q
    np.savez_compressed(OUT / "puzzles.npz",
                        images=np.array([r[0] for r in results]),
                        sizes=np.array([r[3] for r in results]),
                        zones=zones, queens=queens)