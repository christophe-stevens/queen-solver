"""Speed benchmark: backtracking vs the trained network on the same test puzzles."""
import sys
import time
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
sys.path.insert(0, str(ROOT / "Tool"))
sys.path.insert(0, str(RESEARCH))

import numpy as np
import torch
from queens import solvers
from queens_research.queen_model import ZoneSinkhornQueenModel

DATA_FILE = ROOT / "data" / "queens_7x7_speed3_stretch5.npz"
MODEL_FILE = ROOT / "weights" / "queen_plain_big_aug.pt"
N_TEST = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(DATA_FILE)
Z = torch.tensor(data["zones"][-N_TEST:]).long()
Q = torch.tensor(data["queens"][-N_TEST:]).argmax(2)
M = Z.shape[-1]
model = ZoneSinkhornQueenModel(m=M, width=128, depth=8).to(device).load(MODEL_FILE)


def timed(fn):
    # Run fn, return (result, seconds); waits for the GPU to finish
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


# 1. Backtracking only, columns left to right
res, sec = timed(lambda: [solvers.backtrack(z) for z in Z])
tries = [t for _, t in res]
print(f"Backtracking:        {sec / N_TEST * 1000:.2f} ms/puzzle, "
      f"{np.mean(tries):.0f} placements on average")

# 2. Network, one puzzle at a time (as for a single screenshot)
model.solve(Z[0])  # warm-up: the first GPU call is always slow
res, sec = timed(lambda: [model.solve(z) for z in Z])
solved = np.mean([(c == q).all().item() for (c, _), q in zip(res, Q)])
print(f"Network, one by one: {sec / N_TEST * 1000:.2f} ms/puzzle, solved {solved:.1%}")

# 3. Network, all puzzles in one batch (model only, no checks)
with torch.no_grad():
    _, sec = timed(lambda: model(Z.to(device), steps=8))
print(f"Network, batched:    {sec / N_TEST * 1000:.3f} ms/puzzle (scores only)")

# 4. Full pipeline: network first, guided backtracking if it fails
res, sec = timed(lambda: [model.solve_or_search(z) for z in Z])
fallback = [t for _, r, t in res if r is None]
print(f"Network + fallback:  {sec / N_TEST * 1000:.2f} ms/puzzle, "
      f"fallback on {len(fallback)} puzzles, {np.mean(fallback) if fallback else 0:.0f} placements each")