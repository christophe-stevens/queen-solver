"""Control experiment: how many puzzles do the constraints alone solve, with no network?"""
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent  # repo top folder
sys.path.insert(0, str(ROOT / "Tool"))

import numpy as np
import torch
from queens.solvers import rake, assign, is_valid

DATA_FILE = ROOT / "data" / "queens_7x7_speed3_stretch5.npz"
N_TEST = 2000  # same test puzzles as the trained models

data = np.load(DATA_FILE)
Z = torch.tensor(data["zones"][-N_TEST:]).long()
Q = torch.tensor(data["queens"][-N_TEST:]).argmax(2)  # true column per row

s = rake(Z)                                           # (B,rows,cols) log-probabilities
raw = s.argmax(2)                                     # each row's most likely column
assigned = torch.stack([torch.tensor(assign(si)) for si in s])  # one queen per column

print("Rows, columns and zones balanced, no network:")
print("  raw, solved:            ", (raw == Q).all(1).float().mean().item())
print("  with assignment, solved:", (assigned == Q).all(1).float().mean().item())
print("  with assignment, valid: ", np.mean([is_valid(z, c) for z, c in zip(Z, assigned)]))