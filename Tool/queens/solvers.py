"""Rule-based solving of Queens puzzles: no learning involved."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

def is_valid(zones, cols):
    # True if queens use every column and zone once and none touch (one per row by design)
    cols = [int(c) for c in cols]
    m = len(cols)
    return (len(set(cols)) == m
            and len({int(zones[r][c]) for r, c in enumerate(cols)}) == m
            and all(abs(cols[r] - cols[r + 1]) > 1 for r in range(m - 1)))

def same_grouping(a, b):
    # True if two zone grids group the cells identically, whatever the zone numbers
    pairs = set(zip(a.ravel().tolist(), b.ravel().tolist()))
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))


def backtrack(zones, order=None, limit=1):
    # Up to `limit` solutions (column per row), trying columns in `order`;
    # also returns the number of placements tried
    m = len(zones)
    order = order or [list(range(m))] * m  # default: left to right
    sols, cols, used_c, used_z, tries = [], [], set(), set(), [0]

    def place(row):
        if row == m:
            sols.append(cols.copy())  # all rows filled: a solution
            return
        for c in order[row]:
            z = int(zones[row][c])
            if c in used_c or z in used_z or (cols and abs(cols[-1] - c) <= 1):
                continue  # column or zone taken, or touches the queen above
            tries[0] += 1
            cols.append(c); used_c.add(c); used_z.add(z)
            place(row + 1)
            cols.pop(); used_c.remove(c); used_z.remove(z)  # undo, try next
            if len(sols) >= limit:
                return  # found enough

    place(0)
    return sols, tries[0]


def count_solutions(zones, limit=2):
    # Number of solutions, counting at most `limit`
    return len(backtrack(zones, limit=limit)[0])


def assign(logp):
    # (rows,cols) log-probabilities -> best column per row, each column used once
    return linear_sum_assignment(-logp.detach().cpu().numpy())[1]


def balance(s, z, iters=10, use_zones=True):
    # Raking on log-scores (B,rows,cols): rows, columns (and zones) each sum to 1
    m = z.shape[-1]
    mask = F.one_hot(z, m).bool() if use_zones else None
    for _ in range(iters):
        s = s - s.logsumexp(dim=2, keepdim=True)  # rows
        s = s - s.logsumexp(dim=1, keepdim=True)  # columns
        if use_zones:
            zs = s.unsqueeze(-1).masked_fill(~mask, float("-inf"))  # each zone's cells
            s = s - zs.logsumexp(dim=(1, 2)).gather(1, z.flatten(1)).view_as(s)  # zones
    return s


def rake(z, iters=50):
    # No network: start from equal scores and balance rows, columns and zones
    m = z.shape[-1]
    s = torch.zeros(z.shape[0], m, m, device=z.device)
    return balance(s, z, iters)

