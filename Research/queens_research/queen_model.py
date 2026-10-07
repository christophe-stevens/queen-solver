import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class QueenModel(nn.Module):
    """Zone grid -> queen column per row, refined over several rounds."""

    def __init__(self, m=7, width=64, depth=6, steps=6):
        super().__init__()
        self.m, self.steps = m, steps
        layers = [nn.Conv2d(m + 1, width, 3, padding=1), nn.ReLU()]  # zones + previous guess
        for _ in range(depth - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Conv2d(width, 1, 1))     # one score per cell

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, z, steps=None):
        # (B,m,m) zones -> list of log-probabilities (B,cols,rows), one per round
        zones = F.one_hot(z, self.m).permute(0, 3, 1, 2).float()
        probs = torch.zeros(len(z), self.m, self.m, device=z.device)  # no guess yet
        outs = []
        for _ in range(steps or self.steps):
            s = self.net(torch.cat([zones, probs.unsqueeze(1)], dim=1)).squeeze(1)  # (B,rows,cols)
            s = self.normalise(s, z)          # log-probabilities
            outs.append(s.permute(0, 2, 1))  # (B,cols,rows) for the loss and decoding
            probs = s.exp()                # guess fed into the next round
        return outs

    def normalise(self, s, z):
        # Each row sums to 1 (one queen per row)
        return s.log_softmax(dim=2)

    def is_valid(self, z, cols):
        # True if the placement breaks no rule
        m = self.m
        return (len(set(cols.tolist())) == m
                and len(set(z[torch.arange(m), cols].tolist())) == m
                and all(abs(cols[r] - cols[r + 1]) > 1 for r in range(m - 1)))

    @staticmethod
    def decode(scores):
        # Scores (cols,rows) of one puzzle -> one queen per column
        logp = F.log_softmax(scores, dim=0).T
        return torch.tensor(linear_sum_assignment(-logp.numpy())[1])

    def solve(self, z, max_rounds=8):
        # First valid answer over the rounds; round is None if none found
        with torch.no_grad():
            outs = self(z.unsqueeze(0).to(self.device), steps=max_rounds)
        zc = z.cpu()
        for rnd, s in enumerate(outs, start=1):
            cols = self.decode(s[0].cpu())
            if self.is_valid(zc, cols):
                return cols, rnd
        return cols, None

    def solve_or_search(self, z, max_rounds=8):
        # Model first; if it fails, backtracking guided by the model's ranking
        cols, rnd = self.solve(z, max_rounds)
        if rnd is not None:
            return cols, rnd, 0
        with torch.no_grad():
            s = self(z.unsqueeze(0).to(self.device), steps=max_rounds)[-1][0].cpu()
        order = s.T.argsort(dim=1, descending=True).tolist()  # per row: columns, best first
        cols, tries = self.backtrack(z.cpu(), order)
        return cols, None, tries

    def backtrack(self, z, order):
        # First valid placement, trying columns in the given order per row
        cols, used_c, used_z, tries = [], set(), set(), [0]

        def place(row):
            if row == self.m:
                return True
            for c in order[row]:
                zone = int(z[row, c])
                if c in used_c or zone in used_z or (cols and abs(cols[-1] - c) <= 1):
                    continue
                tries[0] += 1
                cols.append(c); used_c.add(c); used_z.add(zone)
                if place(row + 1):
                    return True
                cols.pop(); used_c.remove(c); used_z.remove(zone)
            return False

        place(0)
        return torch.tensor(cols), tries[0]

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        return self

class SinkhornQueenModel(QueenModel):
    """Same model, but each round's guess has rows AND columns summing to 1."""

    def __init__(self, *args, iters=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.iters = iters

    def normalise(self, s, z):
        # Sinkhorn: alternately make rows and columns sum to 1
        for _ in range(self.iters):
            s = s - s.logsumexp(dim=2, keepdim=True)  # rows
            s = s - s.logsumexp(dim=1, keepdim=True)  # columns
        return s


class ZoneSinkhornQueenModel(SinkhornQueenModel):
    """Rows, columns AND zones each sum to 1 (raking on three margins)."""

    def normalise(self, s, z):
        mask = F.one_hot(z, self.m).bool()  # (B,rows,cols,zones): True where a cell is in that zone
        for _ in range(self.iters):
            s = s - s.logsumexp(dim=2, keepdim=True)  # rows
            s = s - s.logsumexp(dim=1, keepdim=True)  # columns
            zs = s.unsqueeze(-1).masked_fill(~mask, float("-inf"))  # each zone's cells; -inf elsewhere
            lse = zs.logsumexp(dim=(1, 2))                          # (B,zones): log total per zone
            s = s - lse.gather(1, z.flatten(1)).view_as(s)          # subtract each cell's zone total
        return s
