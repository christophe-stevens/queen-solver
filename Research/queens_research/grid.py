import random
import numpy as np

class Grid:
    """Random Queens puzzle: zones (input) and queen placement (answer)."""
    COLOURS = "YRBPGWOCKNLTMS"  # yellow, red, blue, purple, green, white, orange, cyan, pink, brown, lime, teal, magenta, sand

    def __init__(self, m=5, speed=10, stretch=2, max_trials=100, unique=True):
        self.m = m
        self.speed = speed      # how much faster one zone grows than the others
        self.stretch = stretch  # 0 = round blobs, higher = longer, snakier zones
        while True:  # regenerate until the puzzle is acceptable
            self.queens = self._place_queens(max_trials)
            self.zones = self._spread_zones()
            if not unique or self.count_solutions() == 1:
                break  # any puzzle if unique=False, else only one-solution puzzles

    def __str__(self):
        # Queens and zones side by side
        return "\n".join(" ".join("Q" if q else "." for q in qr) + "    " +
                         " ".join(self.COLOURS[z] for z in zr)
                         for qr, zr in zip(self.queens, self.zones))

    def as_arrays(self):
        # Public access for building the dataset
        return self.zones.copy(), self.queens.copy()

    def _place_queens(self, max_trials):
        while True:  # retry until a valid placement is found
            try:
                return self._try_place_queens(max_trials)
            except ValueError:
                pass

    def _try_place_queens(self, max_trials):
        grid = np.zeros((self.m, self.m), dtype=int)
        free_cols = list(range(self.m))
        for row in range(self.m):
            for _ in range(max_trials):
                col = random.choice(free_cols)
                if not self._has_adjacent(grid, row, col):
                    break
            else:
                raise ValueError("Stuck: no valid column")
            free_cols.remove(col)
            grid[row, col] = 1
        return grid

    def _has_adjacent(self, grid, row, col):
        # True if a queen sits diagonally next to (row, col)
        return any(0 <= r < self.m and 0 <= c < self.m and grid[r, c]
                   for r in (row - 1, row + 1) for c in (col - 1, col + 1))

    def _spread_zones(self):
        m = self.m
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        zones = np.where(self.queens == 1, np.arange(m)[:, None], -1)  # seed: queen = row number
        weights = np.ones(m)
        weights[random.randrange(m)] = self.speed  # one random zone grows faster
        # each zone's taste for each direction: high values make long or curvy zones
        pref = [{d: random.expovariate(1) ** self.stretch for d in dirs} for _ in range(m)]
        while (zones == -1).any():
            # free neighbouring cells for each zone, with the direction they lie in
            free = {z: [((r + dr, c + dc), (dr, dc))
                        for r, c in zip(*np.where(zones == z))
                        for dr, dc in dirs
                        if 0 <= r + dr < m and 0 <= c + dc < m and zones[r + dr, c + dc] == -1]
                    for z in range(m)}
            growing = [z for z in range(m) if free[z]]  # skip enclosed zones
            z = random.choices(growing, weights=[weights[z] for z in growing])[0]
            cell, _ = random.choices(free[z], weights=[pref[z][d] for _, d in free[z]])[0]
            zones[cell] = z
        return zones

    def count_solutions(self, limit=2):
        # Count solutions by backtracking, stopping once `limit` is reached
        m = len(self.zones)
        cols = []  # queen column for each row placed so far
        used_cols, used_zones = set(), set()

        def place(row):
            if row == m:
                return 1  # all rows filled: one solution found
            found = 0
            for c in range(m):
                z = self.zones[row][c]
                if c in used_cols or z in used_zones:
                    continue  # column or zone already taken
                if cols and abs(cols[-1] - c) <= 1:
                    continue  # touches the queen in the row above
                cols.append(c)
                used_cols.add(c)
                used_zones.add(z)
                found += place(row + 1)
                cols.pop()
                used_cols.remove(c)
                used_zones.remove(z)
                if found >= limit:
                    break  # enough found, stop searching
            return found
        return place(0)
