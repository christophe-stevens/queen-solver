from types import SimpleNamespace
import numpy as np
from queens.drawer import GridDrawer

def solve_image(path, zone_model, queen_model, m, max_rounds=8):
    """Puzzle image -> zones -> queens -> picture of the solution.

    Returns (zones, queens, image, rnd, tries): zones and queens are (m,m)
    numpy arrays, image is the solved-puzzle picture, rnd is the round the
    model solved it in (None if the backup solver was needed instead), and
    tries is the number of placements the backup solver tried (0 if the
    model solved it directly).
    """
    zones = zone_model.read_image(path).cpu()
    cols, rnd, tries = queen_model.solve_or_search(zones, max_rounds=max_rounds)
    queens = np.zeros((m, m), dtype=int)
    queens[np.arange(m), cols.numpy()] = 1
    zones = zones.numpy()
    image = GridDrawer(SimpleNamespace(m=m, zones=zones, queens=queens)).draw(show_queens=True)
    return zones, queens, image, rnd, tries
