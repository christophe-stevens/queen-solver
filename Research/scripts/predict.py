"""Solve a puzzle image: deep learning reads size and zones, backtracking places the queens."""
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent
ROOT = RESEARCH.parent
sys.path.insert(0, str(ROOT / "Tool"))

import json
import time
from types import SimpleNamespace
import numpy as np
from queens.zone_model import ZoneModel
from queens.solvers import backtrack, same_grouping
from queens.drawer import GridDrawer

IMAGE = RESEARCH / "examples" / "example1_7x7.png"
ZONE_WEIGHTS = ROOT / "weights" / "zone_model_any_size.pt"
OUT = ROOT / "output" / "prediction.png"

zone_model = ZoneModel().load(ZONE_WEIGHTS)

# 1. Deep learning: image -> grid size and zones
t0 = time.perf_counter()
zones = zone_model.read_image(IMAGE).cpu().numpy()  # size detected by fit
m = zones.shape[0]
t1 = time.perf_counter()

# 2. Rules: exact backtracking search for the queens
sols, tries = backtrack(zones)
t2 = time.perf_counter()
if not sols:
    sys.exit("No solution found: the zones were probably misread.")
queens = np.zeros((m, m), dtype=int)
queens[np.arange(m), sols[0]] = 1

print("Image:", IMAGE)
print(f"\nDetected size: {m}x{m}")
print("Predicted zones:")
print(zones)
print("\nQueens:")
for row in queens:
    print(" ".join("Q" if q else "." for q in row))
print(f"\nReading the image: {(t1 - t0) * 1000:.0f} ms; "
      f"solving: {(t2 - t1) * 1000:.1f} ms, {tries} placements tried")

# Compare with the answers, if present (JSON or the older text files)
json_file = IMAGE.with_suffix(".json")
zones_file = IMAGE.with_name(IMAGE.stem + "_zones.txt")
queens_file = IMAGE.with_name(IMAGE.stem + "_queens.txt")
if json_file.exists():
    info = json.loads(json_file.read_text())
    true_zones, true_queens = np.array(info["zones"]), np.array(info["queens"])
elif zones_file.exists() and queens_file.exists():
    true_zones, true_queens = np.loadtxt(zones_file, dtype=int), np.loadtxt(queens_file, dtype=int)
else:
    true_zones = true_queens = None
if true_zones is not None:
    print("\nSize correct:", true_zones.shape[0] == m)
    print("Zones correct:", true_zones.shape == zones.shape and same_grouping(zones, true_zones))
    print("Queens correct:", true_queens.shape == queens.shape and bool((queens == true_queens).all()))

# Save a picture of the solution
OUT.parent.mkdir(parents=True, exist_ok=True)
GridDrawer(SimpleNamespace(m=m, zones=zones, queens=queens)).draw(show_queens=True).save(OUT)
print("Saved", OUT)