"""Train the zone model: puzzle image -> grid size and zone number for each cell."""
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent
ROOT = RESEARCH.parent
sys.path.insert(0, str(ROOT / "Tool"))  # make the queens package importable when run directly

import json
import random
import numpy as np
import torch
import torch.nn as nn
from queens.config import SIZES
from queens.zone_model import ZoneModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

DATA_DIR = Path("D:/GAME_AI/QUEEN/")
WEIGHTS = ROOT / "weights" / "zone_model_any_size.pt"
N_TEST = 300                # last puzzles kept for testing (a random mix of sizes)
STEPS, BATCH = 10000, 32    # training batches, images per batch
TRAIN = False                # False: load saved weights instead of training

# Load the mixed puzzles
d = np.load(DATA_DIR / "puzzles.npz")
images, sizes, zones = d["images"], d["sizes"], d["zones"]
test = np.arange(len(sizes)) >= len(sizes) - N_TEST

# Group by size for batching only (labels of different sizes can't share a batch)
X_tr, Y_tr, X_te, Y_te = {}, {}, {}, {}
for m in SIZES:
    for part, X, Y in ((~test, X_tr, Y_tr), (test, X_te, Y_te)):
        idx = (sizes == m) & part
        X[m] = torch.tensor(images[idx]).permute(0, 3, 1, 2)  # uint8 (N,3,H,W)
        Y[m] = torch.tensor(zones[idx, :m, :m]).long()        # un-pad to m x m

model = ZoneModel().to(device)


def to_input(x):
    # uint8 images -> 0-1 decimals on the right device
    return x.to(device).float() / 255


# Train and save, or load a model trained earlier
if TRAIN:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.5)  # halve lr every 3,000 steps
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for step in range(STEPS):
        m = random.choice(SIZES)                              # one grid size per batch
        idx = torch.randint(len(X_tr[m]), (BATCH,))           # random images of that size
        x = to_input(X_tr[m][idx])
        size_target = torch.full((BATCH,), SIZES.index(m), device=device)  # class number of size m
        opt.zero_grad()
        zone_loss = loss_fn(model(x, m), Y_tr[m][idx].to(device))  # zone of each cell
        size_loss = loss_fn(model.size_scores(x), size_target)     # grid size of the image
        loss = zone_loss + size_loss                               # learn both at once
        loss.backward()
        opt.step()
        sched.step()  # count one step towards the next halving
        if step % 200 == 0:
            print(step, "zones:", round(zone_loss.item(), 4), "size:", round(size_loss.item(), 4))
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    model.save(WEIGHTS)
else:
    model.load(WEIGHTS)
model.eval()

# 1. Per size, on the test arrays (size given, to judge zone reading on its own)
print("\nPer size (zones read with the true size):")
with torch.no_grad():
    for m in SIZES:
        if len(X_te[m]) == 0:
            continue  # no test puzzles of this size
        x = to_input(X_te[m])
        size_ok = np.mean([f == m for f in model.detect_size(x)])
        fit_ok = np.mean([f == m for f in model.detect_size_by_fit(x)])
        pred = model(x, m).argmax(1).cpu()
        cells = (pred == Y_te[m]).float().mean().item()
        puzzles = (pred == Y_te[m]).flatten(1).all(1).float().mean().item()
        print(f"{m}x{m} ({len(x)} puzzles): size branch {size_ok:.0%}, by fit {fit_ok:.0%}, "
              f"cells {cells:.1%}, whole puzzles {puzzles:.1%}")

# 2. End to end, like real use: PNG file in, size and zones detected, checked against the JSON
size_right, zones_right = 0, 0
for i in range(len(sizes) - N_TEST, len(sizes)):
    info = json.loads((DATA_DIR / f"puzzle_{i}.json").read_text())
    pred = model.read_image(DATA_DIR / f"puzzle_{i}.png").cpu()  # size detected by the model
    if pred.shape[0] == info["size"]:
        size_right += 1
        zones_right += bool((pred.numpy() == np.array(info["zones"])).all())
print(f"\nFrom PNG files: size right {size_right / N_TEST:.1%}, "
      f"size and all zones right {zones_right / N_TEST:.1%}")