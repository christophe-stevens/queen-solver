"""Train the queen model: zone grid -> queen column per row."""
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent
ROOT = RESEARCH.parent
sys.path.insert(0, str(ROOT / "Tool"))  # queens_research's __init__ pulls in queens.drawer
sys.path.insert(0, str(RESEARCH))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from queens_research import ZoneSinkhornQueenModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

M, STEPS = 7, 6                                  # grid size, rounds during training
WIDTH, DEPTH = 128, 8                            # channels per layer, number of 3x3 layers
N_TEST = 2000                                    # puzzles kept back for testing
TRAIN = True                                     # False: load saved weights instead of training
DATA_FILE = ROOT / "data" / "queens_7x7_speed3_stretch5.npz"
MODEL_CLASS = ZoneSinkhornQueenModel             # or QueenModel, or SinkhornQueenModel
WEIGHTS = ROOT / "weights" / "queen_plain_big_aug.pt"

# Data: zones (input), queen column per row (target)
data = np.load(DATA_FILE)
Z = torch.tensor(data["zones"]).long()
Q = torch.tensor(data["queens"]).argmax(2)
Z_tr, Q_tr, Z_te, Q_te = Z[:-N_TEST], Q[:-N_TEST], Z[-N_TEST:], Q[-N_TEST:]
Z_te = torch.randperm(M)[Z_te]  # hide the zone-number = queen-row link
Z_tr, Q_tr, Z_te, Q_te = Z_tr.to(device), Q_tr.to(device), Z_te.to(device), Q_te.to(device)

model = MODEL_CLASS(m=M, width=WIDTH, depth=DEPTH, steps=STEPS).to(device)


def augment(z, q):
    # Random flip/rotation: same rules, new-looking puzzles (8 symmetries of the square)
    g = F.one_hot(q, M)  # queen column per row -> (B,rows,cols) grid of 0/1
    if random.random() < 0.5:
        z, g = z.flip(1), g.flip(1)                    # upside down
    if random.random() < 0.5:
        z, g = z.flip(2), g.flip(2)                    # mirror left-right
    if random.random() < 0.5:
        z, g = z.transpose(1, 2), g.transpose(1, 2)    # swap rows and columns
    return z, g.argmax(2)                              # back to queen column per row


def row_acc(z, q):
    # Share of rows with the right column, last round, no rules enforced
    with torch.no_grad():
        return (model(z)[-1].argmax(1) == q).float().mean().item()


# Train, or load a model trained earlier
if TRAIN:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)  # halve lr every 30 epochs
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(150):
        model.train()
        total, batches = 0, 0
        for idx in torch.randperm(len(Z_tr), device=device).split(256):  # shuffled batches of 256
            perm = torch.randperm(M, device=device)  # shuffle zone numbers: only the grouping matters
            zb, qb = augment(perm[Z_tr[idx]], Q_tr[idx])
            opt.zero_grad()
            outs = model(zb)
            loss = sum(loss_fn(s, qb) for s in outs) / STEPS
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        sched.step()
        model.eval()
        train_acc = row_acc(Z_tr[:N_TEST], Q_tr[:N_TEST])
        test_acc = row_acc(Z_te, Q_te)
        print(epoch, "loss:", round(total / batches, 4),
              "train rows:", round(train_acc, 3), "test rows:", round(test_acc, 3))
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    model.save(WEIGHTS)
else:
    model.load(WEIGHTS)
model.eval()

# Evaluate on unseen puzzles
with torch.no_grad():
    outs = model(Z_te)
for rnd in (0, STEPS - 1):
    raw = outs[rnd].argmax(1)  # each row's top column, no rules enforced
    print(f"Round {rnd + 1}, raw:", (raw == Q_te).all(1).float().mean().item())

# All test puzzles through the model at once, then check each on the CPU
with torch.no_grad():
    outs8 = [s.cpu() for s in model(Z_te, steps=8)]
zc = Z_te.cpu()
results = []
for i in range(len(zc)):
    for rnd, s in enumerate(outs8, start=1):
        cols = model.decode(s[i])
        if model.is_valid(zc[i], cols):
            results.append((cols, rnd))
            break
    else:
        results.append((cols, None))  # no valid round

pred3 = torch.stack([c for c, _ in results])
rounds = [r for _, r in results if r is not None]
print("Solved (one per column, valid, retry up to 8 rounds):",
      (pred3 == Q_te.cpu()).all(1).float().mean().item())
print("Average rounds needed when solved:", round(np.mean(rounds), 2))
