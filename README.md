# Queens Solver: rules, search or deep learning?

A learning and research project around the **Queens** logic puzzle (the genre is known as *Star Battle*). The question behind it:

> **To solve a logic puzzle, how far do simple rules get you, how far does exhaustive search get you, and what can a neural network actually learn?**

The repository contains a puzzle generator, three kinds of solver, two neural networks (one that reads puzzle images, one that solves puzzles) and a small desktop tool that solves a puzzle from a screenshot. Everything is measured on the same test puzzles, so the approaches can be compared fairly.

The screenshot tool is not the point of the project; it's the last step of it. Once a rule-based solver and two trained networks exist, the natural question is whether the whole pipeline also works end to end on a real puzzle straight off the screen, not just on generated training data. That question is about the solver, not about the puzzle: the goal was to find out how well each approach reasons, not to make the game easier.

This README is written to be pedagogical: each concept is explained the first time it appears.

> This is a research and learning project, not a cheating tool. It is not intended for competitive play: please solve puzzles yourself before using the tool on them.

---

## Contents

1. [The puzzle](#1-the-puzzle)
2. [Three ways to solve it](#2-three-ways-to-solve-it)
3. [Making training data](#3-making-training-data)
4. [The deep learning models](#4-the-deep-learning-models)
5. [Results](#5-results)
6. [Reading puzzles from images](#6-reading-puzzles-from-images)
7. [Conclusions](#7-conclusions)
8. [Repository layout and usage](#8-repository-layout-and-usage)

---

## 1. The puzzle

An m × m grid is divided into m coloured **zones**. Place m queens so that there is:

- exactly **one queen per row**,
- exactly **one queen per column**,
- exactly **one queen per zone**,
- and **no two queens touching**, not even diagonally.

A well-made puzzle has **exactly one solution**. Grids typically range from 7 × 7 to 11 × 11.

```
Zones (7x7)              Solution
3 3 3 3 3 3 3            . . . . Q . .
6 6 6 3 2 2 2            Q . . . . . .
6 4 4 3 2 3 2            . . Q . . . .
6 6 6 3 3 3 2            . . . . . . Q
6 5 6 1 1 3 2            . Q . . . . .
6 6 6 1 1 3 2            . . . Q . . .
0 0 0 0 0 0 0            . . . . . Q .
```

This is a **constraint satisfaction problem**: a set of choices (where each queen goes) that must all satisfy a set of rules at once.

---

## 2. Three ways to solve it

### 2.1 Rule propagation (no search, no learning)

The rules can be written as **totals**: every row, every column and every zone must contain exactly one queen. Start by giving every cell the same probability of holding a queen, then repeatedly rescale:

1. scale each **row** so its probabilities sum to 1,
2. scale each **column** so it sums to 1,
3. scale each **zone** so it sums to 1,

and repeat until the numbers settle. Statisticians know this as **iterative proportional fitting** (or *raking*); in machine learning, the row-and-column version is called **Sinkhorn normalisation**.

This makes the "obvious" deductions automatically: a one-cell zone ends up with probability 1, which pushes its row and column towards 0 elsewhere, and so on. What it cannot do is follow longer chains of reasoning ("if the queen were here, that zone would have no room left"), and it knows nothing about the no-touching rule, which is not a total.

### 2.2 Backtracking (exhaustive search)

Place queens row by row:

1. In the current row, try the first column that breaks no rule so far.
2. Move to the next row and do the same.
3. If a row has no allowed column left (a **dead end**), go back to the previous row and move its queen to its next allowed column.
4. When every row has a queen, that is a solution.

Because it eventually tries every possibility that has not been ruled out, backtracking is **guaranteed** to find the solution. The order in which columns are tried only affects **speed**, not correctness. The same routine also **counts** solutions, which is used to check that generated puzzles are unique.

### 2.3 Deep learning

A **neural network** is a function with many adjustable numbers (**weights**). **Training** shows it many examples with known answers and nudges the weights, step by step, to make its answers less wrong. The measure of "wrong" is the **loss**; the nudging is done by an **optimiser** (here *Adam*) using **gradients** computed by **backpropagation**.

Two networks are used:

- a **zone reader**: puzzle image → grid size and zone of each cell;
- a **queen solver**: zone grid → position of each queen.

The queen solver is the interesting one: can a network learn to *reason* about the rules from examples alone, without being told them and without searching?

---

## 3. Making training data

Deep learning needs many examples, so the project includes a generator (`queens/grid.py`).

### 3.1 Queens first, then zones

1. **Place the queens** at random, one per row and column, with no two touching. This placement becomes the answer.
2. **Grow the zones** around them: each queen's cell seeds its own zone, and zones grow one neighbouring cell at a time (up, down, left or right) until the grid is full. Every zone therefore contains exactly one queen, so the placement is always a valid solution.

### 3.2 The uniqueness problem

Randomly grown zones usually allow **many** solutions. Counting them with the backtracking solver:

| Grid | Unique puzzles | Median number of solutions | Range |
|---|---|---|---|
| 5 × 5 | 0.2% | 6 | 2 to 9 |
| 7 × 7 | 0% | 109 | 31 to 194 |
| 9 × 9 | 0% | 2,400 | 435 to 5,267 |

This matters enormously for learning (see section 5.1). Two generator settings fix it:

- **`speed`**: one random zone grows faster than the others, squeezing the rest into small zones. Small zones constrain the queens, so far fewer solutions remain. Puzzles that still have more than one solution are rejected.
- **`stretch`**: each zone gets its own preferred growth directions, producing long strips, L-shapes and snakes, much like real puzzles.

| Setting | Effect |
|---|---|
| speed 10 | Mostly unique, but easy: one huge zone and several one-cell "giveaway" zones |
| speed 5 | Harder, still quick to generate |
| speed 3, stretch 5 | Realistic snaky zones, the hardest set used here |

Generation runs on all CPU cores (`multiprocessing`) and is saved once to a `.npz` file, so models can be retrained without regenerating.

---

## 4. The deep learning models

### 4.1 The queen solver

**Input.** Zone numbers are categories, not quantities (zone 6 is not "more" than zone 2), so each cell is **one-hot encoded**: one yes/no channel per zone, like dummy coding a factor in a regression.

**Network.** A **convolutional neural network** (CNN): each layer slides small 3 × 3 filters over the grid, so each cell combines information from its neighbours. Stacking 8 layers lets information travel across the whole board. No pooling is used, so the grid stays m × m throughout. The final layer gives each cell one score.

**Output.** For each row, a **softmax** over the columns turns scores into probabilities that sum to 1: "where in this row is the queen?". This builds *one queen per row* into the design. Training uses **cross-entropy**, the same quantity that multinomial logistic regression minimises.

**Rounds of self-correction.** The network runs several times in a row. After each **round**, its current queen probabilities are fed back in as an extra input channel, so the next round can see where it put queens and spot clashes. The same weights are used every round, and the loss is averaged over all rounds, so the model learns both to guess well early and to fix its mistakes later. (This resembles *auto-context* in image segmentation and iterative algorithms such as EM.)

**Decoding with the column rule.** When predicting, the **assignment problem** (solved with `scipy.optimize.linear_sum_assignment`, the Hungarian algorithm) picks the most probable set of columns with **no column used twice**.

**Retries.** If a round's answer breaks a rule, the next round's answer is tried, up to 8 rounds. Because each puzzle has exactly one solution, a valid answer is always the correct one. This step only *checks*; it never creates an answer.

**Three variants** differ only in how each round's scores are normalised:

| Variant | Normalisation inside each round |
|---|---|
| Plain | rows sum to 1 (softmax) |
| Sinkhorn | rows **and** columns sum to 1 |
| Zone Sinkhorn | rows, columns **and** zones sum to 1 |

The Sinkhorn variants build some rules directly into the model, using the same balancing as the rule-propagation solver of section 2.1.

### 4.2 Training techniques that mattered

- **Zone renumbering.** The zone numbers carry no meaning, so they are randomly shuffled in every batch. The generator numbers each zone after its queen's row, which would otherwise leak the answer; test zones are shuffled too, to make the evaluation honest.
- **Symmetry augmentation.** Flipping a puzzle upside down, mirroring it, or swapping rows and columns gives a new, equally valid puzzle with one solution. Combined, these give **8 versions of every puzzle** for free (the symmetries of a square). The queens are transformed together with the zones.
- **Learning-rate schedule.** The step size is halved every 30 epochs, letting the model settle into a better solution.
- **GPU training** (an NVIDIA RTX 2060): a CNN of this size trains in minutes rather than hours.

### 4.3 Measuring progress

| Metric | Meaning |
|---|---|
| **Row accuracy** | Share of rows with the queen in the right column. Moves smoothly during training. |
| **Exact match** | Share of puzzles with every queen right. One wrong row spoils the puzzle. |
| **Valid answers** | Share of answers breaking no rule. Equals exact match when puzzles are unique. |
| **Raw** | The network's own answer, with no rules enforced afterwards: the purest measure of what it has learned. |
| **Solved** | After the column-assignment step and retries. |

**Overfitting** is checked by comparing row accuracy on training puzzles with row accuracy on **unseen test puzzles**. A large gap means the model is memorising rather than learning rules.

---

## 5. Results

### 5.1 The road from 4% to 99%

Each step changed one thing. Results are for 7 × 7 puzzles never seen in training.

| Step | Puzzles solved |
|---|---|
| Random zones (about 100 solutions per puzzle) | 4% exact match (20% valid) |
| Unique puzzles only (speed 10) | 54% to 57% |
| + one queen per column enforced when decoding | 85% |
| + rounds of self-correction and retries | 92% |
| Harder puzzles (speed 5), same method | 80% |
| + 6 rounds, 150 epochs, learning-rate decay | 84% |

Two lessons stand out.

**Fixing the data mattered more than the model.** With ~100 valid solutions per puzzle, the stored answer was one of many correct ones, chosen at random. The network was punished for correct answers, a form of **label noise**, and could not learn. Making puzzles unique took it from 4% to 55% without touching the model.

**Rules at decoding time help a lot.** Many raw mistakes were two rows choosing the same column; the assignment step alone took 57% to 85%.

### 5.2 Hard, realistic puzzles (speed 3, stretch 5)

13,000 training puzzles, 2,000 test puzzles. Network: 128 channels, 8 layers, 6 rounds.

| Model | Train rows | Test rows | Round 1, raw | Last round, raw | Solved |
|---|---|---|---|---|---|
| Plain, no augmentation | 100% | 85% | 55% | 69% | 71% |
| Plain + symmetry augmentation | 98% | 96% | 50% | 93% | 95.2% |
| Sinkhorn (rows, columns) + augmentation | 99% | 98% | 78% | 96% | 96.5% |
| **Zone Sinkhorn (rows, columns, zones) + augmentation** | | | **2%** | **99%** | **99.1%** |

- **Augmentation cured overfitting.** Without it, the large model memorised its training puzzles (100% against 85%). Eight symmetric versions of each puzzle closed the gap and lifted test accuracy by 11 points.
- **Built-in constraints help.** Each added constraint improved the final score, and made training much faster: the zone version reached 97% of rows correct after 3 epochs.
- **The zone version "thinks" over rounds.** Its first round is almost uninformative (2%): with all three totals balanced, it spreads probability evenly instead of committing early. The following rounds sharpen the guess. It needed 1.9 rounds on average, against 1.2 for the plain model.

### 5.3 Is the network learning, or are the rules doing the work?

The **control experiment** runs rule propagation (section 2.1) on the same 2,000 test puzzles, with no network at all:

| Method (same 2,000 test puzzles) | Solved |
|---|---|
| Rule propagation alone (rows, columns, zones balanced) | **11%** |
| Plain network (no rules built in) | 95% |
| Network + rows and columns | 96.5% |
| Network + rows, columns and zones | 99% |
| Backtracking | 100% |

The rules alone solve 11%; the trained network solves 99%. **Roughly 88 percentage points are learned reasoning**, not imposed constraints. The plain network, which has only "one per row" built in, already solves 95% from examples alone.

Note that the zone network's first round (2%) is close to rule propagation alone (1.4% raw), which fits: before seeing any guess, it behaves much like plain balancing. The reasoning happens in the later rounds.

### 5.4 Speed

Same 2,000 hard 7 × 7 test puzzles; Python on CPU for search, RTX 2060 for the network.

| Method | Time per puzzle | Notes |
|---|---|---|
| **Backtracking** | **4 ms** | 126 placements on average; always correct |
| Network, one puzzle at a time | 61 ms | 99% solved |
| Network, all puzzles in one batch | 0.18 ms | scores only, no checks |
| Network + backtracking fallback | 62 ms | fallback on 21 puzzles, 59 placements each |

For a single 7 × 7 puzzle, **backtracking wins easily**. The network's actual computation is tiny (0.18 ms in a batch); nearly all of its 61 ms is overhead from running 8 small GPU jobs per puzzle. Search effort grows quickly with grid size, while the network's cost grows only with the number of cells, so the comparison may change on larger grids.

**The network still makes search smarter.** When backtracking tries columns in the order the network prefers, it needs about half as many placements (59 against 126 on average). On one hand-checked example puzzle: 75 placements unguided, 26 to 50 guided, depending on the model.

---

## 6. Reading puzzles from images

### 6.1 Zones and grid size

The zone reader is a small CNN. Its features are **pooled** into an m × m grid (one feature vector per cell), and each cell is classified into one of 14 colours. It works for any size from 4 × 4 to 14 × 14, trained on 3,300 generated images of mixed sizes.

**Detecting the grid size** was harder than reading the zones:

| Size detection | Overall accuracy | Weakest size |
|---|---|---|
| A separate network branch trained to recognise the size | 93% | 9 × 9: 52% |
| **"By fit": try every size, keep the one the zone reader reads most confidently** | **99.7%** | 6 × 6: 97% |

Why the zone reader succeeds where the size branch struggled: the zone reader is *handed* the cells (the pooling cuts the image into m × m squares), and only has to name colours. Counting the grid is a different skill. The "fit" trick exploits this: at the right size, every square falls inside one colour and the reader is confident everywhere; at a wrong size, squares straddle borders and confidence drops. The smallest well-fitting size is chosen, because a 7 × 7 grid read as 14 × 14 also fits perfectly. **No extra training was needed.**

| End-to-end, from PNG files (300 mixed-size test images) | Accuracy |
|---|---|
| Zone reading, given the true size | 100% at every size |
| Size detected and every zone right | 99.7% |

### 6.2 Real screenshots: rules win again

On a real 9 × 9 screenshot, the network found the right size and layout, but **misnamed colours** it had never seen: one zone's colour was mapped to the same "known" colour as part of another zone, so the grid became unsolvable.

A simple rule fixed it: take the median colour at each cell's centre, and **group cells whose colours match each other**, loosening the match until exactly m groups remain. This needs no knowledge of the palette. On the screenshot, it recovered all 9 zones exactly, and backtracking found the unique solution.

| Task | Best method here | Why |
|---|---|---|
| Grid size | Zone network, "by fit" (a rule-based version is also possible) | Hard to write as a rule for any visual style |
| Zones | Colour grouping (rule) | Works with any palette, no training |
| Queens | Backtracking (rule) | Always right, under a millisecond |

---

## 7. Conclusions

1. **Data quality came first.** Ambiguous puzzles (many solutions) made learning impossible; unique puzzles took accuracy from 4% to 55% with no change to the model. Symmetry augmentation then turned an overfitting model (85% on test) into a generalising one (96%).
2. **Neural networks can learn real reasoning.** A network solves 99% of hard puzzles in a few rounds, with no search, where the rules alone solve 11%. In effect, **it learned to replace search**.
3. **Building known structure into a model helps.** Sinkhorn normalisation of rows, columns and zones made learning faster and more reliable (95% → 99%).
4. **But where the rules are known and the input is clean, rules win.** Backtracking is exact and faster on these grids, and colour grouping read a real screenshot that the trained reader got wrong. The final tool uses deep learning only where a rule is hard to write.
5. **Combining both works best:** learned guidance halves the search effort of an exact solver, and exact checks catch the network's mistakes.

The broader lesson, familiar from clinical decision support as much as from puzzles: **use rules where you know them, and learning where you don't.**

### Limitations and ideas

- Most deep learning results are on 7 × 7 grids; scaling the queen network to all sizes (one-hot over 14 channels, mixed-size batches) is the natural next step, alongside a speed comparison by grid size.
- The generator's zones are realistic but not identical to the real game; a small hand-labelled set of real puzzles would be the honest final test.
- A fully rule-based size detector (checking colour uniformity per cell for each candidate size) would remove PyTorch from the tool entirely.
- Messy input (phone photos, perspective, dark mode, half-filled grids) is where learned image reading should earn its place.
- Differences of 1 to 2 points between models on 2,000 puzzles are near the noise level; McNemar's test on paired results is the right check.

---

## 8. Repository layout and usage

```
queen-solver/
├── Tool/                     # everything needed to run the snip tool, and nothing else
│   ├── snip_tool.py          # desktop tool: select a puzzle on screen, see the solution
│   ├── queens/               # lightweight, rule-based library used by the tool
│   │   ├── zone_model.py     # image -> size and zones (the trained reader)
│   │   ├── drawer.py         # draws puzzles and solutions
│   │   ├── colour_zones.py   # zones by colour grouping
│   │   ├── solvers.py        # backtracking, rule propagation, validity checks
│   │   └── config.py         # shared settings (sizes 4 to 14)
│   └── requirements.txt      # Tool's own, smaller dependency set (no PyTorch training extras)
├── Research/                 # the deep learning study, and everything used to produce it
│   ├── queens_research/
│   │   ├── grid.py           # puzzle generator (queens first, zone growing, uniqueness)
│   │   ├── queen_model.py    # plain, Sinkhorn and zone-Sinkhorn solvers
│   │   └── pipeline.py       # image -> zones -> queens -> picture, end to end
│   ├── scripts/
│   │   ├── generate_puzzles.py      # training data for the queen model, in parallel
│   │   ├── generate_images.py       # training images for the zone model
│   │   ├── train_queen_model.py
│   │   ├── train_zone_model.py
│   │   ├── predict.py               # solve examples/example1_7x7.png
│   │   ├── evaluate_rule_baseline.py
│   │   └── benchmark_speed.py
│   ├── assets/
│   └── examples/
├── data/                      # generated puzzles / training images (gitignored)
├── weights/                   # trained models (.pt), shared by Tool and Research
└── output/                    # captures and predictions (gitignored)
```

Research's scripts import the library straight from `Tool/queens` (one copy, not two) and add their
own extras from `Research/queens_research`.

### Setup

```bash
python -m venv .venv
pip install -r requirements.txt       # torch, numpy, pillow, scipy, tqdm
# just the tool, without the training/research dependencies:
pip install -r Tool/requirements.txt
```

### Run

```bash
python Tool/snip_tool.py                     # click "Select puzzle", drag around the grid, get the solution
python Research/scripts/predict.py           # solve examples/example1_7x7.png
python Research/scripts/train_queen_model.py # retrain a queen solver
```

Select tightly around the grid's outer border: extra margin shifts every cell.

---

*A side project built to learn deep learning engineering end to end: data generation, training and debugging on a GPU, evaluation, and deployment in a small tool. Not affiliated with LinkedIn.*