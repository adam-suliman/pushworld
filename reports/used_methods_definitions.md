# Used Methods And Definitions

This file is the compact glossary for the PushWorld planner-imitation
experiments. Detailed numbers live in:

- `reports/planner_optimization_analysis.md`
- `reports/planner_imitation_time_optimization_results.md`

## Data And Expert Source

**PushWorld puzzle**

A `.pwp` grid puzzle loaded through the upstream `pushworld` package. The
agent chooses one of four actions: `L`, `R`, `U`, `D`.

**Level0**

Generated local data in `data/level0`. Each variant has 2000 train puzzles and
200 test puzzles. The main variants used here were `base`, `all`, `shapes`,
and `obstacles`.

**Level1**

The upstream benchmark split in
`external/pushworld/benchmark/puzzles/level1`, containing 68 hand-authored
puzzles.

**RGD / N+RGD planner**

The upstream C++ expert solver:

```powershell
external\pushworld\cpp\build\bin\run_planner.exe N+RGD <puzzle.pwp>
```

It is used in two distinct ways:

- to generate supervised imitation traces for Level0 training;
- as a planner reference baseline on Level1.

Every emitted plan is validated with `PushWorldPuzzle.is_valid_plan` before it
is accepted. The final RGD reference run used a `10s` per-puzzle timeout and
solved `68/68` Level1 puzzles.

That `68/68` result is Level1-only. It is not directly comparable to the
PushWorld paper's all-level benchmark curve over 223 puzzles from Levels 1-4.
In a quick local `1s` per-puzzle all-level probe, the same `N+RGD` executable
solved `108/223` and timed out on `115`.

## Runtime Device Roles

The recent learned timing runs used this local stack:

```text
python=C:\Users\adams\AppData\Local\Programs\Python\Python313\python.exe
torch=2.10.0.dev20251006+cu130
cuda_available=True
cuda_device=NVIDIA GeForce RTX 2060
```

For learned beam and best-first evaluation, `--device auto` resolved to
`device=cuda` in the saved logs. The transformer model forward passes ran on
the GPU. Puzzle parsing, PushWorld environment stepping, search frontier
management, closed-list checks, and cache dictionaries ran on CPU.

For training, the neural forward/backward/update loop runs on CUDA when
available; the main conv/log run also used AMP. RGD trace generation, puzzle
parsing, dataset materialization, and dataloader work remain CPU-side.

For `N+RGD`, the C++ planner ran as an external CPU process. It did not use the
neural checkpoint or GPU.

## Supervised Imitation Setup

**Imitation trajectory**

For an expert plan of length `T`, each step becomes one training example:

- input: encoded current state;
- action target: next expert action;
- distance target: remaining expert steps, `T - step_index`.

**Board encoding**

Each state is encoded as a padded 7-channel board tensor:

| Channel | Meaning |
| ---: | --- |
| 0 | static walls |
| 1 | agent walls |
| 2 | controlled object / agent object |
| 3 | goal-associated movable objects |
| 4 | other movable objects |
| 5 | goal cells |
| 6 | already satisfied goal cells |

**Transformer policy**

The model has a board stem, positional embeddings, a CLS token, a
`TransformerEncoder`, an action head, and an auxiliary distance head. Training
uses:

```text
CE(action_logits, expert_action)
+ distance_loss_weight * CE(distance_logits, remaining_distance_target)
```

The main runs used `distance_loss_weight=0.2`.

**Linear stem / linear distance**

The original baseline configuration: per-cell linear projection and exact
remaining-step classes `0..max_steps`.

**Conv stem / log distance**

The optimized configuration: a two-layer local convolutional board stem before
the transformer, plus compact log-spaced distance bins:

```text
round(log(remaining_steps + 1))
```

For `max_steps=100`, this reduces the distance head from 101 exact classes to
6 bins. During search scoring, bins are decoded back to approximate remaining
steps.

**Multi4 training set**

The 8000-puzzle Level0 training mix:

- `data/level0/base/train`
- `data/level0/all/train`
- `data/level0/shapes/train`
- `data/level0/obstacles/train`

## Search And Evaluation Methods

**Beam rollout**

A receding-horizon search. At each environment step, it expands candidate
paths to `beam_depth`, keeps `beam_width` paths, ranks them, executes only the
first action of the best path, and repeats until solved or `max_steps`.

Main beam setting:

```text
beam_width=8
beam_depth=8
top_k=3
max_steps=100
repeat_penalty=1.0
beam_score=policy_distance
distance_weight=0.15
```

**Beam score**

The configurable ranker supports:

- `policy`: cumulative negative log policy probability;
- `distance`: predicted remaining distance;
- `policy_distance`: policy cost plus weighted distance estimate.

The main experiments use `policy_distance`.

**Repeat-state penalty**

A search-time cost added when a candidate revisits a state already seen in the
current rollout. It is behavior-changing and was important for the earlier
beam results.

**Best-first search**

A learned policy/value-guided planner added after the beam baseline. It uses
the trained checkpoint as a heuristic, but the search itself is explicit
planning over PushWorld states.

Unlike the beam rollout, best-first is not receding-horizon control. Beam
rollout repeatedly plans a shallow local tree, executes one action, then plans
again from the new real state. Best-first instead starts from the puzzle's
initial state and keeps one global frontier of partial paths until it either
finds a goal or exhausts its search limits.

The frontier is a priority queue. Each queue item stores:

- the current PushWorld state;
- the action path from the initial state to that state;
- the cumulative policy cost of that path;
- a priority score used to decide which partial path to expand next.

Lower priority is better. The priority score is:

```text
policy_cost + distance_weight * predicted_remaining_distance
+ step_penalty * path_length
```

where:

- `policy_cost` is the sum of negative log-probabilities of the chosen actions
  along the path;
- `predicted_remaining_distance` comes from the checkpoint's auxiliary
  distance/value head for the candidate state;
- `distance_weight` controls how much the distance head influences search;
- `step_penalty` is optional and was `0.0` in the main runs.

One iteration of the implementation does this:

1. Pop up to `best_first_batch_size` lowest-priority states from the frontier.
2. Skip states already in the per-puzzle closed set.
3. Run one batched model call on those states to get action log-probabilities.
4. For each expanded state, try only the model's top `best_first_top_k`
   actions.
5. Step the PushWorld environment for each selected action.
6. Drop no-op transitions and states already in the closed set.
7. Deduplicate candidates that reach the same next state, keeping the cheaper
   policy path.
8. Run a second batched model call on the unique candidate states to estimate
   remaining distance.
9. Push candidates back into the priority queue with the score above.

The search stops when it generates or pops a goal state, reaches
`best_first_max_depth`, exhausts the priority queue, or expands
`best_first_budget` states. The budget is therefore an expanded-node cap, not
a wall-clock cap.

This is not an optimal planner in the A* sense: the learned distance head is
not guaranteed to be admissible, and the search only branches over the top
model actions. It is better described as a learned heuristic planner or
policy-guided best-first search.

The main repeated learned headline uses pure best-first with budget `1024`,
`top_k=3`, `max_depth=100`, and `distance_weight=0.15`. Budget `512` is the
fastest learned throughput setting. A later single-run ceiling sweep reached a
higher solve count at budget `16384`, but at much worse solves/minute.
Fallback-to-beam was tested, but pure best-first dominated it in the final
sweep.

**Inference caches**

Evaluation uses in-memory caches keyed by `(puzzle_path, state)`:

- encoded tensor cache: avoids repeated state-to-plane encoding;
- prediction cache: avoids repeated model forwards for the same state.

The two caches have very different memory costs. A prediction-cache entry is
small, but an encoded-state entry stores a full padded board tensor. For the
current checkpoint board size, one encoded state is about `58.6 KiB`, so a
million-entry encoded cache can require tens of GB of CPU RAM. New large-budget
runs should keep the prediction cache large while disabling or sharply capping
the encoded-state cache, for example:

```powershell
--max-cache-entries 1000000 --max-encode-cache-entries 0
```

The cache is behavior-preserving. Cache on/off controls solved the same puzzle
sets for beam, best-first `512`, and best-first `1024`; only wall-clock time
changed.

**Persistent training cache**

The training cache stores validated RGD traces and pre-materialized training
tensors under `data/cache/planner_imitation/...`. On a cache hit, training
skips RGD calls and repeated train-state encoding. This is for experiment
startup time, not closed-loop inference.

## Current Compared Systems

| System | Training / solver | Search |
| --- | --- | --- |
| RGD reference | upstream `N+RGD` C++ planner | direct planner |
| Base linear | Level0 `base`, linear stem, linear distance | beam |
| Multi4 linear | Level0 multi4, linear stem, linear distance | beam |
| Multi4 conv/log | Level0 multi4, conv stem, log distance | beam |
| Optimized learned | same multi4 conv/log checkpoint | best-first + prediction cache |

Current headline learned result:

```text
multi4 conv/log checkpoint + prediction cache + best-first budget 1024
Level1: 32/68
Runtime: 138.02s +/- 1.02s over three full runs
Level0 base sanity: 199/200
```

The RGD row is a Level1-only planner reference, not a learned-policy result or
a reproduction of the paper's all-level benchmark curve. It is much faster on
Level1, but it is also the expert solver used to generate imitation traces.

## Main Code Additions

| Area | Added support |
| --- | --- |
| Model variants | `--encoder-stem`, conv stem, `--distance-target`, log bins, dropout |
| Beam controls | `--beam-score`, `--distance-weight`, `--beam-length-normalization` |
| Eval compatibility | checkpoint-native linear/conv and linear/log distance loading |
| Runtime profiling | train/eval timing sections and cache hit/miss counters |
| Eval caching | encoded-state cache and model-output prediction cache |
| Training cache | persistent trace/tensor cache via `--cache-dir` |
| Search | `--search-mode beam/best_first/best_first_fallback` |
| RGD baseline | `scripts/eval_rgd_baseline.py` |
| Missing-run suite | `scripts/run_missing_time_optimization_experiments.ps1` |

## Interpretation

The finished optimization story has three separable pieces:

1. Better model quality from multi4 conv/log training.
2. Faster evaluation from inference caching.
3. Better closed-loop solve rate from global best-first search.

The final comparison table and detailed timing controls are in
`reports/planner_imitation_time_optimization_results.md`.
