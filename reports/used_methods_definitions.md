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

A global guided search added after the beam baseline. It keeps a priority
queue of partial paths instead of replanning a local beam after every action.
The priority is:

```text
policy_cost + distance_weight * predicted_remaining_distance
+ step_penalty * path_length
```

The final learned headline uses pure best-first with budget `1024`,
`top_k=3`, `max_depth=100`, and the same `distance_weight=0.15`. A smaller
budget `512` is the fastest learned throughput setting. Fallback-to-beam was
tested, but pure best-first dominated it in the final sweep.

**Inference caches**

Evaluation uses in-memory caches keyed by `(puzzle_path, state)`:

- encoded tensor cache: avoids repeated state-to-plane encoding;
- prediction cache: avoids repeated model forwards for the same state.

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
