# Planner Imitation Time Optimization Results

Date: 2026-05-17

This note summarizes the experiment-time optimizations currently implemented
for the PushWorld planner-imitation pipeline and the measurements available so
far. The focus is wall-clock experiment throughput, not changing the policy
checkpoint or search objective.

## Baseline Context

Best checkpoint:

```text
models/planner_imitation_level0_multi4_convlog_e6.pt
```

Headline Level1 search setting:

```text
beam_width=8
beam_depth=8
top_k=3
max_steps=100
repeat_penalty=1.0
beam_score=policy_distance
distance_weight=0.15
beam_length_normalization=0.0
```

Before the optimization pass, the same Level1 eval was observed to take about
`20m26s` while solving `18/68`.

## Implemented Optimizations

### Persistent Planner-Imitation Cache

Training can now use a persistent cache via:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -u scripts\train_planner_imitation_v2.py `
  --cache-dir data\cache\planner_imitation\<cache_name> `
  ...
```

There is also a standalone builder:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -u scripts\build_planner_imitation_cache.py `
  --train-dir data\level0\base\train `
  --cache-dir data\cache\planner_imitation\<cache_name>
```

The cache stores:

- selected train puzzle paths and SHA-256 content hashes;
- RGD expert plans and solve-time metadata;
- pre-encoded base state plane tensors;
- action targets;
- linear remaining-step targets;
- puzzle dimensions and puzzle indices for optional symmetry transforms.

On a cache hit, training skips RGD calls, puzzle parsing for train traces, and
base-state tensor encoding. The manifest is validated against the requested
puzzle list, board shape, schema version, and file content hashes.

### Structured Training Profile

Training summaries now include a `profile` object with:

- `cache`: cache enabled/hit/load/build/size information;
- `data.rgd_solve_time_s`;
- `data.dataset_materialization_time_s`;
- `data.puzzle_parse_time_s`;
- `data.state_encode_time_s`;
- `data.env_step_time_s`;
- `train.dataloader_wait_time_s`;
- `train.forward_backward_update_time_s`;
- `train.optimizer_steps`;
- `train.examples`;
- `train.epochs`;
- `train.total_time_s`.

This makes a training run explain whether time is going into RGD, cache IO,
dataset materialization, dataloader waiting, or the actual optimizer loop.

### Prediction Cache For Beam Evaluation

The large eval speedup comes from caching model outputs, not just encoded
tensors.

The cache key is:

```text
(puzzle_key, state)
```

The cached value is:

```text
(action_log_probs, expected_distance)
```

At every `predict_batch` call, the evaluator:

1. checks whether each requested state already has cached model outputs;
2. collects only unique uncached states;
3. runs one model forward for those misses;
4. stores the policy log-probs and expected distance estimate;
5. reconstructs the batch output in the original requested order.

This matters because closed-loop beam search replans every real environment
step. The lookahead trees overlap heavily between adjacent steps, especially on
failed puzzles that run to the full `100`-step budget. Previously, many of
those repeated states were re-forwarded through the transformer. Now they are
served from the prediction cache.

### Structured Eval Profile

Eval summaries now include:

- total wall time;
- `solves_per_minute`;
- encode cache entries;
- prediction cache entries;
- puzzle parse time;
- state encode time;
- model forward time;
- environment stepping time;
- beam expansion time;
- beam ranking time;
- encode cache hit/miss counts;
- prediction cache hit/miss counts;
- model forward batch/state counts;
- requested vs unique-forward state counts;
- beam candidate counts.

There is also an opt-in `--closed-list-pruning` flag, but the main result below
does not use it. That keeps the comparison behaviorally aligned with the prior
headline setting.

## Current Results

### Full Level1 Eval

Run date: 2026-05-17

Command shape:

```powershell
$env:PYTHONPATH='src'
C:\Users\adams\AppData\Local\Programs\Python\Python313\python.exe -u scripts\eval_planner_imitation.py `
  --checkpoint models\planner_imitation_level0_multi4_convlog_e6.pt `
  --eval-dir external\pushworld\benchmark\puzzles\level1 `
  --split-name level1_multi4_convlog_e6_p1_s100_profiled `
  --all-eval `
  --max-steps 100 `
  --beam-width 8 `
  --beam-depth 8 `
  --top-k 3 `
  --repeat-penalty 1.0 `
  --beam-score policy_distance `
  --distance-weight 0.15 `
  --output reports\eval_level1_multi4_convlog_e6_p1_s100_profiled.json
```

The eval completed, but the JSON write failed in that shell because the active
repo path was outside the session's writable root. The terminal summary was:

| Metric | Value |
| --- | ---: |
| puzzles | 68 |
| solved | 18 |
| success rate | 26.47% |
| wall time | 72.67s |
| solves per minute | 14.86 |
| old observed wall time | about 1226s |
| speedup at same solve count | about 16.9x |

Profile:

| Metric | Value |
| --- | ---: |
| requested prediction states | 660,116 |
| unique model-forward states | 28,740 |
| prediction-cache hits | 631,376 |
| prediction-cache hit rate | 95.6% |
| model forward batches | 7,856 |
| model forward time | 41.73s |
| env stepping time | 12.11s |
| beam expansion time | 19.21s |
| beam ranking time | 0.60s |
| state encode time | 2.10s |
| puzzle parse time | 4.44s |
| encode cache entries | 28,740 |
| prediction cache entries | 28,740 |
| beam candidates considered | 687,932 |

Interpretation:

- The success result stayed at `18/68`, matching the previous best Level1
  result for this checkpoint and search setting.
- The evaluator logically requested `660,116` state predictions, but only
  `28,740` unique states required model forwards.
- The prediction cache avoided about `95.6%` of repeated model-output
  computations.
- Runtime dropped from the old observed `20m26s` to about `1m13s`.

### Small Cache Smoke

Detailed smoke report:

```text
reports/planner_imitation_time_optimization_smoke.md
```

Subset:

- train dir: `data/level0/base/train`;
- train puzzles: `2`;
- examples/actions: `9`;
- board: `7x7`;
- zero-epoch startup comparison plus one cached one-epoch dataloader run.

Results:

| Run | RGD solve time | Dataset materialization | Cache load | Train wrapper time |
| --- | ---: | ---: | ---: | ---: |
| uncached | 0.0286s | 0.0047s | 0.0000s | 1.7952s |
| cache hit | 0.0000s | 0.0000s | 0.0275s | 1.3915s |

One cached one-epoch smoke:

| Metric | Value |
| --- | ---: |
| optimizer steps | 3 |
| examples | 9 |
| dataloader wait time | 0.0104s |
| forward/backward/update time | 0.5476s |

The tiny smoke is too small for meaningful absolute speedup, but it verifies
the intended behavior: a cache-hit run skips RGD and train-trace
materialization.

## Practical Meaning

For Level1 evals, the main bottleneck was repeated transformer scoring during
beam replanning. The prediction cache directly attacks that path and already
turns the full headline Level1 eval into an approximately one-minute run
without losing solved puzzles.

For training, the persistent cache is primarily infrastructure for repeated
experiments. It removes repeated RGD trace generation and base tensor encoding
from subsequent runs. The next full training comparison should use the full
multi4 dataset and report:

- cache build time and disk size;
- cache-hit training startup time;
- 1-epoch uncached vs cached wall time;
- 6-epoch cached training time;
- final Level0 and Level1 eval time with the profiled evaluator.

