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


## Full Level1 Cache On/Off Aggregate

Run files:

- `reports/eval_level1_cache_on_full.json`
- `reports/eval_level1_cache_off_full.json`

Both runs used the same checkpoint and search configuration:

```text
checkpoint=models/planner_imitation_level0_multi4_convlog_e6.pt
beam_width=8
beam_depth=8
top_k=3
max_steps=100
repeat_penalty=1.0
beam_score=policy_distance
distance_weight=0.15
```

### Behavioral Check

| Metric | Cache on | Cache off |
| --- | ---: | ---: |
| solved | 18/68 | 18/68 |
| success rate | 26.47% | 26.47% |
| solved puzzle set | identical | identical |
| total repeated rollout states | 1,876 | 1,876 |

The cache is behavior-preserving for this full Level1 run: it changes runtime,
not the selected solutions.

### Runtime And Throughput

| Metric | Cache on | Cache off | Ratio |
| --- | ---: | ---: | ---: |
| wall time | 78.09s | 1203.29s | 15.41x faster |
| solves per minute | 13.83 | 0.90 | 15.41x higher |
| model forward time | 43.03s | 1073.51s | 24.95x lower |
| state encode time | 2.37s | 47.06s | 19.86x lower |
| env step time | 13.58s | 19.56s | 1.44x lower |
| beam expansion time | 21.91s | 31.73s | 1.45x lower |
| beam ranking time | 0.74s | 1.47s | 1.99x lower |

### Prediction Cache Effect

| Metric | Cache on | Cache off |
| --- | ---: | ---: |
| requested prediction states | 660,116 | 660,116 |
| unique model-forward states | 28,740 | 660,116 |
| prediction-cache hits | 631,376 | 0 |
| prediction-cache misses | 28,740 | 660,116 |
| prediction-cache hit rate | 95.65% | 0.00% |
| forwarded-state reduction | 22.97x | 1.00x |

The important result is that both runs logically evaluate the same `660,116`
beam states, but the cache-on run only forwards `28,740` unique states through
the transformer. That is the mechanism behind the wall-clock speedup.


## Full Level1 Search Sweep Aggregate

Run files:

- `reports/eval_level1_search_beam_baseline.json`
- `reports/eval_level1_search_bf_256.json`
- `reports/eval_level1_search_bf_512.json`
- `reports/eval_level1_search_bf_1024.json`
- `reports/eval_level1_search_bf_fb_128.json`
- `reports/eval_level1_search_bf_fb_256.json`
- `reports/eval_level1_search_bf_fb_512.json`

All runs used the same checkpoint:

```text
models/planner_imitation_level0_multi4_convlog_e6.pt
```

The beam baseline used the current headline beam settings. The best-first runs
used policy/distance ranking over a global frontier instead of replanning a
fixed-width beam at every environment step. The fallback runs tried best-first
first, then used the beam baseline if best-first failed.

### Headline Results

| Run | Search mode | Solved | Wall time | Solves/min | Best-first solved | Fallback count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `level1_beam_baseline` | beam | 18/68 | 92.28s | 11.70 | 0 | 0 |
| `level1_bf_256` | best-first | 11/68 | 48.34s | 13.65 | 11 | 0 |
| `level1_bf_512` | best-first | 20/68 | 84.56s | 14.19 | 20 | 0 |
| `level1_bf_1024` | best-first | 32/68 | 139.43s | 13.77 | 32 | 0 |
| `level1_bf_fb_128` | best-first fallback | 20/68 | 100.75s | 11.91 | 5 | 63 |
| `level1_bf_fb_256` | best-first fallback | 22/68 | 112.24s | 11.76 | 11 | 57 |
| `level1_bf_fb_512` | best-first fallback | 25/68 | 145.82s | 10.29 | 20 | 48 |

### Search Work

| Run | Requested states | Unique forwards | Cache hit rate | Model time | Env step time | BF expanded | BF generated | Beam candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `level1_beam_baseline` | 660,116 | 28,740 | 95.65% | 49.75s | 17.02s | 0 | 0 | 687,932 |
| `level1_bf_256` | 41,889 | 25,258 | 39.70% | 37.99s | 1.12s | 16,288 | 29,669 | 0 |
| `level1_bf_512` | 77,822 | 45,751 | 41.21% | 70.04s | 1.99s | 29,573 | 53,694 | 0 |
| `level1_bf_1024` | 135,253 | 77,246 | 42.89% | 117.96s | 3.52s | 50,305 | 91,154 | 0 |
| `level1_bf_fb_128` | 663,472 | 35,603 | 94.63% | 58.63s | 16.76s | 8,502 | 15,568 | 669,149 |
| `level1_bf_fb_256` | 656,279 | 44,642 | 93.20% | 72.31s | 15.48s | 16,288 | 29,669 | 638,833 |
| `level1_bf_fb_512` | 648,244 | 61,873 | 90.46% | 98.98s | 17.06s | 29,573 | 53,694 | 594,578 |

### Interpretation

The best throughput setting in this sweep is `level1_bf_512`: it solves
`20/68`, beats the beam baseline by two puzzles, and is faster than the beam
baseline (`84.56s` vs `92.28s`). Its throughput is `14.19` solves/minute versus
`11.70` for beam.

The best solve-rate setting is `level1_bf_1024`: it solves `32/68` in
`139.43s`. That is `+14` solves over beam while still keeping throughput close
to the cached beam run. This is the strongest current Level1 result from the
same checkpoint.

The fallback variants are not competitive in this matrix. They add the cost of
best-first and then still fall back to beam on most puzzles, so they solve more
than beam but have worse solves/minute than pure best-first. In particular,
`bf_fb_512` solves `25/68` in `145.82s`, while pure `bf_1024` solves `32/68` in
`139.43s`.

Best-first has a lower prediction-cache hit rate than beam because it explores
more unique states instead of repeatedly replanning overlapping local beam
trees. That is expected and not necessarily bad: the lower hit rate buys better
global search coverage and many more solves.

One profiling caveat: `best_first_rank_time_s` currently includes the batched
candidate prediction calls used for scoring frontier states. It should be read
as "candidate scoring/ranking time", not pure sorting overhead.

### Puzzle Set Delta

Against the beam baseline, `bf_512` gains seven puzzles:

```text
Choose Wisely.pwp
Double Obstacle.pwp
Pick A Tool.pwp
Shape Of You.pwp
Single Obstacle.pwp
Take The Long Route.pwp
Tucked Away.pwp
```

It loses five beam-solved puzzles:

```text
A Perfect Fit.pwp
Building Blocks.pwp
Eyes On The Prize.pwp
Minefield.pwp
Mini Minefield.pwp
```

Against the beam baseline, `bf_1024` gains sixteen puzzles:

```text
A Tight Squeeze.pwp
At Crossroads.pwp
Cage Door.pwp
Choose Wisely.pwp
Clear The Way.pwp
Double Obstacle.pwp
Horizontal Channel.pwp
Pick A Tool.pwp
Shape Of You.pwp
Single Obstacle.pwp
Take Diversion.pwp
Take The Long Route.pwp
Triple Obstacle.pwp
Tucked Away.pwp
Vertical Separation.pwp
Youre In My Spot.pwp
```

It loses two beam-solved puzzles:

```text
Building Blocks.pwp
Eyes On The Prize.pwp
```

### Current Recommendation

Use pure best-first, not fallback, for the next main experiment:

- `best_first` with budget `512` for fastest quality-per-minute checks;
- `best_first` with budget `1024` for the best current Level1 solve rate;
- keep beam as the reproducibility baseline, not as the default optimized
  search path.


## Final Comparison Table

This section aggregates the missing full-run controls from:

- `reports/eval_level1_rgd_t10_full.json`
- `reports/eval_level1_base_repro_p1_s100_profiled.json`
- `reports/eval_level1_multi4_repro_p1_s100_profiled.json`
- `reports/eval_level1_search_bf_512_cache_off.json`
- `reports/eval_level1_search_bf_1024_cache_off.json`
- `reports/eval_level0_base_convlog_beam_cache_on.json`
- `reports/eval_level0_base_convlog_bf_512_cache_on.json`
- `reports/eval_level0_base_convlog_bf_1024_cache_on.json`
- `reports/repeats/eval_level1_repeat*_*.json`

Validation checks:

- RGD solved `68/68` on all three repeats with zero timeouts.
- Base-only linear and multi4 linear profiled reruns matched their original
  solved puzzle sets exactly.
- Beam cache on/off solved sets matched exactly.
- Best-first cache on/off solved sets matched exactly for budgets `512` and
  `1024`.
- The original conv/log beam report and the new profiled beam baseline both
  solved the same `18/68` puzzles.

### Primary Level1 Comparison

All learned closed-loop runs below use the full 68-puzzle Level1 benchmark,
`max_steps=100`, `beam_width=8`, `beam_depth=8`, `top_k=3`,
`repeat_penalty=1.0`, `beam_score=policy_distance`, and
`distance_weight=0.15` unless the row explicitly uses best-first search. The
reported learned repeat rows are means over three full reruns. RGD uses the
upstream C++ `N+RGD` planner with a `10s` per-puzzle timeout.

Important caveat: the RGD row here is Level1-only. It should not be compared
directly to the PushWorld paper's all-level benchmark curve, which covers all
223 benchmark puzzles across Levels 1-4. A quick `1s` per-puzzle probe over all
four local benchmark levels solved only `108/223` and timed out on `115`, while
Level1 alone solved `68/68`.

| System | Model / solver | Search and cache | Checkpoint cost | Level0 base | Level1 solved | Level1 time | Solves/min | Run basis | Role |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Upstream planner | `N+RGD` C++ planner | direct planner, no learned model | n/a | n/a | 68/68 | 6.71s +/- 0.08s | 607.87 | 3 repeats | Level1-only planner reference |
| Learned baseline | base-only linear/linear | beam, cache on | 40.4m | 181/200 | 2/68 | 40.67s | 2.95 | single profiled | old learned baseline |
| Learned baseline | multi4 linear/linear | beam, cache on | ~69m est.; 49.0m recorded resume | 178/200 | 8/68 | 50.97s | 9.42 | single profiled | stronger old baseline |
| Best checkpoint, raw eval | multi4 conv/log | beam, cache off | 29.5m | 188/200 | 18/68 | 1203.29s | 0.90 | single control | no-cache baseline |
| Best checkpoint | multi4 conv/log | beam, cache on | 29.5m | 188/200 | 18/68 | 93.28s +/- 2.52s | 11.59 | 3 repeats | cached beam baseline |
| Optimized throughput | multi4 conv/log | best-first `512`, cache on | 29.5m | 195/200 | 20/68 | 83.71s +/- 0.60s | 14.34 | 3 repeats | fastest learned setting |
| Best learned result | multi4 conv/log | best-first `1024`, cache on | 29.5m | 199/200 | 32/68 | 138.02s +/- 1.02s | 13.91 | 3 repeats | best learned solve rate |

Checkpoint cost is RGD expert generation plus model training time where
available. For multi4 linear, the recorded time is for the resume run; the
earlier analysis estimated about `69m` for the full 20-epoch run. For multi4
conv/log, the checkpoint cost is `60.49s` expert generation plus `1707.44s`
training.

The best learned system is now the same `multi4 conv/log` checkpoint with
best-first budget `1024`: `32/68` Level1 solves. It does not match the upstream
planner, but it is a large search-side improvement over cached beam
(`18/68 -> 32/68`) without additional model training.

The best experiment-throughput learned setting is best-first budget `512`:
`20/68` in `83.71s`, or `14.34` solves/minute. It slightly improves solve rate
over beam and is faster than the repeated cached beam baseline.

### Cache Isolation Controls

These rows isolate caching from search quality. In all three cases, cache on
and cache off solved the same puzzle set; the cache changes runtime, not
behavior.

| Search | Cache off | Cache on | Solved | Wall-clock speedup |
| --- | ---: | ---: | ---: | ---: |
| Beam `8x8` | 1203.29s | 78.09s | 18/68 | 15.41x |
| Best-first `512` | 120.36s | 84.56s | 20/68 | 1.42x |
| Best-first `1024` | 223.79s | 139.43s | 32/68 | 1.61x |

Beam benefits the most from prediction caching because it repeatedly replans
over overlapping local trees. Best-first explores more unique states, so it has
less repeated model work to remove.

### Level0 Sanity Check

Best-first did not only improve Level1. On the Level0 base held-out set, it
also improves solve rate relative to beam, though beam remains faster in raw
wall-clock time.

| Search | Cache | Level0 solved | Time | Solves/min |
| --- | --- | ---: | ---: | ---: |
| Beam `8x8` | on | 188/200 | 36.29s | 310.80 |
| Best-first `512` | on | 195/200 | 51.55s | 226.97 |
| Best-first `1024` | on | 199/200 | 54.78s | 217.94 |

### Final Interpretation

The optimization stack is best described as three distinct gains:

1. Model quality: moving from base linear to multi4 conv/log improved Level1
   from `2/68` to `18/68`.
2. Inference caching: cached beam preserved `18/68` but reduced the comparable
   no-cache beam runtime from `1203.29s` to `78.09s`.
3. Search quality: best-first budget `1024` raised the same checkpoint from
   `18/68` to `32/68`.

The fair headline for learned solving is therefore:

```text
multi4 conv/log checkpoint + prediction cache + best-first budget 1024
Level1: 32/68
Runtime: 138.02s +/- 1.02s over three full runs
Throughput: 13.91 solves/minute
Level0 base sanity: 199/200
```

The RGD row remains a Level1-only planner reference rather than a learned-policy
result or a reproduction of the paper's all-level curve. It solves all Level1
puzzles much faster, but it is the expert solver used to generate imitation
traces, not the neural policy being optimized.
