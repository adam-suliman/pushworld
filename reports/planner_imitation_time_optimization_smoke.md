# Planner Imitation Time Optimization Smoke

Date: 2026-05-16

Small repeatable subset:

- train dir: `data/level0/base/train`
- train puzzles: `2`
- examples/actions: `9`
- board: `7x7`
- training epochs: `0`
- final eval: skipped

## Cache Build

Command output: `reports/planner_imitation_cache_smoke_build.json`

| Metric | Value |
| --- | ---: |
| cache path | `data/cache/planner_imitation/smoke_timeopt` |
| RGD-solved puzzles | 2 |
| cache build time | 0.0206s |
| cache materialization time | 0.0059s |
| puzzle parse time | 0.0050s |
| base-state encode time | 0.0002s |
| cache size | 7.1 KB |

## Startup Comparison

| Run | RGD solve time | Dataset materialization | Cache load | Train wrapper time |
| --- | ---: | ---: | ---: | ---: |
| uncached | 0.0286s | 0.0047s | 0.0000s | 1.7952s |
| cache hit | 0.0000s | 0.0000s | 0.0275s | 1.3915s |

JSON outputs:

- `reports/planner_imitation_cache_smoke_uncached.json`
- `reports/planner_imitation_cache_smoke_cached.json`
- `reports/planner_imitation_cache_smoke_cached_e1.json`

The cache-hit path skips RGD and dataset materialization in the current run.
On this tiny subset, total startup is dominated by model/optimizer setup, so the
important signal is the structured phase breakdown rather than absolute speedup.

The one-epoch cache-hit smoke iterated the cached tensor dataset through the
DataLoader: `3` optimizer steps, `9` examples, `0.0104s` dataloader wait time,
and `0.5476s` forward/backward/update time.

## Eval Profile Smoke

Command output: `reports/planner_imitation_eval_profile_smoke.json`

One Level1 puzzle with `max_steps=2`, `beam_width=2`, `beam_depth=2`,
`top_k=2` produced:

| Metric | Value |
| --- | ---: |
| wall time | 0.4685s |
| encode cache entries | 8 |
| prediction cache entries | 8 |
| prediction cache hits | 9 |
| model forward batches | 4 |
| requested states | 17 |
| unique forward states | 8 |
