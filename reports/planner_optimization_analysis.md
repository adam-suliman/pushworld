# Planner Optimization Experiment Analysis

Date: 2026-05-16

Environment:

- Python: `3.13.3`
- PyTorch: `2.10.0.dev20251006+cu130`
- GPU: `NVIDIA GeForce RTX 2060`
- Expert planner: upstream RGD planner built at `external/pushworld/cpp/build/bin/run_planner.exe`
- Data: `level0` tasks

## Data

The real Level 0 archive is present and was used. It contains 7 variants:
`base`, `all`, `shapes`, `obstacles`, `goals`, `size`, and `walls`.
Each variant has 2000 train puzzles and 200 test puzzles.

Level 1 evaluation used the upstream benchmark directory:
`external/pushworld/benchmark/puzzles/level1` with 68 puzzles.

All closed-loop evaluations below use:

- `max_steps=100`
- `beam_width=8`
- `beam_depth=8`
- `top_k=3`
- `beam_score=policy_distance`
- `distance_weight=0.15`

## Main Results

| Run | Train data | Model config | Epochs | Train time | Level0 base, no repeat penalty | Level0 base, repeat penalty | Level1, repeat penalty |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base-only reproduction | `base` train, 2000 puzzles | linear stem, linear distance | 60 | 2414.8s | 162/200 | 181/200 | 2/68 |
| Multi4 linear | `base+all+shapes+obstacles`, 8000 puzzles | linear stem, linear distance | 20 | 2881.1s for resume epochs 7-20 | 154/200 | 178/200 | 8/68 |
| Multi4 conv/log | `base+all+shapes+obstacles`, 8000 puzzles | conv stem, log distance, 1% dropout | 6 | 1707.4s | not run | 188/200 | 18/68 |

The Wheeler-style conv/log multi4 run did better despite only 6 epochs:
`188/200` on Level0 base and `18/68` on Level1 with repeat penalty.

## Compute And Speed

| Run | Examples | Batch size | Batches/epoch | Effective train speed |
| --- | ---: | ---: | ---: | ---: |
| Base-only linear, 60 epochs | 19,917 | 128 | 156 | about 3.9 batches/s |
| Multi4 linear, resume epochs 7-20 | 95,416 | 64 | 1491 | about 7.2 batches/s |
| Multi4 conv/log, 6 epochs | 95,416 | 64 | 1491 | about 5.2 batches/s |

The conv/log model is slower per batch than the linear-stem model on the same
8000-puzzle multi4 data, roughly `28%` lower batch throughput in these runs.
The loss values are not directly comparable across linear and log distance
targets.

## Experiment-Time Optimization View

The optimization target for this work is not raw batches per second. It is
time-to-useful-result: how much wall-clock training time is needed before a
checkpoint reaches a good closed-loop solve rate.

By that metric, the conv/log change was impactful.

| Run | Training wall-clock used | Level0 base p1 | Level1 p1 | Level1 solves per training minute |
| --- | ---: | ---: | ---: | ---: |
| Multi4 linear | 48.0 min recorded for resume epochs 7-20; about 69 min estimated for all 20 epochs | 178/200 | 8/68 | about 0.12-0.17 |
| Multi4 conv/log | 28.5 min for all 6 epochs | 188/200 | 18/68 | about 0.63 |

So this was not a throughput optimization: each conv/log epoch is more
expensive. The win is that it reached a better checkpoint much earlier. Even
using the conservative denominator of only the recorded linear resume time,
conv/log produced about `3.8x` more Level1 solves per training minute. Compared
with the estimated full 20-epoch linear run, the ratio is about `5.4x`.

The eval rollout wall-clock did not get cheaper in the same way. Level1
closed-loop evaluation remains slow because unsolved puzzles often consume the
full 100-step beam budget. The next wall-clock optimization should therefore
target rollout/evaluation: batched beam candidate scoring, stronger closed-list
pruning, and fewer repeated-state expansions.

## Repeat Penalty Effect

| Run | Split | No penalty | Repeat penalty | Change |
| --- | --- | ---: | ---: | ---: |
| Base-only linear | Level0 base | 162/200 | 181/200 | +19 solves |
| Multi4 linear | Level0 base | 154/200 | 178/200 | +24 solves |
| Multi4 linear | Level1 | 5/68 | 8/68 | +3 solves |

Repeat penalty is not a small cleanup. It is a major search-side intervention:
on Level1 for multi4 linear, repeated states dropped from 5753 to 3466 and
solves increased from 5 to 8. For the conv/log multi4 p1 run, repeated states
were lower again at 1876 total on Level1.

## Wheeler Post Takeaways

 Several ideas from Tim Wheeler's Sokoban transformer post that came to be useful to this PushWorld setup:

- Treat the board as spatial data and use a convolutional board encoder before
  the transformer.
- Predict a discrete log-scaled remaining-step target rather than a raw
  continuous distance.
- Use a small amount of dropout.
- Keep beam search for closed-loop solving, and use the policy plus value or
  distance estimates to rank candidates.
- Use symmetry augmentation when the domain permits it.

Source: https://timallanwheeler.com/blog/2024/06/01/a-transformer-sokoban-policy/

The new multi4 conv/log experiment supports these choices. The model-side
change is slower per batch, but the solve-rate gain is much larger than the
compute cost in this run.

## Beam Search Recommendation

Do not replace beam search wholesale yet. The measured data says the current
beam becomes much stronger when:

- repeated states are penalized,
- candidate states are ranked by policy plus distance/value,
- the policy model is stronger.

The next search improvements should be incremental:

- Batch model scoring across all beam frontier candidates instead of scoring
  many states one by one.
- Keep a per-puzzle closed list or stronger repeat-aware score so the beam
  stops spending budget on cycles.
- Cache encoded states aggressively, keyed by board tensor/state hash.
- Tune `distance_weight`, repeat penalty, and optional length normalization on
  Level0 validation before touching Level1.
- Consider a value/nsteps-head score as a tiebreaker for beams with similar
  action likelihood.

## Conclusion

The earlier bad results were caused by using inadequate smoke data and not the
real Level0 RGD trajectory setup. With the real data, the base-only experiment
matches the friend's Level0 result. The remaining gap was not just "more
training": the linear multi4 run trained longer but underperformed. The
conv/log model-side configuration is the important improvement found here,
raising repeat-penalty results to `188/200` on Level0 base and `18/68` on
Level1.

Recommended next run: continue the `multi4_convlog` checkpoint beyond 6 epochs
with the same eval settings. It already exceeds the target after 6 epochs, so
longer training is the most direct test of whether the Level1 gains keep
scaling.
