# Used Methods And Definitions




## Method Definitions

**PushWorld puzzle**

A grid puzzle loaded from a `.pwp` file through the upstream `pushworld`
package. The model sees a padded board tensor and chooses one of four actions:
left, right, up, or down (`L`, `R`, `U`, `D`).

**Level 0 data**

Generated PushWorld puzzles in `data/level0`. The local archive `level0.zip`
contains the variants `base`, `all`, `shapes`, `obstacles`, `goals`, `size`,
and `walls`. Each variant has 2000 train puzzles and 200 test puzzles.

**Level 1 data**

The upstream hand-authored benchmark puzzles in
`external/pushworld/benchmark/puzzles/level1`. The split used here has 68
puzzles.

**RGD / N+RGD planner**

The expert solver used to produce imitation trajectories. We ran the upstream
C++ executable as:

```powershell
external\pushworld\cpp\build\bin\run_planner.exe N+RGD <puzzle.pwp>
```

The planner emits a string of expert actions. We validate that the emitted plan
solves the puzzle before turning it into training examples.

**Imitation trajectory**

For a plan of length `T`, each step becomes one supervised example:

- input: current board state;
- action target: the next expert action;
- distance target: remaining plan steps, `T - step_index`.


**Board encoding**

Each state is encoded as a 7-channel `float32` board tensor:

- channel 0: static walls;
- channel 1: agent walls;
- channel 2: controlled object / agent object;
- channel 3: goal-associated movable objects;
- channel 4: other movable objects;
- channel 5: goal cells for goal-associated objects;
- channel 6: already satisfied goal cells.

**Transformer policy**

The policy is a board transformer with:

- a board stem that converts board cells to tokens;
- learned positional embeddings plus a learned CLS token;
- a `TransformerEncoder`;
- an action head over `L/R/U/D`;
- an auxiliary remaining-distance head.

The training loss is:

```text
cross_entropy(action_logits, expert_action)
+ distance_loss_weight * cross_entropy(distance_logits, remaining_distance_target)
```

In the main runs, `distance_loss_weight=0.2`.

**Linear board stem**

The original policy stem. Each cell's 7-channel feature vector is projected
independently with a linear layer before the transformer.

**Convolutional board stem**

The Wheeler-inspired policy stem. A small local encoder is applied before
tokenization:

```text
Conv2d(7 -> d_model, 3x3, padding=1)
GELU
Conv2d(d_model -> d_model, 3x3, padding=1)
GELU
```

This lets the model see local spatial patterns before the global transformer
attention layers.

**Linear distance target**

The original auxiliary target. Remaining steps are exact integer classes:
`0..max_steps`, so `max_steps=100` gives 101 distance classes.

**Log distance target**

The optimized auxiliary target. Remaining steps are converted to compact
log-spaced classes:

```text
round(log(remaining_steps + 1))
```

For `max_steps=100`, this uses 6 bins instead of 101. During beam scoring, bins
are mapped back to approximate step values with `expm1(bin)`.

**Beam rollout**

Closed-loop solving does not greedily take one action. At each environment
step, the policy does a short model-guided beam lookahead:

- keep `beam_width` partial candidate paths;
- expand each path up to `beam_depth`;
- consider the model's `top_k` actions per state;
- execute only the first action of the best beam path;
- repeat until solved or `max_steps` is reached.

The main evaluation setting was `beam_width=8`, `beam_depth=8`, `top_k=3`,
`max_steps=100`.

**Beam score**

The previous effective score was hard-coded as policy cost plus a small
distance-head term. It is now configurable:

- `policy`: rank by cumulative negative log policy probability;
- `distance`: rank mainly by predicted remaining distance;
- `policy_distance`: rank by policy cost plus weighted predicted distance.

The main experiments used:

```text
beam_score = policy_distance
distance_weight = 0.15
beam_length_normalization = 0.0
```

**Repeat-state penalty**

A search-time cost added when a candidate revisits a state already seen in the
current rollout. This was already present in the branch; the key experimental
finding is that enabling it is critical. The main repeat-penalty evaluations
used `repeat_penalty=1.0`.

**Multi4 training set**

The 8000-map Level 0 training set formed by combining:

- `data/level0/base/train`;
- `data/level0/all/train`;
- `data/level0/shapes/train`;
- `data/level0/obstacles/train`.


**Experiment-time optimization**

The optimization target in these experiments is wall-clock time to a useful
checkpoint, not raw batches per second. A change can be worthwhile even if each
batch is slower, provided it reaches a higher solve rate in fewer epochs or less
total training time. Under this definition, the conv/log run was an optimization:
it trained more slowly per batch than the linear multi4 model, but reached
`188/200` Level0 base and `18/68` Level1 after 28.5 minutes of training.

## What Changed From The Original Imitation-Learning Branch

The core method did not change from imitation learning to another learning
paradigm. It is still supervised behavior cloning on RGD trajectories with a
policy head and an auxiliary remaining-distance head.

The differences are more specific:

| Area | Original branch behavior | Current experiment behavior |
| --- | --- | --- |
| Baseline model | Linear per-cell board projection; exact linear remaining-step classes. | Reproduced this explicitly for the base-only and linear multi4 baselines. |
| Optimized model | No conv/log model-side options in the original branch code. | Added `--encoder-stem conv`, `--distance-target log`, `--distance-bins`, and `--dropout`. Defaults now use conv/log/dropout. |
| Distance head | Always `max_steps + 1` exact classes. | Supports either exact linear classes or compact log-spaced classes. |
| Beam ranking | Hard-coded policy-plus-distance behavior. | Added `--beam-score`, `--distance-weight`, and `--beam-length-normalization` for controlled beam ablations. |
| Evaluation loader | Assumed linear-stem checkpoints. | Detects/loads both linear and conv-stem checkpoints and uses the checkpoint's distance-target mode by default. |
| Long training workflow | Final rollout eval was tied to training completion. | Added `--skip-final-eval` so long training can save immediately and eval can be run separately. |
| Demo controls | Exposed basic rollout controls. | Added beam-score, distance-weight, and length-normalization controls. |
| Benchmarking | No local compute/speed ablation script for these model-side options. | Added `scripts/run_planner_optimization_experiments.py` for controlled CUDA/CPU and model ablations. |
| Tests | No tests for the new optimization helpers. | Added tests for log distance targets, distance bin decoding, beam scoring, and conv-stem forward shapes. |

## What We Did In The Finished Experiments

**Base-only reproduction**

Trained on all 2000 `data/level0/base/train` puzzles with the original-style
linear model:

- `encoder_stem=linear`;
- `distance_target=linear`;
- `dropout=0.0`;
- 60 epochs;
- RGD expert traces generated locally.

This reproduced the reported Level 0 baseline:

- `162/200` Level0 base without repeat penalty;
- `181/200` Level0 base with repeat penalty;
- `2/68` Level1 with repeat penalty.

**Linear multi4 run**

Trained on all 8000 `base+all+shapes+obstacles` training puzzles while keeping
the original-style model:

- `encoder_stem=linear`;
- `distance_target=linear`;
- `dropout=0.0`;
- 20 epochs total.

Results:

- `154/200` Level0 base without repeat penalty;
- `178/200` Level0 base with repeat penalty;
- `8/68` Level1 with repeat penalty.

This showed that simply adding the multi4 data with the original model config
was not enough.

**Conv/log multi4 run**

Trained on the same 8000 multi4 puzzles with the Wheeler-inspired model-side
changes:

- `encoder_stem=conv`;
- `distance_target=log`;
- `dropout=0.01`;
- 6 epochs.

Results:

- `188/200` Level0 base with repeat penalty;
- `18/68` Level1 with repeat penalty.

This is the main method-side improvement found so far.

## Interpretation


The useful change is a better policy/value model:
local convolution before transformer attention, compact log-scaled distance
targets, and small dropout. Beam search still matters, especially the existing
repeat-state penalty, but the strongest gains came when the model itself became
better.
