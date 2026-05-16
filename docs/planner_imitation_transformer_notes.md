# Planner-Imitation Transformer Notes

This branch is a stronger-baseline direction for PushWorld. The goal is not
environment speed this week. The goal is to train a policy from heuristic
planner trajectories and compare its closed-loop success against PPO/DQN.

## Current Prototype

Script:

- `scripts/train_planner_imitation_smoke.py`
- `scripts/train_planner_imitation_v2.py` for in-memory Level 0 symmetry
  augmentation.
- `scripts/build_level1_augmented_dataset.py` for cached, split-safe Level 1
  augmentation.

Pipeline:

1. select PushWorld puzzles;
2. solve train puzzles with the official C++ `N+RGD` planner;
3. replay each plan into `(state, next_action, remaining_steps)` examples;
4. encode each state as compact symbolic planes;
5. train a small board-transformer policy;
6. evaluate with receding-horizon beam search.

Model:

- input: 7 symbolic planes;
- architecture: per-cell projection + transformer encoder + CLS token;
- heads:
  - next primitive action: `left/right/up/down`;
  - remaining expert-plan steps, used as a beam-search value proxy.

## Smoke Results

Initial greedy one-step imitation was too brittle. With 100 Level 0 train
puzzles, greedy rollout solved only a few puzzles and often collapsed into
repeated actions.

After adding the remaining-steps head and beam search:

| Train set | Examples | Train time | Train eval | Held-out Level 0 | Level 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 Level 0 puzzles | 26 | 5.47s | 5/5 | 0/10 | 0/5 |
| 100 Level 0 puzzles | 941 | 75.3s | 96/100 | 29/50 | 0/5 |

Interpretation:

- The policy can learn planner demonstrations quickly.
- Held-out Level 0 success appears with only 100 training puzzles.
- Level 1 zero-shot is still not solved by this small Level 0-only prototype.
- Beam search matters; plain greedy action classification is not enough.

## Full Base Run Results

Artifact:

- report: `reports/planner_imitation_full_level0_base_small.json`;
- checkpoint: `models/planner_imitation_level0_base_small.pt`;
- TensorBoard: `runs/planner_imitation_level0_base_small`.

Configuration:

- train split: all `data/level0/base/train` puzzles (`2000` puzzles);
- training examples from RGD expert traces: `19902` states;
- model: `d_model=64`, `nhead=4`, `layers=1`, AMP enabled;
- rollout: `8x8` beam, `top_k=3`;
- train-set evaluation skipped.

Results:

| Split | Solved | Total | Success |
| --- | ---: | ---: | ---: |
| Level 0 base test | 163 | 200 | 81.5% |
| Level 1 zero-shot | 3 | 68 | 4.4% |

Solved Level 1 puzzles:

- `Get Out Of My Spot.pwp`;
- `Small Wins.pwp`;
- `Victory Road.pwp`.

Timing:

- RGD trace generation: `23.5s`;
- training: `6680.0s` (`111.3min`, about `1h51m`);
- observed on the laptop run: about `1.7s/batch`, roughly `8min/epoch`.

Important caveat: this training run used the training script's default
`--max-steps 100` for final rollout evaluation. The expert traces for the
larger multi-variant run include plans up to `167` actions, so final reported
numbers should be regenerated with the eval-only script and
`--max-steps 200`.

## Current Optimizations

Already implemented:

- natural numeric puzzle ordering, so small smoke runs use puzzles `0..N`;
- `tqdm` progress bars for expert solving, training, and evaluation;
- optional quick held-out evaluation during training via `--eval-every` and
  `--eval-puzzles`;
- nested batch progress bars inside each epoch;
- TensorBoard batch-loss logging via `--log-every-batches`;
- explicit `--all-train`, `--all-test`, and `--all-level1` flags for full
  split runs;
- repeated `--train-dir` and `--test-dir` arguments for multi-variant Level 0
  training;
- optional `--skip-train-eval` for full runs, because evaluating 2000 train
  puzzles is not needed for the headline held-out result;
- cached state encodings during beam evaluation;
- duplicate beam-state pruning;
- no-op action pruning inside beam expansion;
- repeat-aware rollout scoring via `--repeat-penalty`;
- batched neural scoring of beam states and leaf states;
- shared rollout implementation in `scripts/planner_imitation_rollout.py`, used by
  both training-time eval and standalone eval scripts;
- compact JSON summary output via `--output`;
- model checkpoint output via `--model-output`.
- automatic per-epoch checkpoints for `scripts/train_planner_imitation_v2.py`:
  with `--model-output models/foo.pt`, epoch checkpoints are written to
  `models/foo_epochs/epoch_001.pt`, `epoch_002.pt`, etc.;
- graceful interrupt handling in `scripts/train_planner_imitation_v2.py`: a
  `KeyboardInterrupt` saves `interrupted_epoch_XXX.pt` with model, optimizer,
  scaler, epoch, and global step, then skips final eval;
- resume support via `--resume-checkpoint path/to/epoch_XXX.pt`;
- eval-only checkpoint loading via `scripts/eval_planner_imitation.py`, so
  rollout limits and beam settings can be compared without retraining.
- v2 in-memory Level 0 symmetry augmentation, using the 8 rotation/flip
  transforms without writing augmented `.pwp` files.
- cached Level 1 augmentation builder that splits source symmetry groups before
  generating variants, preventing train/test leakage through rotated or mirrored
  copies.
- Wheeler-inspired v2 model defaults:
  - `--encoder-stem conv` adds a local convolutional board encoder before the
    transformer tokens, matching the blog post's image-like board encoder idea;
  - `--distance-target log` trains the remaining-steps head on compact
    log-spaced bins instead of sparse exact step counts;
  - `--dropout 0.01` mirrors the small regularization used in the post.
- Beam ranking is now configurable through `--beam-score`, `--distance-weight`,
  and `--beam-length-normalization`. The default `policy_distance` with
  `--distance-weight 0.15` preserves the previous policy-plus-value behavior.

Repeat penalty details:

- default `--repeat-penalty 0.0` preserves the old behavior;
- positive values add cost to beam candidates that revisit a state already seen
  during the current rollout;
- greedy rollout uses the same signal by preferring the best non-repeating
  legal action when one exists;
- the penalty is a rollout/evaluation option, not a training loss change.

Suggested first comparison values are `0.0`, `1.0`, and `2.0`. Beam scores are
negative log-probability plus a small value-head distance term, so `2.0` is a
meaningful but non-infinite penalty.

Suggested first beam-ranking ablation for a fixed checkpoint:

```bash
CKPT=models/planner_imitation_all_level0_l1cache_lazyaug_seed1.pt
for score in policy policy_distance distance; do
  uv run python -u scripts/eval_planner_imitation.py \
    --checkpoint "$CKPT" \
    --eval-dir data/level0/base/test \
    --split-name "level0_base_test_${score}" \
    --all-eval \
    --max-steps 200 \
    --beam-width 8 \
    --beam-depth 8 \
    --top-k 3 \
    --repeat-penalty 0 \
    --beam-score "$score" \
    --distance-weight 0.15 \
    --beam-length-normalization 0.0 \
    --output "reports/beam_score_${score}.json"
done
```

Keep `--beam-width`, `--beam-depth`, `--top-k`, `--repeat-penalty`, and
`--max-steps` fixed when measuring whether the alternative ranking preserves
the solved percentage.

The current beam evaluator is still Python-heavy. The cache/pruning changes
should reduce repeated encoding and wasted beam branches, but the exact speedup
needs to be measured against the previous script version or against a
`--beam-width 1 --beam-depth 1` greedy baseline.

## Speedup Log

Fill this table as we run comparable commands:

| Date | Command / setting | Train puzzles | Eval split | Beam | Eval time | Success |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 2026-05-01 | pre-cache prototype | 100 | Level 0 test 50 | 8x8 top-3 | not isolated | 29/50 |
| 2026-05-01 | cached/pruned beam quick check | 20 | Level 0 test 10 | 4x4 top-2 | 1.62s | 0/10 |

For a clean speedup number, run the same checkpoint through
`scripts/eval_planner_imitation.py` with only the beam settings changed.

## Planned Optimizations

Near-term:

- cache expert trajectories and tensors on disk, so repeated training runs do
  not call RGD or re-encode planner states;
- split train/validation and early-stop on held-out Level 0 success;
- save the best checkpoint by periodic held-out evaluation, not only the final
  epoch checkpoint;
- batch larger beam frontier calls more aggressively;
- compare greedy, `4x4`, `8x8`, and `16x8` beam settings.

Model-side:

- add recent-state history, not only the current state;
- try a CNN-only baseline vs transformer encoder;
- add goal/object identity channels if Level 1 transfer remains weak;
- train on all Level 0 variants, not only `base`;
- train with Level 0 symmetry augmentation to reduce orientation overfitting;
- optionally include modern heuristic-solver Level 1 traces as a second-stage
  curriculum.

Wheeler-style systems optimizations:

- Wheeler reports about `32x` faster teacher-forced training after moving
  Sokoban policy training to GPU;
- Wheeler reports about `60x` faster rollout batches after moving rollout state
  advancement to GPU;
- the analogous PushWorld optimization would move batched rollout/beam state
  advancement out of Python and into a vectorized/GPU representation.

For this project, that is a later optimization. The immediate bottleneck is
getting a strong held-out baseline; after that, optimize beam evaluation time.

## Full Run Command

Train on all Level 0 base train puzzles, skip train-set eval, evaluate all
held-out Level 0 base test puzzles and all official Level 1 puzzles:

```bash
uv run python -u scripts/train_planner_imitation_smoke.py \
  --all-train \
  --all-test \
  --all-level1 \
  --epochs 60 \
  --batch-size 128 \
  --lr 0.001 \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --seed 1 \
  --eval-every 10 \
  --eval-puzzles 50 \
  --log-every-batches 10 \
  --skip-train-eval \
  --tensorboard-log runs/planner_imitation_level0_base \
  --output reports/planner_imitation_full_level0_base.json \
  --model-output models/planner_imitation_level0_base.pt
```

Watch it with:

```bash
uv run tensorboard --logdir runs/planner_imitation_level0_base
```

Expected rough runtime on GTX 1650:

- expert trace generation: under a minute for base-only in the observed run;
- training: about `1.5-2h` for `10` epochs with batch size `64`, `d_model=64`,
  `layers=1`, and AMP;
- held-out Level 0 + Level 1 beam eval: depends heavily on `--max-steps` and
  beam settings, so use the eval-only script for final measurements.

## Multi-Variant Level 0 Command

Train on four Level 0 variants (`base`, `all`, `shapes`, `obstacles`) and
evaluate on the corresponding held-out Level 0 splits plus all Level 1 puzzles:

```bash
uv run python -u scripts/train_planner_imitation_smoke.py \
  --train-dir data/level0/base/train \
  --train-dir data/level0/all/train \
  --train-dir data/level0/shapes/train \
  --train-dir data/level0/obstacles/train \
  --test-dir data/level0/base/test \
  --test-dir data/level0/all/test \
  --test-dir data/level0/shapes/test \
  --test-dir data/level0/obstacles/test \
  --all-train \
  --all-test \
  --all-level1 \
  --epochs 6 \
  --batch-size 64 \
  --lr 0.001 \
  --d-model 64 \
  --nhead 4 \
  --layers 1 \
  --amp \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --seed 1 \
  --eval-every 1 \
  --eval-puzzles 100 \
  --log-every-batches 10 \
  --skip-train-eval \
  --tensorboard-log runs/planner_imitation_level0_multi4 \
  --output reports/planner_imitation_level0_multi4.json \
  --model-output models/planner_imitation_level0_multi4.pt
```

This uses `8000` train puzzles and `800` held-out Level 0 puzzles. At the
observed full-base speed, expect roughly `4x` the per-epoch time of base-only.

## Level 0 Symmetry-Augmented V2

`scripts/train_planner_imitation_v2.py` keeps the RGD trace count unchanged and
expands examples in memory. For each expert state/action pair, it adds the 8
dihedral grid transforms and remaps the action label consistently.

Smoke command:

```bash
uv run python -u scripts/train_planner_imitation_v2.py \
  --train-puzzles 3 \
  --test-puzzles 2 \
  --level1-puzzles 1 \
  --epochs 1 \
  --batch-size 16 \
  --lr 0.001 \
  --d-model 32 \
  --nhead 4 \
  --layers 1 \
  --max-steps 30 \
  --beam-width 1 \
  --beam-depth 1 \
  --top-k 1 \
  --seed 1 \
  --eval-every 0 \
  --level0-symmetry-augment \
  --skip-train-eval
```

Smoke result:

- expert traces: `3` puzzles, `18` base states;
- augmented examples: `144`;
- augmentation factor: `8.0x`;
- train/eval path completed without writing generated puzzle files.

Full comparable experiment should use the same settings as the current
multi-variant run, changing only the script name and adding
`--level0-symmetry-augment`.

Training scripts now expose `--seed`; keep it in every reported command.
`scripts/build_level1_augmented_dataset.py` also exposes `--seed` for source
split assignment and advanced proposal sampling. The eval-only script is
deterministic for a fixed checkpoint and puzzle list, so it does not need a seed
for ordinary reporting.

## All-Level-0 Quick Run With Level 1 Cache

Training on all `14k` Level 0 train maps with fully materialized `8x` symmetry
augmentation is not a quick run. The v2 trainer instead uses lazy on-the-fly
Level 0 augmentation: each dataset item corresponds to one expert state, and a
random symmetry transform is chosen when the item is loaded.

Important implementation detail:

- `--level0-symmetry-augment` only augments paths under `data/level0`;
- cached Level 1 train paths are left as-is;
- dataset size stays close to the number of expert states instead of multiplying
  by `8x`;
- Level 0 transforms are re-sampled on access, so later epochs can see different
  orientations.

Train command:

```bash
uv run python -u scripts/train_planner_imitation_v2.py \
  --train-dir data/level0/base/train \
  --train-dir data/level0/all/train \
  --train-dir data/level0/goals/train \
  --train-dir data/level0/obstacles/train \
  --train-dir data/level0/shapes/train \
  --train-dir data/level0/size/train \
  --train-dir data/level0/walls/train \
  --train-dir data/augmented/level1_verified_advanced_seed1_train50_p8/train \
  --all-train \
  --test-puzzles 0 \
  --level1-puzzles 0 \
  --epochs 1 \
  --batch-size 64 \
  --lr 0.001 \
  --d-model 64 \
  --nhead 4 \
  --layers 1 \
  --amp \
  --max-steps 200 \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --eval-every 0 \
  --log-every-batches 10 \
  --skip-train-eval \
  --level0-symmetry-augment \
  --planner-workers 6 \
  --seed 1 \
  --tensorboard-log runs/planner_imitation_all_level0_l1cache_lazyaug_seed1 \
  --output reports/planner_imitation_all_level0_l1cache_lazyaug_seed1.json \
  --model-output models/planner_imitation_all_level0_l1cache_lazyaug_seed1.pt
```

Resume from the latest epoch checkpoint by keeping the same training command
and adding `--resume-checkpoint`. `--epochs` is the final target epoch, not the
number of extra epochs. For example, to continue after epoch 3 up to epoch 10:

```bash
uv run python -u scripts/train_planner_imitation_v2.py \
  --train-dir data/level0/base/train \
  --train-dir data/level0/all/train \
  --train-dir data/level0/goals/train \
  --train-dir data/level0/obstacles/train \
  --train-dir data/level0/shapes/train \
  --train-dir data/level0/size/train \
  --train-dir data/level0/walls/train \
  --train-dir data/augmented/level1_verified_advanced_seed1_train50_p8/train \
  --all-train \
  --test-puzzles 0 \
  --level1-puzzles 0 \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.001 \
  --d-model 64 \
  --nhead 4 \
  --layers 1 \
  --amp \
  --max-steps 200 \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --eval-every 0 \
  --log-every-batches 10 \
  --skip-train-eval \
  --level0-symmetry-augment \
  --planner-workers 6 \
  --seed 1 \
  --resume-checkpoint models/planner_imitation_all_level0_l1cache_lazyaug_seed1_epochs/epoch_003.pt \
  --tensorboard-log runs/planner_imitation_all_level0_l1cache_lazyaug_seed1 \
  --output reports/planner_imitation_all_level0_l1cache_lazyaug_seed1.json \
  --model-output models/planner_imitation_all_level0_l1cache_lazyaug_seed1.pt
```

Split-aware original Level 1 eval:

```bash
uv run python -u scripts/eval_level1_original_by_aug_split.py \
  --checkpoint models/planner_imitation_all_level0_l1cache_lazyaug_seed1.pt \
  --manifest data/augmented/level1_verified_advanced_seed1_train50_p8/manifest.json \
  --max-steps 200 \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --output reports/eval_level1_original_by_aug_split_all_level0_l1cache_lazyaug_seed1.json
```

This reports original Level 1 success separately for source groups that were in
the Level 1 augmented train split and source groups that were held out.

Repeat-penalty comparison for the same checkpoint:

```bash
for penalty in 0 1 2; do
  uv run python -u scripts/eval_level1_original_by_aug_split.py \
    --checkpoint models/planner_imitation_all_level0_l1cache_lazyaug_seed1.pt \
    --manifest data/augmented/level1_verified_advanced_seed1_train50_p8/manifest.json \
    --max-steps 200 \
    --beam-width 8 \
    --beam-depth 8 \
    --top-k 3 \
    --repeat-penalty "$penalty" \
    --output "reports/eval_level1_original_by_aug_split_all_level0_l1cache_lazyaug_seed1_repeat${penalty}.json"
done
```

Use the same `--repeat-penalty` flag with `scripts/eval_planner_imitation.py`
for Level 0 table-style runs.

The Streamlit demo uses the same shared rollout code and exposes the same
repeat penalty control, so demo behavior should match eval-script behavior for
the same checkpoint and rollout settings.

## Level 1 Cached Augmentation

`scripts/build_level1_augmented_dataset.py` builds a cached Level 1 dataset.
Unlike Level 0 augmentation, this writes `.pwp` files because Level 1 has only
`68` originals and advanced perturbations need RGD verification.

Leakage rule:

- group source originals by canonical symmetry orbit;
- split these source groups into train/val/test;
- generate all variants only after the split;
- every generated variant inherits the split of its source group.

This prevents a rotated, mirrored, or transposed copy of a held-out Level 1
source puzzle from appearing in train.

Generation order:

- start from the original source layout or an advanced perturbed layout;
- generate the 8 rotation/flip variants from that base layout;
- parse and RGD-verify the base layout once;
- keep the base layout only if it solves under the plan-length limit;
- emit the 8 rotation/flip variants without extra RGD checks.

Implemented advanced proposal methods:

- `wall_toggle`: add/remove one wall on an empty/wall cell;
- `remove_movable`: remove one non-goal movable obstacle;
- `add_movable`: add a new one- or two-cell non-goal movable obstacle;
- `add_goal`: add a secondary goal for an existing non-goal movable obstacle;
- `move_goal_shift`: move an existing goal shape within a small radius;
- `move_goal_random`: move an existing goal shape to a random valid empty
  placement.

Smoke command:

```bash
uv run python -u scripts/build_level1_augmented_dataset.py \
  --limit-originals 2 \
  --train-ratio 0.5 \
  --seed 11 \
  --advanced-per-original 2 \
  --advanced-max-attempts-per-original 20 \
  --advanced-methods wall_toggle,remove_movable,add_movable,add_goal,move_goal_shift,move_goal_random \
  --max-new-movable-cells 2 \
  --goal-shift-radius 2 \
  --planner-time-limit 5 \
  --max-plan-len 200 \
  --output-dir /tmp/pushworld_level1_aug_advanced_smoke_20260501 \
  --overwrite
```

Smoke result:

- source puzzles: `2`;
- source groups: `2`;
- generated variants: `48`;
- original symmetry variants: `16`;
- accepted advanced variants: `32`;
- accepted advanced kinds: `remove_movable`, `move_goal_shift`,
  `add_movable`, `move_goal_random`;
- rejected base layouts: `4`;
- split counts: `24 train`, `24 test`;
- manifest audit: no source group appears in more than one split.

Full basic-cache command:

```bash
uv run python -u scripts/build_level1_augmented_dataset.py \
  --train-ratio 0.8 \
  --val-ratio 0.0 \
  --seed 1 \
  --planner-time-limit 10 \
  --max-plan-len 200 \
  --output-dir data/augmented/level1_verified_symmetry
```

Full cache with small verified perturbation budget:

```bash
uv run python -u scripts/build_level1_augmented_dataset.py \
  --train-ratio 0.8 \
  --val-ratio 0.0 \
  --seed 1 \
  --planner-time-limit 10 \
  --max-plan-len 200 \
  --advanced-per-original 4 \
  --advanced-max-attempts-per-original 40 \
  --advanced-methods wall_toggle,remove_movable,add_movable,add_goal,move_goal_shift,move_goal_random \
  --max-new-movable-cells 2 \
  --goal-shift-radius 2 \
  --output-dir data/augmented/level1_verified_symmetry_advanced
```

The advanced proposals are intentionally generators, not trusted labels. The
final dataset contains only variants whose base layout parses and solves with
RGD under the configured plan-length limit. Advanced augmentations are currently
independent by default: each accepted candidate applies one advanced generator
to the original source layout, then symmetry expansion. Use
`--advanced-stack-depth 2` or higher to apply multiple advanced generators
sequentially before the single RGD base-layout check.

Stacked smoke command:

```bash
uv run python -u scripts/build_level1_augmented_dataset.py \
  --limit-originals 2 \
  --train-ratio 0.5 \
  --seed 13 \
  --advanced-per-original 2 \
  --advanced-max-attempts-per-original 30 \
  --advanced-stack-depth 2 \
  --advanced-methods wall_toggle,remove_movable,add_movable,add_goal,move_goal_shift,move_goal_random \
  --max-new-movable-cells 2 \
  --goal-shift-radius 2 \
  --planner-time-limit 5 \
  --max-plan-len 200 \
  --output-dir /tmp/pushworld_level1_aug_stack_smoke_20260501 \
  --overwrite
```

Stacked smoke result:

- generated variants: `48`;
- rejected base layouts: `8`;
- accepted stacked kinds included `add_movable+move_goal_random`,
  `remove_movable+wall_toggle`, `move_goal_random+add_movable`, and
  `wall_toggle+add_movable`;
- manifest audit: no source group appears in more than one split.

## Table-2-Style Evaluation Commands

The original PushWorld paper reports train/test success rates by Level 0 task
type. Use the eval-only script to generate the same style of table for any
saved imitation checkpoint.

Set the checkpoint once:

```bash
CKPT=models/planner_imitation_level0_base_small.pt
```

For the currently running multi-variant experiment, replace it with:

```bash
CKPT=models/planner_imitation_level0_multi4.pt
```

Evaluate every Level 0 task type on both train and test splits with a longer
rollout limit:

```bash
for variant in base goals obstacles shapes size walls all; do
  uv run python -u scripts/eval_planner_imitation.py \
    --checkpoint "$CKPT" \
    --eval-dir "data/level0/${variant}/train" \
    --split-name "level0_${variant}_train_200" \
    --all-eval \
    --max-steps 200 \
    --beam-width 8 \
    --beam-depth 8 \
    --top-k 3 \
    --output "reports/table2_like_level0_${variant}_train_200.json" \
    --tensorboard-log "runs/table2_like_level0_${variant}_train_200"

  uv run python -u scripts/eval_planner_imitation.py \
    --checkpoint "$CKPT" \
    --eval-dir "data/level0/${variant}/test" \
    --split-name "level0_${variant}_test_200" \
    --all-eval \
    --max-steps 200 \
    --beam-width 8 \
    --beam-depth 8 \
    --top-k 3 \
    --output "reports/table2_like_level0_${variant}_test_200.json" \
    --tensorboard-log "runs/table2_like_level0_${variant}_test_200"
done
```

Evaluate Level 1 zero-shot with the same rollout limit:

```bash
uv run python -u scripts/eval_planner_imitation.py \
  --checkpoint "$CKPT" \
  --eval-dir external/pushworld/benchmark/puzzles/level1 \
  --split-name level1_200 \
  --all-eval \
  --max-steps 200 \
  --beam-width 8 \
  --beam-depth 8 \
  --top-k 3 \
  --output reports/table2_like_level1_200.json \
  --tensorboard-log runs/table2_like_level1_200
```

Print a compact Markdown table from the generated JSON files:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

print("| Split | Solved | Total | Success |")
print("| --- | ---: | ---: | ---: |")
for path in sorted(Path("reports").glob("table2_like_*_200.json")):
    data = json.loads(path.read_text())
    total = int(data["total"])
    solved = int(data["solved"])
    skipped = int(data.get("skipped", 0))
    denom = max(1, total - skipped)
    print(f"| {data['split']} | {solved} | {total} | {100 * solved / denom:.1f}% |")
PY
```
