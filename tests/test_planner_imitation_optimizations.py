from __future__ import annotations

import sys
import types
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

if "pushworld.puzzle" not in sys.modules:
    pushworld_module = types.ModuleType("pushworld")
    puzzle_module = types.ModuleType("pushworld.puzzle")

    class _Actions:
        LEFT = 0
        FROM_CHAR: dict[str, int] = {}

    class _PushWorldPuzzle:
        pass

    puzzle_module.Actions = _Actions
    puzzle_module.PushWorldPuzzle = _PushWorldPuzzle
    pushworld_module.puzzle = puzzle_module
    sys.modules["pushworld"] = pushworld_module
    sys.modules["pushworld.puzzle"] = puzzle_module

from planner_imitation_rollout import (  # noqa: E402
    auto_distance_bins,
    beam_rank_score,
    distance_bin_values,
    distance_targets,
)
from train_planner_imitation_v2 import BoardTransformerPolicy  # noqa: E402


def test_log_distance_targets_are_compact_and_monotonic() -> None:
    remaining = torch.tensor([0, 1, 2, 4, 10, 100, 200])
    bins = auto_distance_bins(max_steps=200, distance_target="log")

    targets = distance_targets(remaining, bins, "log")

    assert bins < 201
    assert torch.all(targets[1:] >= targets[:-1])
    assert int(targets[-1]) < bins


def test_log_distance_bin_values_approximate_step_scale() -> None:
    values = distance_bin_values(
        distance_bins=7,
        distance_target="log",
        max_steps=200,
        device=torch.device("cpu"),
    )

    assert values.tolist()[0] == 0.0
    assert torch.all(values[1:] >= values[:-1])
    assert float(values[-1]) <= 200.0


def test_policy_distance_beam_score_preserves_old_default_shape() -> None:
    policy_only = beam_rank_score(
        policy_cost=3.0,
        expected_distance=8.0,
        path_len=2,
        beam_score="policy",
        distance_weight=0.15,
        beam_length_normalization=0.0,
    )
    policy_distance = beam_rank_score(
        policy_cost=3.0,
        expected_distance=8.0,
        path_len=2,
        beam_score="policy_distance",
        distance_weight=0.15,
        beam_length_normalization=0.0,
    )

    assert policy_only == 3.0
    assert policy_distance == 4.2


def test_conv_stem_policy_forward_shapes() -> None:
    model = BoardTransformerPolicy(
        channels=7,
        height=4,
        width=5,
        d_model=16,
        nhead=4,
        layers=1,
        distance_bins=7,
        encoder_stem="conv",
        dropout=0.01,
    )
    states = torch.zeros(2, 7, 4, 5)

    action_logits, distance_logits = model(states)

    assert action_logits.shape == (2, 4)
    assert distance_logits.shape == (2, 7)
