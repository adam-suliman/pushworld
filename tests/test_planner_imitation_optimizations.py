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
    RolloutProfile,
    auto_distance_bins,
    best_first_search,
    beam_rank_score,
    distance_bin_values,
    distance_targets,
    predict_batch,
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


def test_predict_batch_caches_duplicate_model_outputs() -> None:
    class _Movable:
        cells = {(0, 0)}

    class _Puzzle:
        wall_positions: list[tuple[int, int]] = []
        agent_wall_positions: list[tuple[int, int]] = []
        movable_objects = [_Movable()]
        goal_state: tuple[tuple[int, int], ...] = ()

    class _CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls += 1
            return torch.zeros(states.shape[0], 4), torch.zeros(states.shape[0], 3)

    model = _CountingModel()
    puzzle = _Puzzle()
    state = ((0, 0),)
    encode_cache = {}
    prediction_cache = {}
    profile = RolloutProfile()

    actions, distances = predict_batch(
        model,
        [(puzzle, "puzzle", state), (puzzle, "puzzle", state)],
        height=2,
        width=2,
        device=torch.device("cpu"),
        encode_cache=encode_cache,
        max_cache_entries=10,
        distance_target="linear",
        distance_max_steps=10,
        prediction_cache=prediction_cache,
        profile=profile,
    )
    predict_batch(
        model,
        [(puzzle, "puzzle", state)],
        height=2,
        width=2,
        device=torch.device("cpu"),
        encode_cache=encode_cache,
        max_cache_entries=10,
        distance_target="linear",
        distance_max_steps=10,
        prediction_cache=prediction_cache,
        profile=profile,
    )

    assert actions.shape == (2, 4)
    assert distances.shape == (2,)
    assert model.calls == 1
    assert len(prediction_cache) == 1
    assert profile.model_forward_states == 1
    assert profile.prediction_cache_hits == 1



def test_best_first_search_finds_policy_guided_path() -> None:
    class _Movable:
        cells = {(0, 0)}

    class _Puzzle:
        wall_positions: list[tuple[int, int]] = []
        agent_wall_positions: list[tuple[int, int]] = []
        movable_objects = [_Movable()]
        goal_state: tuple[tuple[int, int], ...] = ()

        def get_next_state(self, state, action: int):
            x, y = state[0]
            if action == 1:
                x = min(2, x + 1)
            elif action == 0:
                x = max(0, x - 1)
            return ((x, y),)

        def is_goal_state(self, state) -> bool:
            return state[0] == (2, 0)

    class _RightModel(torch.nn.Module):
        def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            action_logits = torch.full((states.shape[0], 4), -4.0)
            action_logits[:, 1] = 4.0
            distance_logits = torch.zeros(states.shape[0], 4)
            return action_logits, distance_logits

    profile = RolloutProfile()
    result = best_first_search(
        model=_RightModel(),
        puzzle=_Puzzle(),
        state=((0, 0),),
        height=1,
        width=3,
        device=torch.device("cpu"),
        puzzle_key="toy",
        encode_cache={},
        max_cache_entries=100,
        node_budget=8,
        batch_size=2,
        top_k=2,
        max_depth=4,
        prediction_cache={},
        profile=profile,
    )

    assert result.solved
    assert result.path == (1, 1)
    assert profile.best_first_nodes_expanded > 0
    assert profile.best_first_nodes_generated > 0
