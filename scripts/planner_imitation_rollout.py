from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from torch import nn

from pushworld_study.paths import ensure_upstream_pushworld_on_path


ensure_upstream_pushworld_on_path()

from pushworld.puzzle import Actions, PushWorldPuzzle  # noqa: E402


State = tuple[tuple[int, int], ...]
EncodeCache = dict[tuple[str, State], torch.Tensor]
ACTION_COUNT = 4
DISTANCE_TARGETS = ("linear", "log")
BEAM_SCORE_MODES = ("policy", "policy_distance", "distance")


def set_cells(
    planes: np.ndarray,
    channel: int,
    origin: tuple[int, int],
    cells: set[tuple[int, int]],
) -> None:
    origin_x, origin_y = origin
    _, height, width = planes.shape
    for cell_x, cell_y in cells:
        x = origin_x + cell_x
        y = origin_y + cell_y
        if 0 <= x < width and 0 <= y < height:
            planes[channel, y, x] = 1.0


def encode_state(
    puzzle: PushWorldPuzzle,
    state: State,
    height: int,
    width: int,
) -> np.ndarray:
    planes = np.zeros((7, height, width), dtype=np.float32)

    for x, y in puzzle.wall_positions:
        if 0 <= x < width and 0 <= y < height:
            planes[0, y, x] = 1.0
    for x, y in puzzle.agent_wall_positions:
        if 0 <= x < width and 0 <= y < height:
            planes[1, y, x] = 1.0

    goal_count = len(puzzle.goal_state)
    for movable_idx, movable in enumerate(puzzle.movable_objects):
        if movable_idx == 0:
            channel = 2
        elif movable_idx <= goal_count:
            channel = 3
        else:
            channel = 4
        set_cells(planes, channel, state[movable_idx], movable.cells)

    for goal_idx, goal in enumerate(puzzle.goal_state, start=1):
        if goal_idx < len(puzzle.movable_objects):
            set_cells(planes, 5, goal, puzzle.movable_objects[goal_idx].cells)
            if state[goal_idx] == goal:
                set_cells(planes, 6, goal, puzzle.movable_objects[goal_idx].cells)

    return planes


def encode_cached(
    puzzle: PushWorldPuzzle,
    puzzle_key: str,
    state: State,
    height: int,
    width: int,
    cache: EncodeCache,
    max_cache_entries: int,
) -> torch.Tensor:
    key = (puzzle_key, state)
    cached = cache.get(key)
    if cached is not None:
        return cached
    encoded = torch.from_numpy(encode_state(puzzle, state, height, width))
    if max_cache_entries > 0 and len(cache) < max_cache_entries:
        cache[key] = encoded
    return encoded


def distance_targets(
    remaining: torch.Tensor,
    distance_bins: int,
    distance_target: str,
) -> torch.Tensor:
    if distance_target not in DISTANCE_TARGETS:
        raise ValueError(f"Unknown distance target {distance_target!r}; expected one of {DISTANCE_TARGETS}")
    if distance_bins <= 1:
        raise ValueError("distance_bins must be > 1")
    if distance_target == "log":
        targets = torch.round(torch.log(remaining.float() + 1.0)).long()
    else:
        targets = remaining.long()
    return targets.clamp_(min=0, max=distance_bins - 1)


def auto_distance_bins(max_steps: int, distance_target: str) -> int:
    if distance_target not in DISTANCE_TARGETS:
        raise ValueError(f"Unknown distance target {distance_target!r}; expected one of {DISTANCE_TARGETS}")
    if distance_target == "log":
        return max(2, int(math.ceil(math.log(max_steps + 1))) + 1)
    return max_steps + 1


def distance_bin_values(
    distance_bins: int,
    distance_target: str,
    max_steps: int | None,
    device: torch.device,
) -> torch.Tensor:
    if distance_target not in DISTANCE_TARGETS:
        raise ValueError(f"Unknown distance target {distance_target!r}; expected one of {DISTANCE_TARGETS}")
    bins = torch.arange(distance_bins, device=device, dtype=torch.float32)
    if distance_target == "log":
        values = torch.expm1(bins)
    else:
        values = bins
    if max_steps is not None:
        values = torch.clamp(values, max=float(max_steps))
    return values


def predict_batch(
    model: nn.Module,
    puzzle_states: list[tuple[PushWorldPuzzle, str, State]],
    height: int,
    width: int,
    device: torch.device,
    encode_cache: EncodeCache,
    max_cache_entries: int,
    distance_target: str = "linear",
    distance_max_steps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [
        encode_cached(puzzle, puzzle_key, state, height, width, encode_cache, max_cache_entries)
        for puzzle, puzzle_key, state in puzzle_states
    ]
    batch = torch.stack(encoded).to(device)
    action_logits, distance_logits = model(batch)
    action_log_probs = torch.log_softmax(action_logits, dim=-1)
    distance_probs = torch.softmax(distance_logits, dim=-1)
    distances = distance_bin_values(
        distance_logits.shape[-1],
        distance_target,
        distance_max_steps,
        device,
    )
    expected_distance = torch.sum(distance_probs * distances.unsqueeze(0), dim=-1)
    return action_log_probs.cpu(), expected_distance.cpu()


def beam_rank_score(
    policy_cost: float,
    expected_distance: float,
    path_len: int,
    beam_score: str,
    distance_weight: float,
    beam_length_normalization: float,
) -> float:
    if beam_score not in BEAM_SCORE_MODES:
        raise ValueError(f"Unknown beam score mode {beam_score!r}; expected one of {BEAM_SCORE_MODES}")
    if beam_length_normalization < 0.0:
        raise ValueError("beam_length_normalization must be >= 0")
    if distance_weight < 0.0:
        raise ValueError("distance_weight must be >= 0")

    normalized_policy_cost = policy_cost / (max(1, path_len) ** beam_length_normalization)
    if beam_score == "policy":
        return normalized_policy_cost
    if beam_score == "distance":
        return expected_distance + distance_weight * normalized_policy_cost
    return normalized_policy_cost + distance_weight * expected_distance


def choose_action(
    model: nn.Module,
    puzzle: PushWorldPuzzle,
    state: State,
    height: int,
    width: int,
    device: torch.device,
    beam_width: int,
    beam_depth: int,
    top_k: int,
    puzzle_key: str,
    encode_cache: EncodeCache,
    max_cache_entries: int,
    seen_states: Iterable[State] | None = None,
    repeat_penalty: float = 0.0,
    distance_target: str = "linear",
    distance_max_steps: int | None = None,
    beam_score: str = "policy_distance",
    distance_weight: float = 0.15,
    beam_length_normalization: float = 0.0,
) -> int:
    seen = set(seen_states) if seen_states is not None and repeat_penalty > 0.0 else set()

    if beam_width <= 1 or beam_depth <= 1:
        action_log_probs, _ = predict_batch(
            model,
            [(puzzle, puzzle_key, state)],
            height,
            width,
            device,
            encode_cache,
            max_cache_entries,
            distance_target,
            distance_max_steps,
        )
        fallback_action: int | None = None
        for action in torch.argsort(action_log_probs[0], descending=True).tolist():
            next_state = puzzle.get_next_state(state, int(action))
            if next_state == state:
                continue
            if fallback_action is None:
                fallback_action = int(action)
            if next_state not in seen:
                return int(action)
        if fallback_action is not None:
            return fallback_action
        return int(torch.argmax(action_log_probs[0]).item())

    beams: list[tuple[State, tuple[int, ...], float]] = [(state, (), 0.0)]
    best_solved: tuple[int, ...] | None = None
    best_nonempty_path: tuple[int, ...] | None = None
    for _ in range(beam_depth):
        predictions = predict_batch(
            model,
            [(puzzle, puzzle_key, beam_state) for beam_state, _, _ in beams],
            height,
            width,
            device,
            encode_cache,
            max_cache_entries,
            distance_target,
            distance_max_steps,
        )
        action_log_probs, _ = predictions
        candidates_by_state: dict[State, tuple[State, tuple[int, ...], float]] = {}
        for beam_idx, (beam_state, path, score) in enumerate(beams):
            action_count = min(top_k, ACTION_COUNT)
            top_actions = torch.topk(action_log_probs[beam_idx], k=action_count).indices.tolist()
            for action in top_actions:
                next_state = puzzle.get_next_state(beam_state, int(action))
                if next_state == beam_state:
                    continue
                next_path = path + (int(action),)
                best_nonempty_path = best_nonempty_path or next_path
                next_score = score - float(action_log_probs[beam_idx, action])
                if next_state in seen:
                    next_score += repeat_penalty
                if puzzle.is_goal_state(next_state):
                    best_solved = next_path
                    break
                previous = candidates_by_state.get(next_state)
                if previous is None or next_score < previous[2]:
                    candidates_by_state[next_state] = (next_state, next_path, next_score)
            if best_solved is not None:
                break
        if best_solved is not None:
            return best_solved[0]
        candidates = list(candidates_by_state.values())
        if not candidates:
            break
        leaf_log_probs, leaf_distances = predict_batch(
            model,
            [(puzzle, puzzle_key, candidate[0]) for candidate in candidates],
            height,
            width,
            device,
            encode_cache,
            max_cache_entries,
            distance_target,
            distance_max_steps,
        )
        del leaf_log_probs
        ranked = sorted(
            zip(candidates, leaf_distances.tolist(), strict=True),
            key=lambda item: beam_rank_score(
                policy_cost=item[0][2],
                expected_distance=float(item[1]),
                path_len=len(item[0][1]),
                beam_score=beam_score,
                distance_weight=distance_weight,
                beam_length_normalization=beam_length_normalization,
            ),
        )
        beams = [candidate for candidate, _ in ranked[:beam_width]]

    if beams and beams[0][1]:
        return beams[0][1][0]
    if best_nonempty_path:
        return best_nonempty_path[0]
    return Actions.LEFT
