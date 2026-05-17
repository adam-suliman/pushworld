from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from pushworld_study.paths import ensure_upstream_pushworld_on_path


ensure_upstream_pushworld_on_path()

from pushworld.puzzle import Actions, PushWorldPuzzle  # noqa: E402


State = tuple[tuple[int, int], ...]
EncodeCache = dict[tuple[str, State], torch.Tensor]
PredictionCache = dict[tuple[str, State], tuple[torch.Tensor, torch.Tensor]]
ACTION_COUNT = 4
DISTANCE_TARGETS = ("linear", "log")
BEAM_SCORE_MODES = ("policy", "policy_distance", "distance")
SEARCH_MODES = ("beam", "best_first", "best_first_fallback")


@dataclass
class RolloutProfile:
    puzzle_parse_time_s: float = 0.0
    eval_loop_time_s: float = 0.0
    encode_time_s: float = 0.0
    model_forward_time_s: float = 0.0
    env_step_time_s: float = 0.0
    beam_expand_time_s: float = 0.0
    beam_rank_time_s: float = 0.0
    encode_cache_hits: int = 0
    encode_cache_misses: int = 0
    prediction_cache_hits: int = 0
    prediction_cache_misses: int = 0
    model_forward_batches: int = 0
    model_forward_states: int = 0
    predict_batch_calls: int = 0
    predict_batch_requested_states: int = 0
    predict_batch_unique_forward_states: int = 0
    beam_candidate_count: int = 0
    beam_closed_list_prunes: int = 0
    best_first_expand_time_s: float = 0.0
    best_first_rank_time_s: float = 0.0
    best_first_nodes_expanded: int = 0
    best_first_nodes_generated: int = 0
    best_first_closed_prunes: int = 0
    best_first_queue_max: int = 0
    best_first_batches: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "puzzle_parse_time_s": self.puzzle_parse_time_s,
            "eval_loop_time_s": self.eval_loop_time_s,
            "encode_time_s": self.encode_time_s,
            "model_forward_time_s": self.model_forward_time_s,
            "env_step_time_s": self.env_step_time_s,
            "beam_expand_time_s": self.beam_expand_time_s,
            "beam_rank_time_s": self.beam_rank_time_s,
            "encode_cache_hits": self.encode_cache_hits,
            "encode_cache_misses": self.encode_cache_misses,
            "prediction_cache_hits": self.prediction_cache_hits,
            "prediction_cache_misses": self.prediction_cache_misses,
            "model_forward_batches": self.model_forward_batches,
            "model_forward_states": self.model_forward_states,
            "predict_batch_calls": self.predict_batch_calls,
            "predict_batch_requested_states": self.predict_batch_requested_states,
            "predict_batch_unique_forward_states": self.predict_batch_unique_forward_states,
            "beam_candidate_count": self.beam_candidate_count,
            "beam_closed_list_prunes": self.beam_closed_list_prunes,
            "best_first_expand_time_s": self.best_first_expand_time_s,
            "best_first_rank_time_s": self.best_first_rank_time_s,
            "best_first_nodes_expanded": self.best_first_nodes_expanded,
            "best_first_nodes_generated": self.best_first_nodes_generated,
            "best_first_closed_prunes": self.best_first_closed_prunes,
            "best_first_queue_max": self.best_first_queue_max,
            "best_first_batches": self.best_first_batches,
        }


@dataclass(frozen=True)
class BestFirstSearchResult:
    solved: bool
    path: tuple[int, ...]
    expanded: int
    generated: int
    closed: int
    frontier: int


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
    profile: RolloutProfile | None = None,
) -> torch.Tensor:
    key = (puzzle_key, state)
    cached = cache.get(key)
    if cached is not None:
        if profile is not None:
            profile.encode_cache_hits += 1
        return cached
    if profile is not None:
        profile.encode_cache_misses += 1
    start = time.perf_counter()
    encoded = torch.from_numpy(encode_state(puzzle, state, height, width))
    if profile is not None:
        profile.encode_time_s += time.perf_counter() - start
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
    prediction_cache: PredictionCache | None = None,
    profile: RolloutProfile | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if profile is not None:
        profile.predict_batch_calls += 1
        profile.predict_batch_requested_states += len(puzzle_states)

    cached_results: dict[tuple[str, State], tuple[torch.Tensor, torch.Tensor]] = {}
    uncached: list[tuple[PushWorldPuzzle, str, State]] = []
    uncached_keys: set[tuple[str, State]] = set()
    for puzzle, puzzle_key, state in puzzle_states:
        key = (puzzle_key, state)
        cached = prediction_cache.get(key) if prediction_cache is not None else None
        if cached is not None:
            if profile is not None:
                profile.prediction_cache_hits += 1
            cached_results[key] = cached
            continue
        if key not in uncached_keys:
            uncached.append((puzzle, puzzle_key, state))
            uncached_keys.add(key)
        if profile is not None:
            profile.prediction_cache_misses += 1

    if uncached:
        encoded = [
            encode_cached(
                puzzle,
                puzzle_key,
                state,
                height,
                width,
                encode_cache,
                max_cache_entries,
                profile,
            )
            for puzzle, puzzle_key, state in uncached
        ]
        batch = torch.stack(encoded).to(device)
        start = time.perf_counter()
        action_logits, distance_logits = model(batch)
        action_log_probs = torch.log_softmax(action_logits, dim=-1).cpu()
        distance_probs = torch.softmax(distance_logits, dim=-1)
        distances = distance_bin_values(
            distance_logits.shape[-1],
            distance_target,
            distance_max_steps,
            device,
        )
        expected_distance = torch.sum(distance_probs * distances.unsqueeze(0), dim=-1).cpu()
        if profile is not None:
            profile.model_forward_time_s += time.perf_counter() - start
            profile.model_forward_batches += 1
            profile.model_forward_states += len(uncached)
            profile.predict_batch_unique_forward_states += len(uncached)
        for idx, (_, puzzle_key, state) in enumerate(uncached):
            key = (puzzle_key, state)
            result = (action_log_probs[idx], expected_distance[idx])
            cached_results[key] = result
            if (
                prediction_cache is not None
                and max_cache_entries > 0
                and len(prediction_cache) < max_cache_entries
            ):
                prediction_cache[key] = result

    action_rows = []
    distance_rows = []
    for _, puzzle_key, state in puzzle_states:
        action_log_probs, expected_distance = cached_results[(puzzle_key, state)]
        action_rows.append(action_log_probs)
        distance_rows.append(expected_distance)
    return torch.stack(action_rows), torch.stack(distance_rows)


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


def best_first_priority(
    policy_cost: float,
    expected_distance: float,
    path_len: int,
    distance_weight: float,
    step_penalty: float,
) -> float:
    return policy_cost + distance_weight * expected_distance + step_penalty * path_len


def best_first_search(
    model: nn.Module,
    puzzle: PushWorldPuzzle,
    state: State,
    height: int,
    width: int,
    device: torch.device,
    puzzle_key: str,
    encode_cache: EncodeCache,
    max_cache_entries: int,
    node_budget: int,
    batch_size: int,
    top_k: int,
    max_depth: int,
    distance_target: str = "linear",
    distance_max_steps: int | None = None,
    distance_weight: float = 0.15,
    step_penalty: float = 0.0,
    prediction_cache: PredictionCache | None = None,
    profile: RolloutProfile | None = None,
) -> BestFirstSearchResult:
    if node_budget <= 0 or batch_size <= 0 or top_k <= 0 or max_depth <= 0:
        return BestFirstSearchResult(False, (), 0, 0, 0, 0)
    if puzzle.is_goal_state(state):
        return BestFirstSearchResult(True, (), 0, 0, 0, 0)

    counter = itertools.count()
    frontier: list[tuple[float, int, int, State, tuple[int, ...], float]] = [
        (0.0, 0, next(counter), state, (), 0.0)
    ]
    closed: set[State] = set()
    expanded = 0
    generated = 0

    while frontier and expanded < node_budget:
        nodes: list[tuple[State, tuple[int, ...], float]] = []
        while frontier and len(nodes) < batch_size and expanded + len(nodes) < node_budget:
            _, _, _, node_state, path, policy_cost = heapq.heappop(frontier)
            if node_state in closed:
                if profile is not None:
                    profile.best_first_closed_prunes += 1
                continue
            closed.add(node_state)
            nodes.append((node_state, path, policy_cost))
        if not nodes:
            continue

        if profile is not None:
            profile.best_first_batches += 1
            profile.best_first_nodes_expanded += len(nodes)

        action_log_probs, _ = predict_batch(
            model,
            [(puzzle, puzzle_key, node_state) for node_state, _, _ in nodes],
            height,
            width,
            device,
            encode_cache,
            max_cache_entries,
            distance_target,
            distance_max_steps,
            prediction_cache,
            profile,
        )

        candidates_by_state: dict[State, tuple[State, tuple[int, ...], float]] = {}
        expand_start = time.perf_counter()
        for node_idx, (node_state, path, policy_cost) in enumerate(nodes):
            expanded += 1
            if puzzle.is_goal_state(node_state):
                return BestFirstSearchResult(True, path, expanded, generated, len(closed), len(frontier))
            if len(path) >= max_depth:
                continue
            action_count = min(top_k, ACTION_COUNT)
            top_actions = torch.topk(action_log_probs[node_idx], k=action_count).indices.tolist()
            for action in top_actions:
                step_start = time.perf_counter()
                next_state = puzzle.get_next_state(node_state, int(action))
                if profile is not None:
                    profile.env_step_time_s += time.perf_counter() - step_start
                if next_state == node_state:
                    continue
                if next_state in closed:
                    if profile is not None:
                        profile.best_first_closed_prunes += 1
                    continue
                next_path = path + (int(action),)
                next_policy_cost = policy_cost - float(action_log_probs[node_idx, action])
                generated += 1
                if profile is not None:
                    profile.best_first_nodes_generated += 1
                if puzzle.is_goal_state(next_state):
                    if profile is not None:
                        profile.best_first_expand_time_s += time.perf_counter() - expand_start
                    return BestFirstSearchResult(
                        True,
                        next_path,
                        expanded,
                        generated,
                        len(closed),
                        len(frontier),
                    )
                previous = candidates_by_state.get(next_state)
                if previous is None or next_policy_cost < previous[2]:
                    candidates_by_state[next_state] = (next_state, next_path, next_policy_cost)
        if profile is not None:
            profile.best_first_expand_time_s += time.perf_counter() - expand_start

        candidates = list(candidates_by_state.values())
        if not candidates:
            continue
        rank_start = time.perf_counter()
        _, leaf_distances = predict_batch(
            model,
            [(puzzle, puzzle_key, candidate[0]) for candidate in candidates],
            height,
            width,
            device,
            encode_cache,
            max_cache_entries,
            distance_target,
            distance_max_steps,
            prediction_cache,
            profile,
        )
        for candidate, expected_distance in zip(candidates, leaf_distances.tolist(), strict=True):
            _, path, policy_cost = candidate
            priority = best_first_priority(
                policy_cost=policy_cost,
                expected_distance=float(expected_distance),
                path_len=len(path),
                distance_weight=distance_weight,
                step_penalty=step_penalty,
            )
            heapq.heappush(frontier, (priority, len(path), next(counter), candidate[0], path, policy_cost))
        if profile is not None:
            profile.best_first_rank_time_s += time.perf_counter() - rank_start
            profile.best_first_queue_max = max(profile.best_first_queue_max, len(frontier))

    return BestFirstSearchResult(False, (), expanded, generated, len(closed), len(frontier))


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
    prediction_cache: PredictionCache | None = None,
    profile: RolloutProfile | None = None,
    closed_list_pruning: bool = False,
) -> int:
    seen = set(seen_states) if seen_states is not None and repeat_penalty > 0.0 else set()
    closed = set(seen_states) if seen_states is not None and closed_list_pruning else set()

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
            prediction_cache,
            profile,
        )
        fallback_action: int | None = None
        for action in torch.argsort(action_log_probs[0], descending=True).tolist():
            step_start = time.perf_counter()
            next_state = puzzle.get_next_state(state, int(action))
            if profile is not None:
                profile.env_step_time_s += time.perf_counter() - step_start
            if next_state == state:
                continue
            if fallback_action is None:
                fallback_action = int(action)
            if next_state not in seen and next_state not in closed:
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
            prediction_cache,
            profile,
        )
        action_log_probs, _ = predictions
        candidates_by_state: dict[State, tuple[State, tuple[int, ...], float]] = {}
        expand_start = time.perf_counter()
        for beam_idx, (beam_state, path, score) in enumerate(beams):
            action_count = min(top_k, ACTION_COUNT)
            top_actions = torch.topk(action_log_probs[beam_idx], k=action_count).indices.tolist()
            for action in top_actions:
                step_start = time.perf_counter()
                next_state = puzzle.get_next_state(beam_state, int(action))
                if profile is not None:
                    profile.env_step_time_s += time.perf_counter() - step_start
                if next_state == beam_state:
                    continue
                if next_state in closed:
                    if profile is not None:
                        profile.beam_closed_list_prunes += 1
                    continue
                next_path = path + (int(action),)
                best_nonempty_path = best_nonempty_path or next_path
                next_score = score - float(action_log_probs[beam_idx, action])
                if next_state in seen:
                    next_score += repeat_penalty
                if profile is not None:
                    profile.beam_candidate_count += 1
                if puzzle.is_goal_state(next_state):
                    best_solved = next_path
                    break
                previous = candidates_by_state.get(next_state)
                if previous is None or next_score < previous[2]:
                    candidates_by_state[next_state] = (next_state, next_path, next_score)
            if best_solved is not None:
                break
        if profile is not None:
            profile.beam_expand_time_s += time.perf_counter() - expand_start
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
            prediction_cache,
            profile,
        )
        del leaf_log_probs
        rank_start = time.perf_counter()
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
        if profile is not None:
            profile.beam_rank_time_s += time.perf_counter() - rank_start
        beams = [candidate for candidate, _ in ranked[:beam_width]]

    if beams and beams[0][1]:
        return beams[0][1][0]
    if best_nonempty_path:
        return best_nonempty_path[0]
    return Actions.LEFT
