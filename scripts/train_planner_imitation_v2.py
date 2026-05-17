from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from planner_imitation_rollout import (
    BEAM_SCORE_MODES,
    DISTANCE_TARGETS,
    RolloutProfile,
    auto_distance_bins,
    choose_action,
    distance_targets,
    encode_state,
)
from pushworld_study.paths import PROJECT_ROOT, ensure_upstream_pushworld_on_path


ensure_upstream_pushworld_on_path()

from pushworld.puzzle import Actions, PushWorldPuzzle  # noqa: E402


ACTION_CHARS = "LRUD"
ACTION_NAMES = ("left", "right", "up", "down")
SYMMETRY_TRANSFORMS = (
    "r0",
    "r90",
    "r180",
    "r270",
    "r0_flipped",
    "r90_flipped",
    "r180_flipped",
    "r270_flipped",
)
ACTION_CHAR_TO_INDEX = {char: idx for idx, char in enumerate(ACTION_CHARS)}
ACTION_INDEX_TO_CHAR = {idx: char for idx, char in enumerate(ACTION_CHARS)}
CACHE_SCHEMA_VERSION = 1
TRANSFORM_ACTION_MAP = {
    "r0": {"L": "L", "R": "R", "U": "U", "D": "D"},
    "r90": {"L": "U", "R": "D", "U": "R", "D": "L"},
    "r180": {"L": "R", "R": "L", "U": "D", "D": "U"},
    "r270": {"L": "D", "R": "U", "U": "L", "D": "R"},
    "r0_flipped": {"L": "L", "R": "R", "U": "D", "D": "U"},
    "r90_flipped": {"L": "U", "R": "D", "U": "L", "D": "R"},
    "r180_flipped": {"L": "R", "R": "L", "U": "U", "D": "D"},
    "r270_flipped": {"L": "D", "R": "U", "U": "R", "D": "L"},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_metadata(device: torch.device, amp: bool) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    cuda_device = None
    if cuda_available:
        cuda_index = device.index if device.type == "cuda" and device.index is not None else 0
        cuda_device = torch.cuda.get_device_name(cuda_index)
    return {
        "training_device": str(device),
        "amp_enabled": bool(amp and device.type == "cuda"),
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "torch_version": torch.__version__,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_components": [
            "RGD planner subprocesses",
            "puzzle parsing",
            "dataset materialization",
            "dataloader",
            "closed-loop evaluation search",
        ],
    }


def default_planner_path() -> Path:
    base = PROJECT_ROOT / "external/pushworld/cpp/build/bin/run_planner"
    exe = base.with_suffix(".exe")
    return exe if exe.exists() else base


@dataclass(frozen=True)
class Trajectory:
    puzzle_path: Path
    plan: str
    solve_time_s: float


@dataclass(frozen=True)
class ExpertStep:
    puzzle: PushWorldPuzzle
    puzzle_height: int
    puzzle_width: int
    state: tuple[tuple[int, int], ...]
    action: int
    remaining: int
    transforms: tuple[str, ...]


class TrainingInterrupted(Exception):
    def __init__(self, losses: list[float]) -> None:
        super().__init__("Training interrupted")
        self.losses = losses


class ExpertDataset(Dataset):
    def __init__(
        self,
        trajectories: list[Trajectory],
        height: int,
        width: int,
        transforms: tuple[str, ...] = ("r0",),
        transform_level0_only: bool = False,
        seed: int = 1,
    ) -> None:
        self.steps: list[ExpertStep] = []
        self.height = height
        self.width = width
        self.transforms = transforms
        self.transform_level0_only = transform_level0_only
        self.base_examples = 0
        self.augmented_base_examples = 0
        self.unaugmented_base_examples = 0
        self.rng = random.Random(seed)
        self.profile: dict[str, float] = {
            "dataset_materialization_time_s": 0.0,
            "puzzle_parse_time_s": 0.0,
            "state_encode_time_s": 0.0,
            "env_step_time_s": 0.0,
        }

        materialize_start = time.perf_counter()
        for trajectory in trajectories:
            parse_start = time.perf_counter()
            puzzle = PushWorldPuzzle(str(trajectory.puzzle_path))
            self.profile["puzzle_parse_time_s"] += time.perf_counter() - parse_start
            puzzle_width, puzzle_height = puzzle.dimensions
            state = puzzle.initial_state
            plan_length = len(trajectory.plan)
            trajectory_transforms = transforms
            if transform_level0_only and not is_level0_path(trajectory.puzzle_path):
                trajectory_transforms = ("r0",)
            for step_idx, action_char in enumerate(trajectory.plan):
                action = Actions.FROM_CHAR[action_char]
                remaining = plan_length - step_idx
                self.base_examples += 1
                if len(trajectory_transforms) > 1:
                    self.augmented_base_examples += 1
                else:
                    self.unaugmented_base_examples += 1
                self.steps.append(
                    ExpertStep(
                        puzzle=puzzle,
                        puzzle_height=puzzle_height,
                        puzzle_width=puzzle_width,
                        state=state,
                        action=action,
                        remaining=remaining,
                        transforms=trajectory_transforms,
                    )
                )
                step_start = time.perf_counter()
                state = puzzle.get_next_state(state, action)
                self.profile["env_step_time_s"] += time.perf_counter() - step_start
            if not puzzle.is_goal_state(state):
                raise ValueError(f"Planner trace does not solve {trajectory.puzzle_path}")
        self.profile["dataset_materialization_time_s"] = time.perf_counter() - materialize_start

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        step = self.steps[idx]
        transform = self.rng.choice(step.transforms) if len(step.transforms) > 1 else "r0"
        state = transform_encoded_state(
            encode_state(step.puzzle, step.state, self.height, self.width),
            puzzle_height=step.puzzle_height,
            puzzle_width=step.puzzle_width,
            height=self.height,
            width=self.width,
            transform=transform,
        )
        action = transform_action(step.action, transform)
        return (
            torch.from_numpy(state),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(step.remaining, dtype=torch.long),
        )


class CachedExpertDataset(Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        remaining: torch.Tensor,
        puzzle_heights: torch.Tensor,
        puzzle_widths: torch.Tensor,
        puzzle_indices: torch.Tensor,
        puzzle_paths: list[Path],
        height: int,
        width: int,
        transforms: tuple[str, ...] = ("r0",),
        transform_level0_only: bool = False,
        seed: int = 1,
        profile: dict[str, float] | None = None,
    ) -> None:
        self.states = states.cpu()
        self.actions = actions.cpu()
        self.remaining = remaining.cpu()
        self.puzzle_heights = puzzle_heights.cpu()
        self.puzzle_widths = puzzle_widths.cpu()
        self.puzzle_indices = puzzle_indices.cpu()
        self.puzzle_paths = puzzle_paths
        self.height = height
        self.width = width
        self.transforms = transforms
        self.transform_level0_only = transform_level0_only
        self.rng = random.Random(seed)
        self.profile = profile or {
            "dataset_materialization_time_s": 0.0,
            "puzzle_parse_time_s": 0.0,
            "state_encode_time_s": 0.0,
            "env_step_time_s": 0.0,
        }
        self.base_examples = int(self.actions.numel())
        self.augmented_base_examples = 0
        self.unaugmented_base_examples = 0
        for idx in range(self.base_examples):
            if len(self._transforms_for_index(idx)) > 1:
                self.augmented_base_examples += 1
            else:
                self.unaugmented_base_examples += 1

    def _transforms_for_index(self, idx: int) -> tuple[str, ...]:
        puzzle_path = self.puzzle_paths[int(self.puzzle_indices[idx])]
        if self.transform_level0_only and not is_level0_path(puzzle_path):
            return ("r0",)
        return self.transforms

    def __len__(self) -> int:
        return self.base_examples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transforms = self._transforms_for_index(idx)
        transform = self.rng.choice(transforms) if len(transforms) > 1 else "r0"
        action = int(self.actions[idx])
        if transform == "r0":
            state = self.states[idx].float()
        else:
            state = torch.from_numpy(
                transform_encoded_state(
                    self.states[idx].numpy(),
                    puzzle_height=int(self.puzzle_heights[idx]),
                    puzzle_width=int(self.puzzle_widths[idx]),
                    height=self.height,
                    width=self.width,
                    transform=transform,
                )
            ).float()
            action = transform_action(action, transform)
        return (
            state,
            torch.tensor(action, dtype=torch.long),
            self.remaining[idx].long(),
        )


def is_level0_path(path: Path) -> bool:
    try:
        path.resolve().relative_to((PROJECT_ROOT / "data/level0").resolve())
        return True
    except ValueError:
        return False


class BoardTransformerPolicy(nn.Module):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        d_model: int = 96,
        nhead: int = 4,
        layers: int = 2,
        distance_bins: int = 101,
        encoder_stem: str = "linear",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if encoder_stem not in ("linear", "conv"):
            raise ValueError("--encoder-stem must be either 'linear' or 'conv'")
        self.height = height
        self.width = width
        self.encoder_stem = encoder_stem
        if encoder_stem == "conv":
            self.conv_stem = nn.Sequential(
                nn.Conv2d(channels, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
            )
        else:
            self.token_proj = nn.Linear(channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, height * width + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.action_head = nn.Linear(d_model, len(ACTION_NAMES))
        self.distance_head = nn.Linear(d_model, distance_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.encoder_stem == "conv":
            tokens = self.conv_stem(x).flatten(2).transpose(1, 2)
        else:
            tokens = x.permute(0, 2, 3, 1).reshape(batch, self.height * self.width, -1)
            tokens = self.token_proj(tokens)
        cls = self.cls.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        encoded = self.encoder(tokens)
        pooled = encoded[:, 0]
        return self.action_head(pooled), self.distance_head(pooled)


def sorted_puzzles(path: Path) -> list[Path]:
    def key(puzzle_path: Path) -> tuple[str, int]:
        match = re.search(r"_(\d+)$", puzzle_path.stem)
        return (puzzle_path.stem[: match.start()] if match else puzzle_path.stem, int(match.group(1)) if match else -1)

    return sorted(path.glob("*.pwp"), key=key)


def select_puzzles(paths: list[Path], limit: int, use_all: bool) -> list[Path]:
    puzzle_paths = []
    seen = set()
    for path in paths:
        for puzzle_path in sorted_puzzles(path):
            resolved = puzzle_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            puzzle_paths.append(puzzle_path)
    paths = puzzle_paths
    return paths if use_all else paths[:limit]


def max_dimensions(paths: list[Path]) -> tuple[int, int]:
    widths = []
    heights = []
    for path in paths:
        puzzle = PushWorldPuzzle(str(path))
        width, height = puzzle.dimensions
        widths.append(width)
        heights.append(height)
    return max(heights), max(widths)


def transform_action(action: int, transform: str) -> int:
    action_char = ACTION_INDEX_TO_CHAR[action]
    transformed_char = TRANSFORM_ACTION_MAP[transform][action_char]
    return ACTION_CHAR_TO_INDEX[transformed_char]


def transform_encoded_state(
    planes: np.ndarray,
    puzzle_height: int,
    puzzle_width: int,
    height: int,
    width: int,
    transform: str,
) -> np.ndarray:
    crop = planes[:, :puzzle_height, :puzzle_width]
    transformed = crop
    if transform.endswith("_flipped"):
        transformed = np.flip(transformed, axis=1)
    rotation = int(transform.split("_", maxsplit=1)[0][1:])
    for _ in range(rotation // 90):
        transformed = np.rot90(transformed, axes=(2, 1))

    transformed = transformed.copy()
    transformed_height, transformed_width = transformed.shape[1:]
    if transformed_height > height or transformed_width > width:
        raise ValueError(
            f"Transformed state {transform} has shape "
            f"{transformed_height}x{transformed_width}, exceeding {height}x{width}"
        )
    output = np.zeros_like(planes)
    output[:, :transformed_height, :transformed_width] = transformed
    return output


def solve_with_rgd(planner: Path, puzzle_path: Path, time_limit: float) -> Trajectory:
    start = time.perf_counter()
    result = subprocess.run(
        [str(planner), "N+RGD", str(puzzle_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=time_limit + 1.0,
    )
    elapsed = time.perf_counter() - start
    plan = result.stdout.strip()
    if result.returncode != 0 or not plan or not set(plan).issubset(set(ACTION_CHARS)):
        raise RuntimeError(
            f"Failed to solve {puzzle_path}: returncode={result.returncode}, "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
    puzzle = PushWorldPuzzle(str(puzzle_path))
    actions = [Actions.FROM_CHAR[ch] for ch in plan]
    if not puzzle.is_valid_plan(actions):
        raise RuntimeError(f"Planner returned invalid plan for {puzzle_path}: {plan}")
    return Trajectory(puzzle_path=puzzle_path, plan=plan, solve_time_s=elapsed)


def solve_trajectories(
    planner: Path,
    puzzle_paths: list[Path],
    time_limit: float,
    workers: int,
) -> list[Trajectory]:
    if workers <= 1:
        return [
            solve_with_rgd(planner, path, time_limit)
            for path in tqdm(puzzle_paths, desc="solve expert traces", unit="puzzle")
        ]

    trajectories: list[Trajectory | None] = [None] * len(puzzle_paths)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(solve_with_rgd, planner, path, time_limit): idx
            for idx, path in enumerate(puzzle_paths)
        }
        progress = tqdm(as_completed(futures), total=len(futures), desc="solve expert traces", unit="puzzle")
        for future in progress:
            idx = futures[future]
            trajectories[idx] = future.result()
    return [trajectory for trajectory in trajectories if trajectory is not None]


def project_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_dir_size_bytes(cache_dir: Path) -> int:
    if not cache_dir.exists():
        return 0
    return sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())


def cache_files(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "manifest.json", cache_dir / "data.pt"


def planner_imitation_cache_key(puzzle_paths: list[Path], height: int, width: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"schema={CACHE_SCHEMA_VERSION};height={height};width={width}".encode("utf-8"))
    for path in puzzle_paths:
        digest.update(project_relative_path(path).encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()[:16]


def cache_exists(cache_dir: Path) -> bool:
    manifest_path, data_path = cache_files(cache_dir)
    return manifest_path.exists() and data_path.exists()


def validate_cache_manifest(
    manifest: dict[str, object],
    puzzle_paths: list[Path],
    height: int,
    width: int,
) -> None:
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Cache schema {manifest.get('schema_version')} does not match expected {CACHE_SCHEMA_VERSION}"
        )
    if int(manifest.get("height", -1)) != height or int(manifest.get("width", -1)) != width:
        raise ValueError(
            f"Cache board {manifest.get('height')}x{manifest.get('width')} does not match requested {height}x{width}"
        )
    cached_puzzles = manifest.get("puzzles")
    if not isinstance(cached_puzzles, list):
        raise ValueError("Cache manifest is missing a puzzle list")
    if len(cached_puzzles) != len(puzzle_paths):
        raise ValueError(f"Cache puzzle count {len(cached_puzzles)} does not match requested {len(puzzle_paths)}")
    for idx, (cached, requested_path) in enumerate(zip(cached_puzzles, puzzle_paths, strict=True)):
        if not isinstance(cached, dict):
            raise ValueError(f"Cache puzzle entry {idx} is not an object")
        requested_rel = project_relative_path(requested_path)
        cached_rel = str(cached.get("path"))
        if cached_rel != requested_rel:
            raise ValueError(f"Cache puzzle {idx} path {cached_rel!r} does not match requested {requested_rel!r}")
        requested_hash = file_sha256(requested_path)
        cached_hash = str(cached.get("sha256"))
        if cached_hash != requested_hash:
            raise ValueError(f"Cache puzzle {requested_rel} content hash changed; rebuild the cache")


def load_planner_imitation_cache(
    cache_dir: Path,
    puzzle_paths: list[Path],
    height: int,
    width: int,
    transforms: tuple[str, ...],
    transform_level0_only: bool,
    seed: int,
) -> tuple[list[Trajectory], CachedExpertDataset, dict[str, object]]:
    start = time.perf_counter()
    manifest_path, data_path = cache_files(cache_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_cache_manifest(manifest, puzzle_paths, height, width)
    payload = torch.load(data_path, map_location="cpu", weights_only=True)
    cached_puzzles = manifest["puzzles"]
    trajectories = [
        Trajectory(
            puzzle_path=resolve_manifest_path(str(item["path"])),
            plan=str(item["plan"]),
            solve_time_s=float(item.get("solve_time_s", 0.0)),
        )
        for item in cached_puzzles
    ]
    dataset = CachedExpertDataset(
        states=payload["states"],
        actions=payload["actions"],
        remaining=payload["remaining"],
        puzzle_heights=payload["puzzle_heights"],
        puzzle_widths=payload["puzzle_widths"],
        puzzle_indices=payload["puzzle_indices"],
        puzzle_paths=[trajectory.puzzle_path for trajectory in trajectories],
        height=height,
        width=width,
        transforms=transforms,
        transform_level0_only=transform_level0_only,
        seed=seed,
        profile={
            "dataset_materialization_time_s": 0.0,
            "puzzle_parse_time_s": 0.0,
            "state_encode_time_s": 0.0,
            "env_step_time_s": 0.0,
        },
    )
    cache_profile = {
        "hit": True,
        "cache_dir": str(cache_dir),
        "load_time_s": time.perf_counter() - start,
        "build_time_s": 0.0,
        "size_bytes": cache_dir_size_bytes(cache_dir),
        "manifest_build_profile": manifest.get("build_profile", {}),
    }
    return trajectories, dataset, cache_profile


def build_planner_imitation_cache(
    cache_dir: Path,
    trajectories: list[Trajectory],
    height: int,
    width: int,
    transforms: tuple[str, ...],
    transform_level0_only: bool,
    seed: int,
) -> tuple[CachedExpertDataset, dict[str, object]]:
    start = time.perf_counter()
    states: list[torch.Tensor] = []
    actions: list[int] = []
    remaining_targets: list[int] = []
    puzzle_heights: list[int] = []
    puzzle_widths: list[int] = []
    puzzle_indices: list[int] = []
    profile: dict[str, float] = {
        "dataset_materialization_time_s": 0.0,
        "puzzle_parse_time_s": 0.0,
        "state_encode_time_s": 0.0,
        "env_step_time_s": 0.0,
    }
    materialize_start = time.perf_counter()
    for puzzle_idx, trajectory in enumerate(tqdm(trajectories, desc="materialize cache", unit="puzzle")):
        parse_start = time.perf_counter()
        puzzle = PushWorldPuzzle(str(trajectory.puzzle_path))
        profile["puzzle_parse_time_s"] += time.perf_counter() - parse_start
        puzzle_width, puzzle_height = puzzle.dimensions
        state = puzzle.initial_state
        plan_length = len(trajectory.plan)
        for step_idx, action_char in enumerate(trajectory.plan):
            encode_start = time.perf_counter()
            planes = encode_state(puzzle, state, height, width)
            profile["state_encode_time_s"] += time.perf_counter() - encode_start
            states.append(torch.from_numpy(planes.astype(np.uint8)))
            action = Actions.FROM_CHAR[action_char]
            actions.append(action)
            remaining_targets.append(plan_length - step_idx)
            puzzle_heights.append(puzzle_height)
            puzzle_widths.append(puzzle_width)
            puzzle_indices.append(puzzle_idx)
            step_start = time.perf_counter()
            state = puzzle.get_next_state(state, action)
            profile["env_step_time_s"] += time.perf_counter() - step_start
        if not puzzle.is_goal_state(state):
            raise ValueError(f"Planner trace does not solve {trajectory.puzzle_path}")
    profile["dataset_materialization_time_s"] = time.perf_counter() - materialize_start

    payload = {
        "states": torch.stack(states) if states else torch.zeros((0, 7, height, width), dtype=torch.uint8),
        "actions": torch.tensor(actions, dtype=torch.long),
        "remaining": torch.tensor(remaining_targets, dtype=torch.long),
        "puzzle_heights": torch.tensor(puzzle_heights, dtype=torch.int16),
        "puzzle_widths": torch.tensor(puzzle_widths, dtype=torch.int16),
        "puzzle_indices": torch.tensor(puzzle_indices, dtype=torch.long),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, data_path = cache_files(cache_dir)
    torch.save(payload, data_path)
    manifest: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "height": height,
        "width": width,
        "channels": 7,
        "examples": len(actions),
        "puzzles": [
            {
                "path": project_relative_path(trajectory.puzzle_path),
                "sha256": file_sha256(trajectory.puzzle_path),
                "plan": trajectory.plan,
                "plan_length": len(trajectory.plan),
                "solve_time_s": trajectory.solve_time_s,
            }
            for trajectory in trajectories
        ],
        "build_profile": profile,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    size_bytes = cache_dir_size_bytes(cache_dir)
    manifest["size_bytes"] = size_bytes
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    dataset = CachedExpertDataset(
        states=payload["states"],
        actions=payload["actions"],
        remaining=payload["remaining"],
        puzzle_heights=payload["puzzle_heights"],
        puzzle_widths=payload["puzzle_widths"],
        puzzle_indices=payload["puzzle_indices"],
        puzzle_paths=[trajectory.puzzle_path for trajectory in trajectories],
        height=height,
        width=width,
        transforms=transforms,
        transform_level0_only=transform_level0_only,
        seed=seed,
        profile=profile,
    )
    cache_profile = {
        "hit": False,
        "cache_dir": str(cache_dir),
        "load_time_s": 0.0,
        "build_time_s": time.perf_counter() - start,
        "size_bytes": size_bytes,
        "manifest_build_profile": profile,
    }
    return dataset, cache_profile


def train(
    model: nn.Module,
    dataset: ExpertDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    distance_loss_weight: float,
    quick_eval_paths: list[Path],
    height: int,
    width: int,
    max_steps: int,
    beam_width: int,
    beam_depth: int,
    top_k: int,
    max_cache_entries: int,
    repeat_penalty: float,
    distance_target: str,
    beam_score: str,
    distance_weight: float,
    beam_length_normalization: float,
    closed_list_pruning: bool,
    quick_eval_every: int,
    log_every_batches: int,
    amp: bool,
    seed: int,
    start_epoch: int = 0,
    initial_global_step: int = 0,
    optimizer_state_dict: dict[str, object] | None = None,
    scaler_state_dict: dict[str, object] | None = None,
    writer: object | None = None,
    checkpoint_callback: Callable[
        [int, float, nn.Module, torch.optim.Optimizer, torch.amp.GradScaler, int, str | None],
        None,
    ]
    | None = None,
    profile: dict[str, float | int] | None = None,
) -> list[float]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scaler_state_dict is not None:
        scaler.load_state_dict(scaler_state_dict)
    if profile is not None:
        profile.setdefault("dataloader_wait_time_s", 0.0)
        profile.setdefault("forward_backward_update_time_s", 0.0)
        profile.setdefault("optimizer_steps", 0)
        profile.setdefault("examples", 0)
        profile.setdefault("epochs", 0)
    losses = []
    global_step = initial_global_step
    model.train()
    current_epoch = 0
    total_loss = 0.0
    total_count = 0
    try:
        progress = tqdm(range(start_epoch + 1, epochs + 1), desc="train", unit="epoch")
        for epoch in progress:
            current_epoch = epoch
            total_loss = 0.0
            total_count = 0
            loader_iter = iter(loader)
            with tqdm(total=len(loader), desc=f"epoch {current_epoch}/{epochs}", unit="batch", leave=False) as batch_progress:
                while True:
                    wait_start = time.perf_counter()
                    try:
                        states, actions, remaining = next(loader_iter)
                    except StopIteration:
                        break
                    if profile is not None:
                        profile["dataloader_wait_time_s"] = float(profile["dataloader_wait_time_s"]) + (
                            time.perf_counter() - wait_start
                        )
                    global_step += 1
                    step_start = time.perf_counter()
                    states = states.to(device)
                    actions = actions.to(device)
                    remaining = distance_targets(
                        remaining.to(device),
                        model.distance_head.out_features,
                        distance_target,
                    )
                    with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                        action_logits, distance_logits = model(states)
                        action_loss = nn.functional.cross_entropy(action_logits, actions)
                        distance_loss = nn.functional.cross_entropy(distance_logits, remaining)
                        loss = action_loss + distance_loss_weight * distance_loss
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if profile is not None:
                        profile["forward_backward_update_time_s"] = float(
                            profile["forward_backward_update_time_s"]
                        ) + (time.perf_counter() - step_start)
                        profile["optimizer_steps"] = int(profile["optimizer_steps"]) + 1
                        profile["examples"] = int(profile["examples"]) + int(states.shape[0])
                    total_loss += float(loss.detach().cpu()) * states.shape[0]
                    total_count += states.shape[0]
                    batch_loss = float(loss.detach().cpu())
                    batch_progress.set_postfix(loss=f"{batch_loss:.4f}")
                    batch_progress.update(1)
                    if writer is not None and log_every_batches > 0 and global_step % log_every_batches == 0:
                        writer.add_scalar("train/batch_loss", batch_loss, global_step)
                        writer.add_scalar("train/action_loss", float(action_loss.detach().cpu()), global_step)
                        writer.add_scalar("train/distance_loss", float(distance_loss.detach().cpu()), global_step)
            epoch_loss = total_loss / max(1, total_count)
            losses.append(epoch_loss)
            if profile is not None:
                profile["epochs"] = int(profile["epochs"]) + 1
            if writer is not None:
                writer.add_scalar("train/loss", epoch_loss, current_epoch)
                writer.add_scalar("train/global_step", global_step, current_epoch)
            progress.set_postfix(loss=f"{epoch_loss:.4f}")

            if quick_eval_paths and quick_eval_every > 0 and current_epoch % quick_eval_every == 0:
                quick_eval = evaluate(
                    model,
                    quick_eval_paths,
                    height,
                    width,
                    device,
                    max_steps,
                    beam_width,
                    beam_depth,
                    top_k,
                    f"quick epoch {current_epoch}",
                    max_cache_entries,
                    repeat_penalty,
                    distance_target,
                    max_steps,
                    beam_score,
                    distance_weight,
                    beam_length_normalization,
                    closed_list_pruning=closed_list_pruning,
                    leave=False,
                )
                success_rate = quick_eval["solved"] / max(1, quick_eval["total"])
                if writer is not None:
                    writer.add_scalar("eval_quick/level0_solved", quick_eval["solved"], current_epoch)
                    writer.add_scalar("eval_quick/level0_total", quick_eval["total"], current_epoch)
                    writer.add_scalar("eval_quick/level0_success_rate", success_rate, current_epoch)
                model.train()
            if checkpoint_callback is not None:
                checkpoint_callback(current_epoch, epoch_loss, model, optimizer, scaler, global_step, None)
    except KeyboardInterrupt as exc:
        partial_loss = total_loss / total_count if total_count > 0 else (losses[-1] if losses else float("nan"))
        interrupt_epoch = max(current_epoch, start_epoch + len(losses) + 1)
        if checkpoint_callback is not None:
            checkpoint_callback(interrupt_epoch, partial_loss, model, optimizer, scaler, global_step, "interrupted")
        raise TrainingInterrupted(losses) from exc
    return losses


def make_tensorboard_writer(log_dir: Path | None) -> object | None:
    if log_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging requested, but tensorboard is not installed. "
            "Install it with: uv pip install tensorboard"
        ) from exc
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def evaluate(
    model: nn.Module,
    puzzle_paths: list[Path],
    height: int,
    width: int,
    device: torch.device,
    max_steps: int,
    beam_width: int,
    beam_depth: int,
    top_k: int,
    label: str,
    max_cache_entries: int,
    repeat_penalty: float = 0.0,
    distance_target: str = "linear",
    distance_max_steps: int | None = None,
    beam_score: str = "policy_distance",
    distance_weight: float = 0.15,
    beam_length_normalization: float = 0.0,
    closed_list_pruning: bool = False,
    leave: bool = True,
) -> dict[str, object]:
    model.eval()
    solved = 0
    results = []
    encode_cache: dict[tuple[str, tuple[tuple[int, int], ...]], torch.Tensor] = {}
    prediction_cache: dict[tuple[str, tuple[tuple[int, int], ...]], tuple[torch.Tensor, torch.Tensor]] = {}
    profile = RolloutProfile()
    start = time.perf_counter()
    with torch.inference_mode():
        progress = tqdm(puzzle_paths, desc=f"eval {label}", unit="puzzle", leave=leave)
        for path in progress:
            parse_start = time.perf_counter()
            puzzle = PushWorldPuzzle(str(path))
            profile.puzzle_parse_time_s += time.perf_counter() - parse_start
            state = puzzle.initial_state
            actions: list[str] = []
            repeated_states = 0
            seen = {state}
            for _ in range(max_steps):
                if puzzle.is_goal_state(state):
                    break
                action = choose_action(
                    model,
                    puzzle,
                    state,
                    height,
                    width,
                    device,
                    beam_width=beam_width,
                    beam_depth=beam_depth,
                    top_k=top_k,
                    puzzle_key=str(path),
                    encode_cache=encode_cache,
                    max_cache_entries=max_cache_entries,
                    seen_states=seen,
                    repeat_penalty=repeat_penalty,
                    distance_target=distance_target,
                    distance_max_steps=distance_max_steps,
                    beam_score=beam_score,
                    distance_weight=distance_weight,
                    beam_length_normalization=beam_length_normalization,
                    prediction_cache=prediction_cache,
                    profile=profile,
                    closed_list_pruning=closed_list_pruning,
                )
                actions.append(ACTION_CHARS[action])
                step_start = time.perf_counter()
                state = puzzle.get_next_state(state, action)
                profile.env_step_time_s += time.perf_counter() - step_start
                if state in seen:
                    repeated_states += 1
                seen.add(state)
            did_solve = puzzle.is_goal_state(state)
            solved += int(did_solve)
            elapsed = max(time.perf_counter() - start, 1e-9)
            results.append(
                {
                    "puzzle": path.name,
                    "solved": did_solve,
                    "steps": len(actions),
                    "actions": "".join(actions),
                    "repeated_states": repeated_states,
                }
            )
            rate = len(results) / elapsed
            remaining = (len(puzzle_paths) - len(results)) / rate if rate > 0 else 0.0
            progress.set_postfix(
                solved=f"{solved}/{len(results)}",
                eta_s=f"{remaining:.0f}",
            )
    elapsed = time.perf_counter() - start
    profile.eval_loop_time_s = elapsed
    return {
        "solved": solved,
        "total": len(puzzle_paths),
        "time_s": elapsed,
        "solves_per_minute": solved * 60.0 / max(elapsed, 1e-9),
        "cache_entries": len(encode_cache),
        "prediction_cache_entries": len(prediction_cache),
        "repeat_penalty": repeat_penalty,
        "distance_target": distance_target,
        "beam_score": beam_score,
        "distance_weight": distance_weight,
        "beam_length_normalization": beam_length_normalization,
        "closed_list_pruning": closed_list_pruning,
        "profile": profile.to_dict(),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir",
        type=Path,
        action="append",
        default=None,
        help="Training puzzle directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        action="append",
        default=None,
        help="Held-out Level 0 puzzle directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--level1-dir",
        type=Path,
        default=PROJECT_ROOT / "external/pushworld/benchmark/puzzles/level1",
    )
    parser.add_argument(
        "--planner",
        type=Path,
        default=default_planner_path(),
    )
    parser.add_argument("--train-puzzles", type=int, default=5, help="Number of train puzzles to use unless --all-train is set.")
    parser.add_argument("--test-puzzles", type=int, default=10, help="Number of held-out Level 0 puzzles to evaluate unless --all-test is set.")
    parser.add_argument("--level1-puzzles", type=int, default=5, help="Number of Level 1 puzzles to evaluate unless --all-level1 is set.")
    parser.add_argument("--all-train", action="store_true", help="Use all train puzzles.")
    parser.add_argument("--all-test", action="store_true", help="Evaluate all held-out Level 0 puzzles.")
    parser.add_argument("--all-level1", action="store_true", help="Evaluate all Level 1 puzzles.")
    parser.add_argument(
        "--level0-symmetry-augment",
        action="store_true",
        help="Augment train examples in memory with Level-0 rotation/flip symmetries.",
    )
    parser.add_argument(
        "--augment-transforms",
        default="all",
        help=(
            "Comma-separated transform names, or 'all'. Available: "
            + ",".join(SYMMETRY_TRANSFORMS)
        ),
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--distance-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--distance-target",
        choices=DISTANCE_TARGETS,
        default="log",
        help="Remaining-step target encoding for the auxiliary value head.",
    )
    parser.add_argument(
        "--distance-bins",
        type=int,
        default=0,
        help="Value-head bins. 0 picks max_steps+1 for linear or a compact log scale for log targets.",
    )
    parser.add_argument(
        "--encoder-stem",
        choices=["linear", "conv"],
        default="conv",
        help="Use a linear per-cell projection or a local convolutional board encoder before the transformer.",
    )
    parser.add_argument("--dropout", type=float, default=0.01, help="Transformer dropout probability.")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision during training.")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--beam-depth", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=0.0,
        help="Add this beam cost when a rollout candidate revisits a state already seen in the current rollout.",
    )
    parser.add_argument(
        "--beam-score",
        choices=BEAM_SCORE_MODES,
        default="policy_distance",
        help="Beam ranking objective: policy cost, value distance, or the weighted combination.",
    )
    parser.add_argument(
        "--distance-weight",
        type=float,
        default=0.15,
        help="Weight for the auxiliary distance estimate in policy_distance beam scoring.",
    )
    parser.add_argument(
        "--beam-length-normalization",
        type=float,
        default=0.0,
        help="Divide cumulative policy cost by path_length^N before beam ranking; 0 preserves old scoring.",
    )
    parser.add_argument(
        "--closed-list-pruning",
        action="store_true",
        help="Drop beam candidates that revisit states already seen in the current rollout.",
    )
    parser.add_argument("--eval-every", type=int, default=0, help="Run quick held-out Level 0 eval every N epochs; 0 disables it.")
    parser.add_argument("--eval-puzzles", type=int, default=50, help="Number of held-out Level 0 puzzles for periodic quick eval.")
    parser.add_argument("--log-every-batches", type=int, default=10, help="Log batch losses to TensorBoard every N optimizer steps; 0 disables batch logging.")
    parser.add_argument("--max-cache-entries", type=int, default=250_000)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Persistent planner-imitation tensor cache directory. Existing valid caches skip RGD and base-state encoding.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild --cache-dir even if manifest.json and data.pt already exist.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model weights, optimizer/scaler state, epoch, and global step from a saved checkpoint.",
    )
    parser.add_argument(
        "--epoch-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-epoch checkpoints. If omitted and --model-output "
            "is set, uses '<model-output stem>_epochs' next to --model-output."
        ),
    )
    parser.add_argument(
        "--no-epoch-checkpoints",
        action="store_true",
        help="Disable automatic per-epoch checkpoints.",
    )
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    parser.add_argument("--skip-train-eval", action="store_true")
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Skip train/test/Level-1 rollout evaluation after training and save the checkpoint immediately.",
    )
    parser.add_argument("--print-expert-plans", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--planner-time-limit", type=float, default=10.0)
    parser.add_argument("--planner-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.planner_workers < 1:
        raise ValueError("--planner-workers must be >= 1")
    if args.repeat_penalty < 0.0:
        raise ValueError("--repeat-penalty must be >= 0")
    if args.distance_bins < 0:
        raise ValueError("--distance-bins must be >= 0")
    if args.distance_weight < 0.0:
        raise ValueError("--distance-weight must be >= 0")
    if args.beam_length_normalization < 0.0:
        raise ValueError("--beam-length-normalization must be >= 0")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.rebuild_cache and args.cache_dir is None:
        raise ValueError("--rebuild-cache requires --cache-dir")

    if args.augment_transforms == "all":
        train_transforms = SYMMETRY_TRANSFORMS if args.level0_symmetry_augment else ("r0",)
    else:
        train_transforms = tuple(name.strip() for name in args.augment_transforms.split(",") if name.strip())
        unknown_transforms = sorted(set(train_transforms) - set(SYMMETRY_TRANSFORMS))
        if unknown_transforms:
            raise ValueError(f"Unknown --augment-transforms values: {unknown_transforms}")
        if not args.level0_symmetry_augment:
            train_transforms = ("r0",)

    train_dirs = args.train_dir or [PROJECT_ROOT / "data/level0/base/train"]
    test_dirs = args.test_dir or [PROJECT_ROOT / "data/level0/base/test"]
    level1_dirs = [args.level1_dir]

    if len(train_dirs) > 1 and not args.all_train and args.train_puzzles == 5:
        raise ValueError(
            "Multiple --train-dir values were provided, but --all-train was not set "
            "and --train-puzzles is still the default 5. Pass --all-train for the "
            "full run, or set --train-puzzles explicitly for a small mixed smoke."
        )

    train_paths = select_puzzles(train_dirs, args.train_puzzles, args.all_train)
    test_paths = select_puzzles(test_dirs, args.test_puzzles, args.all_test)
    level1_paths = select_puzzles(level1_dirs, args.level1_puzzles, args.all_level1)
    all_paths = train_paths + test_paths + level1_paths
    height, width = max_dimensions(all_paths)
    if args.level0_symmetry_augment:
        max_side = max(height, width)
        height = max_side
        width = max_side

    print(f"device cuda={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train_puzzles={len(train_paths)} test_puzzles={len(test_paths)} level1={len(level1_paths)}")
    print("train_dirs=" + json.dumps([str(path) for path in train_dirs], indent=2))
    print("test_dirs=" + json.dumps([str(path) for path in test_dirs], indent=2))
    print(f"level0_symmetry_augment={args.level0_symmetry_augment} transforms={list(train_transforms)}")
    print(
        f"board={height}x{width} planner={args.planner} "
        f"planner_workers={args.planner_workers} repeat_penalty={args.repeat_penalty}"
    )
    distance_bins = args.distance_bins or auto_distance_bins(args.max_steps, args.distance_target)
    print(
        f"model encoder_stem={args.encoder_stem} dropout={args.dropout} "
        f"distance_target={args.distance_target} distance_bins={distance_bins} "
        f"beam_score={args.beam_score} distance_weight={args.distance_weight} "
        f"beam_length_normalization={args.beam_length_normalization}"
    )
    writer = make_tensorboard_writer(args.tensorboard_log)
    if writer is not None:
        writer.add_text("config/args", json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    cache_profile: dict[str, object] = {
        "enabled": args.cache_dir is not None,
        "hit": False,
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "load_time_s": 0.0,
        "build_time_s": 0.0,
        "size_bytes": 0,
    }
    solve_time = 0.0
    if args.cache_dir is not None and cache_exists(args.cache_dir) and not args.rebuild_cache:
        trajectories, dataset, loaded_cache_profile = load_planner_imitation_cache(
            cache_dir=args.cache_dir,
            puzzle_paths=train_paths,
            height=height,
            width=width,
            transforms=train_transforms,
            transform_level0_only=args.level0_symmetry_augment,
            seed=args.seed,
        )
        cache_profile.update(loaded_cache_profile)
        print(
            "cache_summary="
            + json.dumps(
                {
                    "hit": True,
                    "cache_dir": str(args.cache_dir),
                    "load_time_s": round(float(cache_profile["load_time_s"]), 3),
                    "size_mb": round(float(cache_profile["size_bytes"]) / 1024 / 1024, 2),
                },
                indent=2,
            )
        )
    else:
        solve_start = time.perf_counter()
        trajectories = solve_trajectories(args.planner, train_paths, args.planner_time_limit, args.planner_workers)
        solve_time = time.perf_counter() - solve_start
        if args.cache_dir is not None:
            dataset, built_cache_profile = build_planner_imitation_cache(
                cache_dir=args.cache_dir,
                trajectories=trajectories,
                height=height,
                width=width,
                transforms=train_transforms,
                transform_level0_only=args.level0_symmetry_augment,
                seed=args.seed,
            )
            cache_profile.update(built_cache_profile)
            print(
                "cache_summary="
                + json.dumps(
                    {
                        "hit": False,
                        "cache_dir": str(args.cache_dir),
                        "build_time_s": round(float(cache_profile["build_time_s"]), 3),
                        "size_mb": round(float(cache_profile["size_bytes"]) / 1024 / 1024, 2),
                    },
                    indent=2,
                )
            )
        else:
            dataset = ExpertDataset(
                trajectories,
                height=height,
                width=width,
                transforms=train_transforms,
                transform_level0_only=args.level0_symmetry_augment,
                seed=args.seed,
            )
    expert_plan_summary = [
        {"puzzle": t.puzzle_path.name, "plan": t.plan, "solve_time_s": round(t.solve_time_s, 4)}
        for t in trajectories
    ]
    if args.print_expert_plans:
        print("expert_plans=" + json.dumps(expert_plan_summary, indent=2))
    else:
        plan_lengths = [len(t.plan) for t in trajectories]
        print(
            "expert_trace_summary="
            + json.dumps(
                {
                    "count": len(trajectories),
                    "total_actions": sum(plan_lengths),
                    "mean_plan_len": round(sum(plan_lengths) / max(1, len(plan_lengths)), 2),
                    "max_plan_len": max(plan_lengths) if plan_lengths else 0,
                    "solve_time_s": round(solve_time, 3),
                },
                indent=2,
            )
        )

    dataset_profile = dict(getattr(dataset, "profile", {}))
    print(
        "dataset_summary="
        + json.dumps(
            {
                "base_examples": dataset.base_examples,
                "augmented_examples": len(dataset),
                "augmentation_factor": round(len(dataset) / max(1, dataset.base_examples), 2),
                "level0_augmented_base_examples": dataset.augmented_base_examples,
                "unaugmented_base_examples": dataset.unaugmented_base_examples,
                "transforms": list(train_transforms),
                "profile": dataset_profile,
            },
            indent=2,
        )
    )
    if writer is not None:
        writer.add_scalar("data/train_puzzles", len(train_paths), 0)
        writer.add_scalar("data/base_examples", dataset.base_examples, 0)
        writer.add_scalar("data/level0_augmented_base_examples", dataset.augmented_base_examples, 0)
        writer.add_scalar("data/unaugmented_base_examples", dataset.unaugmented_base_examples, 0)
        writer.add_scalar("data/examples", len(dataset), 0)
        writer.add_scalar("data/augmentation_factor", len(dataset) / max(1, dataset.base_examples), 0)
        writer.add_scalar("time/expert_solve_s", solve_time, 0)
        writer.add_scalar("time/cache_load_s", float(cache_profile["load_time_s"]), 0)
        writer.add_scalar("time/cache_build_s", float(cache_profile["build_time_s"]), 0)
    model = BoardTransformerPolicy(
        channels=7,
        height=height,
        width=width,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        distance_bins=distance_bins,
        encoder_stem=args.encoder_stem,
        dropout=args.dropout,
    ).to(device)
    resume_epoch = 0
    resume_global_step = 0
    resume_optimizer_state: dict[str, object] | None = None
    resume_scaler_state: dict[str, object] | None = None
    if args.resume_checkpoint is not None:
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_height = int(resume_checkpoint["height"])
        checkpoint_width = int(resume_checkpoint["width"])
        if checkpoint_height != height or checkpoint_width != width:
            raise ValueError(
                f"Resume checkpoint board {checkpoint_height}x{checkpoint_width} does not match "
                f"current board {height}x{width}. Use the same train/eval dirs and augmentation padding."
            )
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        resume_epoch = int(resume_checkpoint.get("epoch", 0) or 0)
        resume_global_step = int(resume_checkpoint.get("global_step", 0) or 0)
        resume_optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        resume_scaler_state = resume_checkpoint.get("scaler_state_dict")
        print(
            f"resumed checkpoint={args.resume_checkpoint} "
            f"epoch={resume_epoch} global_step={resume_global_step} "
            f"optimizer={'yes' if resume_optimizer_state is not None else 'no'} "
            f"scaler={'yes' if resume_scaler_state is not None else 'no'}"
        )
        if resume_epoch >= args.epochs:
            print(f"resume epoch {resume_epoch} is >= requested --epochs {args.epochs}; no training batches will run")
    quick_eval_paths = test_paths[: args.eval_puzzles] if args.eval_every > 0 else []

    epoch_checkpoint_dir = args.epoch_checkpoint_dir
    if epoch_checkpoint_dir is None and args.model_output is not None and not args.no_epoch_checkpoints:
        epoch_checkpoint_dir = args.model_output.parent / f"{args.model_output.stem}_epochs"

    def save_checkpoint(
        path: Path,
        epoch: int | None = None,
        epoch_loss: float | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        global_step: int | None = None,
        interrupted: bool = False,
    ) -> None:
        payload: dict[str, object] = {
            "model_state_dict": model.state_dict(),
            "height": height,
            "width": width,
            "channels": 7,
            "args": vars(args),
        }
        if epoch is not None:
            payload["epoch"] = epoch
        if epoch_loss is not None:
            payload["epoch_loss"] = epoch_loss
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scaler is not None:
            payload["scaler_state_dict"] = scaler.state_dict()
        if global_step is not None:
            payload["global_step"] = global_step
        payload["interrupted"] = interrupted
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def save_epoch_checkpoint(
        epoch: int,
        epoch_loss: float,
        _model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        global_step: int,
        tag: str | None,
    ) -> None:
        del _model
        if epoch_checkpoint_dir is None or args.no_epoch_checkpoints:
            return
        filename = f"{tag}_epoch_{epoch:03d}.pt" if tag else f"epoch_{epoch:03d}.pt"
        path = epoch_checkpoint_dir / filename
        save_checkpoint(path, epoch, epoch_loss, optimizer, scaler, global_step, interrupted=tag == "interrupted")
        if writer is not None:
            writer.add_text("checkpoint/latest_epoch", str(path), epoch)
        print(f"wrote {path}")

    train_profile: dict[str, float | int] = {}
    train_start = time.perf_counter()
    interrupted = False
    try:
        losses = train(
            model,
            dataset,
            device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.distance_loss_weight,
            quick_eval_paths,
            height,
            width,
            args.max_steps,
            args.beam_width,
            args.beam_depth,
            args.top_k,
            args.max_cache_entries,
            args.repeat_penalty,
            args.distance_target,
            args.beam_score,
            args.distance_weight,
            args.beam_length_normalization,
            args.closed_list_pruning,
            args.eval_every,
            args.log_every_batches,
            args.amp,
            args.seed,
            resume_epoch,
            resume_global_step,
            resume_optimizer_state,
            resume_scaler_state,
            writer,
            save_epoch_checkpoint,
            train_profile,
        )
    except TrainingInterrupted as exc:
        losses = exc.losses
        interrupted = True
    train_time = time.perf_counter() - train_start
    train_profile["total_time_s"] = train_time

    if interrupted:
        print("training interrupted; saved interrupt checkpoint and skipped final evaluation")
        if writer is not None:
            writer.add_scalar("time/train_s", train_time, 0)
            writer.flush()
            writer.close()
        return

    if args.skip_final_eval:
        train_eval = {"skipped": True, "solved": None, "total": len(train_paths)}
        test_eval = {"skipped": True, "solved": None, "total": len(test_paths)}
        level1_eval = {"skipped": True, "solved": None, "total": len(level1_paths)}
    else:
        if args.skip_train_eval:
            train_eval = {"skipped": True, "solved": None, "total": len(train_paths)}
        else:
            train_eval = evaluate(
                model,
                train_paths,
                height,
                width,
                device,
                args.max_steps,
                args.beam_width,
                args.beam_depth,
                args.top_k,
                "train",
                args.max_cache_entries,
                args.repeat_penalty,
                args.distance_target,
                args.max_steps,
                args.beam_score,
                args.distance_weight,
                args.beam_length_normalization,
                closed_list_pruning=args.closed_list_pruning,
            )
        test_eval = evaluate(
            model,
            test_paths,
            height,
            width,
            device,
            args.max_steps,
            args.beam_width,
            args.beam_depth,
            args.top_k,
            "level0 test",
            args.max_cache_entries,
            args.repeat_penalty,
            args.distance_target,
            args.max_steps,
            args.beam_score,
            args.distance_weight,
            args.beam_length_normalization,
            closed_list_pruning=args.closed_list_pruning,
        )
        level1_eval = evaluate(
            model,
            level1_paths,
            height,
            width,
            device,
            args.max_steps,
            args.beam_width,
            args.beam_depth,
            args.top_k,
            "level1",
            args.max_cache_entries,
            args.repeat_penalty,
            args.distance_target,
            args.max_steps,
            args.beam_score,
            args.distance_weight,
            args.beam_length_normalization,
            closed_list_pruning=args.closed_list_pruning,
        )

    def compact_eval(payload: dict[str, object]) -> dict[str, object]:
        if payload.get("skipped"):
            return payload
        if args.verbose:
            return payload
        return {
            "solved": payload["solved"],
            "total": payload["total"],
            "success_rate": payload["solved"] / max(1, payload["total"]),
            "time_s": payload.get("time_s"),
            "solves_per_minute": payload.get("solves_per_minute"),
            "cache_entries": payload.get("cache_entries"),
            "prediction_cache_entries": payload.get("prediction_cache_entries"),
            "profile": payload.get("profile"),
            "solved_puzzles": [
                result["puzzle"]
                for result in payload["results"]
                if result["solved"]
            ],
        }

    summary = {
        "examples": len(dataset),
        "base_examples": dataset.base_examples,
        "augmentation_factor": round(len(dataset) / max(1, dataset.base_examples), 2),
        "level0_augmented_base_examples": dataset.augmented_base_examples,
        "unaugmented_base_examples": dataset.unaugmented_base_examples,
        "transforms": list(train_transforms),
        "solve_time_s": round(solve_time, 3),
        "train_time_s": round(train_time, 3),
        "final_loss": round(losses[-1], 6) if losses else None,
        "beam_width": args.beam_width,
        "beam_depth": args.beam_depth,
        "top_k": args.top_k,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "layers": args.layers,
        "encoder_stem": args.encoder_stem,
        "dropout": args.dropout,
        "distance_target": args.distance_target,
        "distance_bins": distance_bins,
        "amp": args.amp,
        "seed": args.seed,
        "planner_workers": args.planner_workers,
        "repeat_penalty": args.repeat_penalty,
        "beam_score": args.beam_score,
        "distance_weight": args.distance_weight,
        "beam_length_normalization": args.beam_length_normalization,
        "closed_list_pruning": args.closed_list_pruning,
        "max_cache_entries": args.max_cache_entries,
        "profile": {
            "schema_version": 1,
            "cache": cache_profile,
            "data": {
                "rgd_solve_time_s": solve_time,
                **dataset_profile,
            },
            "train": train_profile,
        },
        "train_eval": compact_eval(train_eval),
        "level0_test_eval": compact_eval(test_eval),
        "level1_eval": compact_eval(level1_eval),
        "runtime": runtime_metadata(device, args.amp),
    }
    if writer is not None:
        train_solved = train_eval["solved"] or 0
        test_solved = test_eval["solved"] or 0
        level1_solved = level1_eval["solved"] or 0
        writer.add_scalar("time/train_s", train_time, 0)
        writer.add_scalar("eval/train_solved", train_solved, 0)
        writer.add_scalar("eval/train_total", train_eval["total"], 0)
        writer.add_scalar("eval/level0_test_solved", test_solved, 0)
        writer.add_scalar("eval/level0_test_total", test_eval["total"], 0)
        writer.add_scalar("eval/level1_solved", level1_solved, 0)
        writer.add_scalar("eval/level1_total", level1_eval["total"], 0)
        writer.add_scalar(
            "eval/level0_test_success_rate",
            test_solved / max(1, test_eval["total"]),
            0,
        )
        writer.add_scalar(
            "eval/level1_success_rate",
            level1_solved / max(1, level1_eval["total"]),
            0,
        )
        writer.flush()
        writer.close()
    print("summary=" + json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    if args.model_output is not None:
        save_checkpoint(args.model_output)
        print(f"wrote {args.model_output}")


if __name__ == "__main__":
    main()
