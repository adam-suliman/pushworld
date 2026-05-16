from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from planner_imitation_rollout import auto_distance_bins, distance_targets  # noqa: E402
from train_planner_imitation_v2 import (  # noqa: E402
    BoardTransformerPolicy,
    ExpertDataset,
    Trajectory,
    evaluate,
    max_dimensions,
    set_seed,
)
from pushworld_study.paths import ensure_upstream_pushworld_on_path  # noqa: E402


ensure_upstream_pushworld_on_path()


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    device: str
    encoder_stem: str
    distance_target: str
    dropout: float
    d_model: int
    layers: int
    epochs: int


class MaterializedExpertDataset(Dataset):
    def __init__(self, source: ExpertDataset) -> None:
        self.examples = [source[idx] for idx in range(len(source))]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.examples[idx]


def parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        values[key.strip()] = value.strip()
    return values


def load_level1_trajectories(limit: int | None = None) -> list[Trajectory]:
    puzzle_dir = PROJECT_ROOT / "external/pushworld/benchmark/puzzles/level1"
    solution_dir = PROJECT_ROOT / "external/pushworld/benchmark/solutions/level1"
    trajectories: list[Trajectory] = []
    for solution_path in sorted(solution_dir.glob("*.yaml"), key=lambda path: path.stem.casefold()):
        values = parse_simple_yaml(solution_path)
        puzzle_name = values.get("puzzle", solution_path.stem)
        plan = values.get("plan", "")
        puzzle_path = puzzle_dir / f"{puzzle_name}.pwp"
        if not puzzle_path.exists() or not plan:
            continue
        trajectories.append(Trajectory(puzzle_path=puzzle_path, plan=plan, solve_time_s=0.0))
        if limit is not None and len(trajectories) >= limit:
            break
    return trajectories


def split_trajectories(
    trajectories: list[Trajectory],
    train_count: int,
    eval_count: int,
    seed: int,
) -> tuple[list[Trajectory], list[Trajectory]]:
    rng = random.Random(seed)
    shuffled = list(trajectories)
    rng.shuffle(shuffled)
    train = shuffled[:train_count]
    eval_ = shuffled[train_count : train_count + eval_count]
    if len(train) < train_count or len(eval_) < eval_count:
        raise ValueError(
            f"Not enough trajectories for train_count={train_count}, eval_count={eval_count}; "
            f"available={len(trajectories)}"
        )
    return train, eval_


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def evaluate_action_accuracy(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    distance_target: str,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    loss_sum = 0.0
    distance_loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for states, actions, remaining in loader:
            states = states.to(device)
            actions = actions.to(device)
            remaining = distance_targets(
                remaining.to(device),
                model.distance_head.out_features,
                distance_target,
            )
            action_logits, distance_logits = model(states)
            action_loss = nn.functional.cross_entropy(action_logits, actions, reduction="sum")
            distance_loss = nn.functional.cross_entropy(distance_logits, remaining, reduction="sum")
            loss_sum += float(action_loss.cpu())
            distance_loss_sum += float(distance_loss.cpu())
            correct += int((torch.argmax(action_logits, dim=-1) == actions).sum().cpu())
            total += int(actions.numel())
    return {
        "action_accuracy": correct / max(1, total),
        "action_loss": loss_sum / max(1, total),
        "distance_loss": distance_loss_sum / max(1, total),
        "examples": total,
    }


def train_one(
    config: ExperimentConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    eval_paths: list[Path],
    height: int,
    width: int,
    batch_size: int,
    lr: float,
    distance_loss_weight: float,
    max_steps: int,
    beam_width: int,
    beam_depth: int,
    top_k: int,
    repeat_penalty: float,
    seed: int,
) -> dict[str, object]:
    set_seed(seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA experiment requested but torch.cuda.is_available() is false")
    device = torch.device(config.device)
    distance_bins = auto_distance_bins(max_steps, config.distance_target)
    model = BoardTransformerPolicy(
        channels=7,
        height=height,
        width=width,
        d_model=config.d_model,
        nhead=4,
        layers=config.layers,
        distance_bins=distance_bins,
        encoder_stem=config.encoder_stem,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    epoch_metrics: list[dict[str, float]] = []
    train_start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        action_loss_sum = 0.0
        distance_loss_sum = 0.0
        total = 0
        for states, actions, remaining in loader:
            states = states.to(device)
            actions = actions.to(device)
            remaining = distance_targets(
                remaining.to(device),
                model.distance_head.out_features,
                config.distance_target,
            )
            action_logits, distance_logits = model(states)
            action_loss = nn.functional.cross_entropy(action_logits, actions)
            distance_loss = nn.functional.cross_entropy(distance_logits, remaining)
            loss = action_loss + distance_loss_weight * distance_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_count = int(actions.numel())
            loss_sum += float(loss.detach().cpu()) * batch_count
            action_loss_sum += float(action_loss.detach().cpu()) * batch_count
            distance_loss_sum += float(distance_loss.detach().cpu()) * batch_count
            total += batch_count
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        epoch_metrics.append(
            {
                "epoch": float(epoch),
                "time_s": elapsed,
                "examples_per_s": total / max(elapsed, 1e-9),
                "loss": loss_sum / max(1, total),
                "action_loss": action_loss_sum / max(1, total),
                "distance_loss": distance_loss_sum / max(1, total),
            }
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_time = time.perf_counter() - train_start
    train_accuracy = evaluate_action_accuracy(
        model,
        train_dataset,
        device,
        batch_size,
        config.distance_target,
    )
    eval_accuracy = evaluate_action_accuracy(
        model,
        eval_dataset,
        device,
        batch_size,
        config.distance_target,
    )
    rollout = evaluate(
        model=model,
        puzzle_paths=eval_paths,
        height=height,
        width=width,
        device=device,
        max_steps=max_steps,
        beam_width=beam_width,
        beam_depth=beam_depth,
        top_k=top_k,
        label=config.name,
        max_cache_entries=100_000,
        repeat_penalty=repeat_penalty,
        distance_target=config.distance_target,
        distance_max_steps=max_steps,
        beam_score="policy_distance",
        distance_weight=0.15,
        beam_length_normalization=0.0,
        leave=False,
    )
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    else:
        peak_memory_mb = 0.0
    return {
        "name": config.name,
        "device": config.device,
        "encoder_stem": config.encoder_stem,
        "distance_target": config.distance_target,
        "dropout": config.dropout,
        "d_model": config.d_model,
        "layers": config.layers,
        "epochs": config.epochs,
        "distance_bins": distance_bins,
        "parameters": parameter_count(model),
        "train_time_s": train_time,
        "train_examples_per_s": len(train_dataset) * config.epochs / max(train_time, 1e-9),
        "peak_cuda_memory_mb": peak_memory_mb,
        "epoch_metrics": epoch_metrics,
        "train_accuracy": train_accuracy,
        "eval_accuracy": eval_accuracy,
        "rollout": {
            "solved": rollout["solved"],
            "total": rollout["total"],
            "success_rate": rollout["solved"] / max(1, rollout["total"]),
            "time_s": rollout["time_s"],
            "cache_entries": rollout["cache_entries"],
            "results": [
                {
                    "puzzle": item["puzzle"],
                    "solved": item["solved"],
                    "steps": item["steps"],
                    "repeated_states": item["repeated_states"],
                }
                for item in rollout["results"]
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--eval-count", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--distance-loss-weight", type=float, default=0.2)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--beam-depth", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/planner_optimization_experiments.json")
    parser.add_argument(
        "--include-cpu",
        action="store_true",
        help="Also run CPU training for the baseline model to compare device compute.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trajectories = load_level1_trajectories()
    train_trajectories, eval_trajectories = split_trajectories(
        trajectories,
        train_count=args.train_count,
        eval_count=args.eval_count,
        seed=args.seed,
    )
    all_paths = [trajectory.puzzle_path for trajectory in train_trajectories + eval_trajectories]
    height, width = max_dimensions(all_paths)
    train_dataset = MaterializedExpertDataset(
        ExpertDataset(train_trajectories, height=height, width=width, transforms=("r0",))
    )
    eval_dataset = MaterializedExpertDataset(
        ExpertDataset(eval_trajectories, height=height, width=width, transforms=("r0",))
    )
    eval_paths = [trajectory.puzzle_path for trajectory in eval_trajectories]
    device_configs = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    if args.include_cpu and "cpu" not in device_configs:
        device_configs.append("cpu")

    configs: list[ExperimentConfig] = []
    for device in device_configs:
        configs.append(
            ExperimentConfig(
                name=f"baseline_linear_exact_{device}",
                device=device,
                encoder_stem="linear",
                distance_target="linear",
                dropout=0.0,
                d_model=args.d_model,
                layers=args.layers,
                epochs=args.epochs,
            )
        )
        if device == "cuda":
            configs.append(
                ExperimentConfig(
                    name=f"optimized_conv_log_{device}",
                    device=device,
                    encoder_stem="conv",
                    distance_target="log",
                    dropout=0.01,
                    d_model=args.d_model,
                    layers=args.layers,
                    epochs=args.epochs,
                )
            )

    start = time.perf_counter()
    results = [
        train_one(
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_paths=eval_paths,
            height=height,
            width=width,
            batch_size=args.batch_size,
            lr=args.lr,
            distance_loss_weight=args.distance_loss_weight,
            max_steps=args.max_steps,
            beam_width=args.beam_width,
            beam_depth=args.beam_depth,
            top_k=args.top_k,
            repeat_penalty=args.repeat_penalty,
            seed=args.seed,
        )
        for config in configs
    ]
    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data": {
            "source": "external/pushworld benchmark level1 reference solutions",
            "train_puzzles": len(train_trajectories),
            "eval_puzzles": len(eval_trajectories),
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "height": height,
            "width": width,
            "train_puzzle_names": [trajectory.puzzle_path.name for trajectory in train_trajectories],
            "eval_puzzle_names": [trajectory.puzzle_path.name for trajectory in eval_trajectories],
        },
        "total_time_s": time.perf_counter() - start,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
