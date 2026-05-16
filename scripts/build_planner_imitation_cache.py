from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_planner_imitation_v2 import (  # noqa: E402
    SYMMETRY_TRANSFORMS,
    build_planner_imitation_cache,
    cache_exists,
    max_dimensions,
    planner_imitation_cache_key,
    select_puzzles,
    solve_trajectories,
)
from pushworld_study.paths import PROJECT_ROOT as PACKAGE_PROJECT_ROOT  # noqa: E402


def default_planner_path() -> Path:
    base = PACKAGE_PROJECT_ROOT / "external/pushworld/cpp/build/bin/run_planner"
    exe = base.with_suffix(".exe")
    return exe if exe.exists() else base


def parse_transforms(value: str, level0_symmetry_augment: bool) -> tuple[str, ...]:
    if value == "all":
        return SYMMETRY_TRANSFORMS if level0_symmetry_augment else ("r0",)
    transforms = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = sorted(set(transforms) - set(SYMMETRY_TRANSFORMS))
    if unknown:
        raise ValueError(f"Unknown --augment-transforms values: {unknown}")
    return transforms if level0_symmetry_augment else ("r0",)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir",
        type=Path,
        action="append",
        default=None,
        help="Training puzzle directory. Can be passed multiple times.",
    )
    parser.add_argument("--train-puzzles", type=int, default=5)
    parser.add_argument("--all-train", action="store_true")
    parser.add_argument("--planner", type=Path, default=default_planner_path())
    parser.add_argument("--planner-time-limit", type=float, default=10.0)
    parser.add_argument("--planner-workers", type=int, default=1)
    parser.add_argument("--level0-symmetry-augment", action="store_true")
    parser.add_argument("--augment-transforms", default="all")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--print-expert-plans", action="store_true")
    args = parser.parse_args()

    if args.planner_workers < 1:
        raise ValueError("--planner-workers must be >= 1")

    train_dirs = args.train_dir or [PACKAGE_PROJECT_ROOT / "data/level0/base/train"]
    train_paths = select_puzzles(train_dirs, args.train_puzzles, args.all_train)
    if not train_paths:
        raise ValueError("No training puzzles selected")
    height, width = max_dimensions(train_paths)
    if args.level0_symmetry_augment:
        max_side = max(height, width)
        height = max_side
        width = max_side
    transforms = parse_transforms(args.augment_transforms, args.level0_symmetry_augment)

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_key = planner_imitation_cache_key(train_paths, height, width)
        cache_dir = PACKAGE_PROJECT_ROOT / "data/cache/planner_imitation" / cache_key
    if cache_exists(cache_dir) and not args.overwrite:
        raise FileExistsError(f"Cache already exists at {cache_dir}; pass --overwrite to rebuild it")

    trajectories = solve_trajectories(args.planner, train_paths, args.planner_time_limit, args.planner_workers)
    dataset, cache_profile = build_planner_imitation_cache(
        cache_dir=cache_dir,
        trajectories=trajectories,
        height=height,
        width=width,
        transforms=transforms,
        transform_level0_only=args.level0_symmetry_augment,
        seed=args.seed,
    )
    plan_lengths = [len(trajectory.plan) for trajectory in trajectories]
    summary = {
        "cache_dir": str(cache_dir),
        "height": height,
        "width": width,
        "train_dirs": [str(path) for path in train_dirs],
        "train_puzzles": len(train_paths),
        "examples": len(dataset),
        "base_examples": dataset.base_examples,
        "total_actions": sum(plan_lengths),
        "mean_plan_len": sum(plan_lengths) / max(1, len(plan_lengths)),
        "max_plan_len": max(plan_lengths) if plan_lengths else 0,
        "transforms": list(transforms),
        "level0_symmetry_augment": args.level0_symmetry_augment,
        "cache": cache_profile,
    }
    if args.print_expert_plans:
        summary["expert_plans"] = [
            {
                "puzzle": trajectory.puzzle_path.name,
                "plan": trajectory.plan,
                "solve_time_s": trajectory.solve_time_s,
            }
            for trajectory in trajectories
        ]

    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
