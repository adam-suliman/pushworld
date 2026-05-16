from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval_planner_imitation import evaluate_split, load_checkpoint
from planner_imitation_rollout import BEAM_SCORE_MODES, DISTANCE_TARGETS
from pushworld_study.paths import PROJECT_ROOT


def source_splits_from_manifest(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits: dict[str, str] = {}
    for row in manifest:
        if row.get("kind") != "original" or row.get("transform") != "r0":
            continue
        source = row["source_puzzle"]
        split = row["split"]
        previous = splits.get(source)
        if previous is not None and previous != split:
            raise ValueError(f"{source} appears in both {previous} and {split}")
        splits[source] = split
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--level1-dir",
        type=Path,
        default=PROJECT_ROOT / "external/pushworld/benchmark/puzzles/level1",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--beam-depth", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repeat-penalty", type=float, default=0.0)
    parser.add_argument(
        "--distance-target",
        choices=("checkpoint", *DISTANCE_TARGETS),
        default="checkpoint",
        help="Override the checkpoint's value-head target mode for rollout scoring.",
    )
    parser.add_argument("--beam-score", choices=BEAM_SCORE_MODES, default="policy_distance")
    parser.add_argument("--distance-weight", type=float, default=0.15)
    parser.add_argument("--beam-length-normalization", type=float, default=0.0)
    parser.add_argument("--max-cache-entries", type=int, default=250_000)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.repeat_penalty < 0.0:
        raise ValueError("--repeat-penalty must be >= 0")
    if args.distance_weight < 0.0:
        raise ValueError("--distance-weight must be >= 0")
    if args.beam_length_normalization < 0.0:
        raise ValueError("--beam-length-normalization must be >= 0")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, height, width, checkpoint_args = load_checkpoint(args.checkpoint, device)
    checkpoint_distance_target = str(checkpoint_args.get("distance_target", "linear"))
    distance_target = checkpoint_distance_target if args.distance_target == "checkpoint" else args.distance_target
    distance_max_steps = int(checkpoint_args.get("max_steps", args.max_steps))
    source_splits = source_splits_from_manifest(args.manifest)
    all_paths = sorted(args.level1_dir.glob("*.pwp"), key=lambda path: path.name.casefold())
    paths_by_split: dict[str, list[Path]] = {}
    missing = []
    for path in all_paths:
        split = source_splits.get(path.name)
        if split is None:
            missing.append(path.name)
            continue
        paths_by_split.setdefault(split, []).append(path)
    if missing:
        raise ValueError(f"Manifest has no split for Level-1 originals: {missing}")

    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"manifest={args.manifest}")
    print(
        f"repeat_penalty={args.repeat_penalty} distance_target={distance_target} "
        f"beam_score={args.beam_score} distance_weight={args.distance_weight} "
        f"beam_length_normalization={args.beam_length_normalization}"
    )
    print("source_split_counts=" + json.dumps({k: len(v) for k, v in sorted(paths_by_split.items())}, indent=2))

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "checkpoint_args": {key: str(value) for key, value in checkpoint_args.items()},
        "splits": {},
    }
    for split, paths in sorted(paths_by_split.items()):
        result = evaluate_split(
            model,
            paths,
            height,
            width,
            device,
            args.max_steps,
            args.beam_width,
            args.beam_depth,
            args.top_k,
            args.max_cache_entries,
            f"level1_original_{split}",
            args.repeat_penalty,
            distance_target,
            distance_max_steps,
            args.beam_score,
            args.distance_weight,
            args.beam_length_normalization,
        )
        if not args.verbose:
            result["results"] = [
                {
                    "puzzle": Path(item["puzzle"]).name,
                    "solved": item["solved"],
                    "steps": item["steps"],
                    "repeated_states": item["repeated_states"],
                    "skipped": item.get("skipped", False),
                }
                for item in result["results"]
            ]
        summary["splits"][split] = result

    print("summary=" + json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
