from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from planner_imitation_rollout import (
    SEARCH_MODES,
    RolloutProfile,
    best_first_search,
    choose_action,
)
from planner_imitation_rollout import BEAM_SCORE_MODES, DISTANCE_TARGETS
from pushworld_study.paths import PROJECT_ROOT, ensure_upstream_pushworld_on_path
from train_planner_imitation_v2 import (
    ACTION_CHARS,
    BoardTransformerPolicy,
    select_puzzles,
)


ensure_upstream_pushworld_on_path()

from pushworld.puzzle import PushWorldPuzzle  # noqa: E402


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    args = checkpoint.get("args", {})
    height = int(checkpoint["height"])
    width = int(checkpoint["width"])
    encoder_stem = str(args.get("encoder_stem") or ("conv" if "conv_stem.0.weight" in state_dict else "linear"))
    if encoder_stem == "conv":
        d_model = int(args.get("d_model", state_dict["conv_stem.0.weight"].shape[0]))
    else:
        d_model = int(args.get("d_model", state_dict["token_proj.weight"].shape[0]))
    nhead = int(args.get("nhead", 4))
    layers = int(args.get("layers", 1))
    distance_bins = int(state_dict["distance_head.weight"].shape[0])
    dropout = float(args.get("dropout", 0.0))

    model = BoardTransformerPolicy(
        channels=int(checkpoint.get("channels", 7)),
        height=height,
        width=width,
        d_model=d_model,
        nhead=nhead,
        layers=layers,
        distance_bins=distance_bins,
        encoder_stem=encoder_stem,
        dropout=dropout,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, height, width, args


def evaluate_split(
    model: BoardTransformerPolicy,
    puzzle_paths: list[Path],
    height: int,
    width: int,
    device: torch.device,
    max_steps: int,
    beam_width: int,
    beam_depth: int,
    top_k: int,
    max_cache_entries: int,
    split_name: str,
    repeat_penalty: float = 0.0,
    distance_target: str = "linear",
    distance_max_steps: int | None = None,
    beam_score: str = "policy_distance",
    distance_weight: float = 0.15,
    beam_length_normalization: float = 0.0,
    closed_list_pruning: bool = False,
    search_mode: str = "beam",
    best_first_budget: int = 512,
    best_first_batch_size: int = 32,
    best_first_top_k: int | None = None,
    best_first_max_depth: int | None = None,
    best_first_step_penalty: float = 0.0,
) -> dict[str, object]:
    solved = 0
    results = []
    encode_cache: dict[tuple[str, tuple[tuple[int, int], ...]], torch.Tensor] = {}
    prediction_cache: dict[tuple[str, tuple[tuple[int, int], ...]], tuple[torch.Tensor, torch.Tensor]] = {}
    profile = RolloutProfile()
    fallback_count = 0
    best_first_solved = 0
    start = time.perf_counter()
    with torch.inference_mode():
        progress = tqdm(puzzle_paths, desc=f"eval {split_name}", unit="puzzle")
        for path in progress:
            parse_start = time.perf_counter()
            puzzle = PushWorldPuzzle(str(path))
            profile.puzzle_parse_time_s += time.perf_counter() - parse_start
            if puzzle.dimensions[1] > height or puzzle.dimensions[0] > width:
                result = {
                    "puzzle": str(path),
                    "solved": False,
                    "steps": 0,
                    "actions": "",
                    "repeated_states": 0,
                    "skipped": True,
                    "reason": (
                        f"puzzle dimensions {puzzle.dimensions} exceed checkpoint "
                        f"padding width={width}, height={height}"
                    ),
                }
                results.append(result)
                progress.set_postfix(solved=f"{solved}/{len(results)}")
                continue

            state = puzzle.initial_state
            actions = []
            repeated_states = 0
            seen = {state}
            search_source = search_mode

            if search_mode in ("best_first", "best_first_fallback"):
                best_first = best_first_search(
                    model=model,
                    puzzle=puzzle,
                    state=state,
                    height=height,
                    width=width,
                    device=device,
                    puzzle_key=str(path),
                    encode_cache=encode_cache,
                    max_cache_entries=max_cache_entries,
                    node_budget=best_first_budget,
                    batch_size=best_first_batch_size,
                    top_k=best_first_top_k or top_k,
                    max_depth=best_first_max_depth or max_steps,
                    distance_target=distance_target,
                    distance_max_steps=distance_max_steps,
                    distance_weight=distance_weight,
                    step_penalty=best_first_step_penalty,
                    prediction_cache=prediction_cache,
                    profile=profile,
                )
                if best_first.solved:
                    search_source = "best_first"
                    best_first_solved += 1
                    for action in best_first.path[:max_steps]:
                        if puzzle.is_goal_state(state):
                            break
                        actions.append(ACTION_CHARS[action])
                        step_start = time.perf_counter()
                        state = puzzle.get_next_state(state, action)
                        profile.env_step_time_s += time.perf_counter() - step_start
                        repeated_states += int(state in seen)
                        seen.add(state)
                elif search_mode == "best_first_fallback":
                    search_source = "beam_fallback"
                    fallback_count += 1
                else:
                    search_source = "best_first_failed"

            if search_mode == "beam" or search_source == "beam_fallback":
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
                    repeated_states += int(state in seen)
                    seen.add(state)

            did_solve = puzzle.is_goal_state(state)
            solved += int(did_solve)
            results.append(
                {
                    "puzzle": str(path),
                    "solved": did_solve,
                    "steps": len(actions),
                    "actions": "".join(actions),
                    "repeated_states": repeated_states,
                    "skipped": False,
                    "search_source": search_source,
                }
            )
            progress.set_postfix(solved=f"{solved}/{len(results)}")

    total = len(puzzle_paths)
    skipped = sum(int(result.get("skipped", False)) for result in results)
    elapsed = time.perf_counter() - start
    profile.eval_loop_time_s = elapsed
    return {
        "split": split_name,
        "solved": solved,
        "total": total,
        "skipped": skipped,
        "success_rate": solved / max(1, total - skipped),
        "time_s": elapsed,
        "solves_per_minute": solved * 60.0 / max(elapsed, 1e-9),
        "max_steps": max_steps,
        "beam_width": beam_width,
        "beam_depth": beam_depth,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "distance_target": distance_target,
        "beam_score": beam_score,
        "distance_weight": distance_weight,
        "beam_length_normalization": beam_length_normalization,
        "closed_list_pruning": closed_list_pruning,
        "search_mode": search_mode,
        "best_first_budget": best_first_budget,
        "best_first_batch_size": best_first_batch_size,
        "best_first_top_k": best_first_top_k or top_k,
        "best_first_max_depth": best_first_max_depth or max_steps,
        "best_first_step_penalty": best_first_step_penalty,
        "best_first_solved": best_first_solved,
        "fallback_count": fallback_count,
        "cache_entries": len(encode_cache),
        "prediction_cache_entries": len(prediction_cache),
        "profile": profile.to_dict(),
        "results": results,
    }


def make_writer(log_dir: Path | None):
    if log_dir is None:
        return None
    from torch.utils.tensorboard import SummaryWriter

    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, action="append", required=True)
    parser.add_argument("--split-name", default="eval")
    parser.add_argument("--eval-puzzles", type=int, default=100)
    parser.add_argument("--all-eval", action="store_true")
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
    parser.add_argument(
        "--closed-list-pruning",
        action="store_true",
        help="Drop beam candidates that revisit states already seen in the current rollout.",
    )
    parser.add_argument("--search-mode", choices=SEARCH_MODES, default="beam")
    parser.add_argument("--best-first-budget", type=int, default=512)
    parser.add_argument("--best-first-batch-size", type=int, default=32)
    parser.add_argument("--best-first-top-k", type=int, default=0)
    parser.add_argument("--best-first-max-depth", type=int, default=0)
    parser.add_argument("--best-first-step-penalty", type=float, default=0.0)
    parser.add_argument("--max-cache-entries", type=int, default=250_000)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.repeat_penalty < 0.0:
        raise ValueError("--repeat-penalty must be >= 0")
    if args.distance_weight < 0.0:
        raise ValueError("--distance-weight must be >= 0")
    if args.beam_length_normalization < 0.0:
        raise ValueError("--beam-length-normalization must be >= 0")
    if args.best_first_budget < 0:
        raise ValueError("--best-first-budget must be >= 0")
    if args.best_first_batch_size < 1:
        raise ValueError("--best-first-batch-size must be >= 1")
    if args.best_first_top_k < 0:
        raise ValueError("--best-first-top-k must be >= 0")
    if args.best_first_max_depth < 0:
        raise ValueError("--best-first-max-depth must be >= 0")
    if args.best_first_step_penalty < 0.0:
        raise ValueError("--best-first-step-penalty must be >= 0")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    puzzle_paths = select_puzzles(args.eval_dir, args.eval_puzzles, args.all_eval)
    model, height, width, checkpoint_args = load_checkpoint(args.checkpoint, device)
    checkpoint_distance_target = str(checkpoint_args.get("distance_target", "linear"))
    distance_target = checkpoint_distance_target if args.distance_target == "checkpoint" else args.distance_target
    distance_max_steps = int(checkpoint_args.get("max_steps", args.max_steps))

    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_board=height:{height}, width:{width}")
    print("eval_dirs=" + json.dumps([str(path) for path in args.eval_dir], indent=2))
    print(
        f"eval_puzzles={len(puzzle_paths)} max_steps={args.max_steps} "
        f"repeat_penalty={args.repeat_penalty} distance_target={distance_target} "
        f"beam_score={args.beam_score} distance_weight={args.distance_weight} "
        f"beam_length_normalization={args.beam_length_normalization} "
        f"search_mode={args.search_mode} best_first_budget={args.best_first_budget}"
    )

    result = evaluate_split(
        model,
        puzzle_paths,
        height,
        width,
        device,
        args.max_steps,
        args.beam_width,
        args.beam_depth,
        args.top_k,
        args.max_cache_entries,
        args.split_name,
        args.repeat_penalty,
        distance_target,
        distance_max_steps,
        args.beam_score,
        args.distance_weight,
        args.beam_length_normalization,
        args.closed_list_pruning,
        args.search_mode,
        args.best_first_budget,
        args.best_first_batch_size,
        args.best_first_top_k or None,
        args.best_first_max_depth or None,
        args.best_first_step_penalty,
    )

    summary = dict(result)
    if not args.verbose:
        summary["results"] = [
            {
                "puzzle": item["puzzle"],
                "solved": item["solved"],
                "steps": item["steps"],
                "repeated_states": item["repeated_states"],
                "skipped": item.get("skipped", False),
                "search_source": item.get("search_source"),
            }
            for item in result["results"]
        ]
    summary["checkpoint_args"] = {key: str(value) for key, value in checkpoint_args.items()}

    print("summary=" + json.dumps(summary, indent=2))

    writer = make_writer(args.tensorboard_log)
    if writer is not None:
        writer.add_scalar(f"{args.split_name}/solved", result["solved"], 0)
        writer.add_scalar(f"{args.split_name}/total", result["total"], 0)
        writer.add_scalar(f"{args.split_name}/success_rate", result["success_rate"], 0)
        writer.add_scalar(f"{args.split_name}/time_s", result["time_s"], 0)
        writer.add_scalar(f"{args.split_name}/solves_per_minute", result["solves_per_minute"], 0)
        writer.add_text(f"{args.split_name}/config", json.dumps(summary, indent=2))
        writer.flush()
        writer.close()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
