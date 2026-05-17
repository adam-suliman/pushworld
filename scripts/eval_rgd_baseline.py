from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for bare Python envs
    def tqdm(iterable, **_: Any):
        return iterable

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pushworld_study.paths import PROJECT_ROOT, ensure_upstream_pushworld_on_path  # noqa: E402


ensure_upstream_pushworld_on_path()

from pushworld.puzzle import Actions, PushWorldPuzzle  # noqa: E402


ACTION_CHARS = set(Actions.FROM_CHAR)


def default_planner_path() -> Path:
    base = PROJECT_ROOT / "external/pushworld/cpp/build/bin/run_planner"
    exe = base.with_suffix(".exe")
    return exe if exe.exists() else base


def select_puzzles(eval_dirs: list[Path], eval_puzzles: int, all_eval: bool) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for directory in eval_dirs:
        for path in sorted(directory.glob("*.pwp"), key=lambda item: item.name.casefold()):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    if all_eval:
        return paths
    return paths[:eval_puzzles]


def run_planner(
    planner: Path,
    planner_mode: str,
    puzzle_path: Path,
    time_limit_s: float,
    include_plan: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    timed_out = False
    stdout = ""
    stderr = ""
    returncode: int | None = None
    try:
        result = subprocess.run(
            [str(planner), planner_mode, str(puzzle_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=time_limit_s,
        )
        returncode = result.returncode
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
    elapsed = time.perf_counter() - start

    plan = stdout.strip().upper()
    valid_chars = bool(plan) and set(plan).issubset(ACTION_CHARS)
    solved = False
    valid_plan = False
    validation_error: str | None = None
    if not timed_out and returncode == 0 and valid_chars:
        try:
            puzzle = PushWorldPuzzle(str(puzzle_path))
            actions = [Actions.FROM_CHAR[ch] for ch in plan]
            valid_plan = puzzle.is_valid_plan(actions)
            solved = valid_plan
        except Exception as exc:  # pragma: no cover - diagnostic path
            validation_error = repr(exc)

    row: dict[str, Any] = {
        "puzzle": str(puzzle_path),
        "solved": solved,
        "valid_plan": valid_plan,
        "timed_out": timed_out,
        "returncode": returncode,
        "time_s": elapsed,
        "plan_length": len(plan) if valid_chars else 0,
    }
    if include_plan:
        row["plan"] = plan if valid_chars else stdout
    if stderr:
        row["stderr"] = stderr[-1000:]
    if validation_error is not None:
        row["validation_error"] = validation_error
    if not valid_chars and stdout:
        row["stdout"] = stdout[-1000:]
    return row


def summarize_repeat(repeat_idx: int, results: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    solved_results = [row for row in results if row["solved"]]
    solved_times = [float(row["time_s"]) for row in solved_results]
    plan_lengths = [int(row["plan_length"]) for row in solved_results]
    solved = len(solved_results)
    total = len(results)
    return {
        "repeat": repeat_idx,
        "solved": solved,
        "total": total,
        "success_rate": solved / max(1, total),
        "time_s": elapsed,
        "solves_per_minute": solved * 60.0 / max(elapsed, 1e-9),
        "timeouts": sum(int(row["timed_out"]) for row in results),
        "invalid_or_failed": sum(int(not row["solved"] and not row["timed_out"]) for row in results),
        "mean_solved_time_s": statistics.mean(solved_times) if solved_times else None,
        "median_solved_time_s": statistics.median(solved_times) if solved_times else None,
        "mean_plan_length": statistics.mean(plan_lengths) if plan_lengths else None,
        "max_plan_length": max(plan_lengths) if plan_lengths else None,
        "results": results,
    }


def aggregate_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "solved",
        "time_s",
        "solves_per_minute",
        "timeouts",
        "invalid_or_failed",
        "mean_solved_time_s",
        "median_solved_time_s",
        "mean_plan_length",
        "max_plan_length",
    ]
    aggregate: dict[str, Any] = {"repeats": len(repeats)}
    for key in numeric_keys:
        values = [repeat[key] for repeat in repeats if repeat.get(key) is not None]
        if not values:
            aggregate[f"{key}_mean"] = None
            aggregate[f"{key}_min"] = None
            aggregate[f"{key}_max"] = None
            continue
        aggregate[f"{key}_mean"] = statistics.mean(values)
        aggregate[f"{key}_min"] = min(values)
        aggregate[f"{key}_max"] = max(values)
    return aggregate


def runtime_metadata() -> dict[str, Any]:
    return {
        "planner_process_device": "cpu",
        "neural_model_device": None,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", type=Path, default=default_planner_path())
    parser.add_argument("--planner-mode", default="N+RGD")
    parser.add_argument("--eval-dir", type=Path, action="append", required=True)
    parser.add_argument("--eval-puzzles", type=int, default=100)
    parser.add_argument("--all-eval", action="store_true")
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--split-name", default="rgd_eval")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-plans", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.time_limit <= 0:
        raise ValueError("--time-limit must be > 0")

    puzzle_paths = select_puzzles(args.eval_dir, args.eval_puzzles, args.all_eval)
    if not puzzle_paths:
        raise ValueError("No puzzles selected")

    repeat_summaries: list[dict[str, Any]] = []
    for repeat_idx in range(1, args.repeats + 1):
        start = time.perf_counter()
        results = []
        progress = tqdm(puzzle_paths, desc=f"rgd {args.split_name} repeat {repeat_idx}", unit="puzzle")
        solved = 0
        for puzzle_path in progress:
            row = run_planner(
                args.planner,
                args.planner_mode,
                puzzle_path,
                args.time_limit,
                args.include_plans,
            )
            solved += int(row["solved"])
            results.append(row)
            progress.set_postfix(solved=f"{solved}/{len(results)}")
        repeat_summaries.append(summarize_repeat(repeat_idx, results, time.perf_counter() - start))

    payload = {
        "split": args.split_name,
        "planner": str(args.planner),
        "planner_mode": args.planner_mode,
        "eval_dirs": [str(path) for path in args.eval_dir],
        "eval_puzzles": len(puzzle_paths),
        "time_limit_s": args.time_limit,
        "repeats": args.repeats,
        "repeat_results": repeat_summaries,
        "aggregate": aggregate_repeats(repeat_summaries),
        "runtime": runtime_metadata(),
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
