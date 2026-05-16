from __future__ import annotations

import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from pushworld_study.paths import PROJECT_ROOT, ensure_upstream_pushworld_on_path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_planner_imitation import load_checkpoint
from planner_imitation_rollout import BEAM_SCORE_MODES, choose_action
from train_planner_imitation_v2 import (
    ACTION_CHARS,
)


ensure_upstream_pushworld_on_path()

from pushworld.puzzle import Actions, PushWorldPuzzle  # noqa: E402


DEFAULT_PLANNER = PROJECT_ROOT / "external/pushworld/cpp/build/bin/run_planner"
CHECKPOINT_OPTIONS = {
    "Multi4 Level-0": PROJECT_ROOT / "models/planner_imitation_level0_multi4.pt",
    "Base Level-0 2k": PROJECT_ROOT / "models/planner_imitation_level0_base_small.pt",
}
LEVEL0_TEST_DIRS = {
    "base": PROJECT_ROOT / "data/level0/base/test",
    "all": PROJECT_ROOT / "data/level0/all/test",
    "goals": PROJECT_ROOT / "data/level0/goals/test",
    "obstacles": PROJECT_ROOT / "data/level0/obstacles/test",
    "shapes": PROJECT_ROOT / "data/level0/shapes/test",
    "size": PROJECT_ROOT / "data/level0/size/test",
    "walls": PROJECT_ROOT / "data/level0/walls/test",
}
LEVEL0_BASE_TEST = LEVEL0_TEST_DIRS["base"]
LEVEL1 = PROJECT_ROOT / "external/pushworld/benchmark/puzzles/level1"
LEGEND = {
    ".": "empty cell",
    "W": "wall",
    "A": "agent cell",
    "AW": "agent-only wall / gate",
    "M0, M1, ...": "movable object cells; same id = same rigid object",
    "G0, G1, ...": "goal cells; G0 is target for M0, G1 for M1, etc.",
    "G0+M1": "multiple annotations in one cell; here goal G0 overlaps movable M1",
}


def numeric_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    suffix = stem.rsplit("_", 1)[-1]
    return (stem[: -len(suffix)], int(suffix)) if suffix.isdigit() else (stem, -1)


def list_puzzles(path: Path) -> list[Path]:
    return sorted(path.glob("*.pwp"), key=numeric_key)


def format_pwp_text(text: str) -> str:
    token_rows = []
    for raw_line in text.strip().splitlines():
        cells = raw_line.split()
        if not cells:
            continue
        token_rows.append(cells)
    if not token_rows:
        return ""
    cell_width = max(2, *(len(cell) for row in token_rows for cell in row))
    rows = ["  ".join(cell.ljust(cell_width) for cell in cells).rstrip() for cells in token_rows]
    return "\n".join(rows) + ("\n" if rows else "")


def set_editor_text(path: Path) -> None:
    st.session_state.selected_puzzle = path
    st.session_state.puzzle_text = format_pwp_text(path.read_text(encoding="utf-8"))
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.session_state.pop("last_rollout", None)
    st.session_state.pop("last_rgd", None)


def set_editor_text_value(text: str) -> None:
    st.session_state.puzzle_text = format_pwp_text(text)
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.session_state.pop("last_rollout", None)
    st.session_state.pop("last_rgd", None)


@st.cache_resource(show_spinner="Loading policy checkpoint...")
def load_policy(checkpoint_path: str, device_name: str):
    device = torch.device(device_name)
    model, height, width, checkpoint_args = load_checkpoint(Path(checkpoint_path), device)
    return model, height, width, device, checkpoint_args


def write_temp_puzzle(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".pwp",
        prefix="pushworld_demo_",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        handle.write(text.strip() + "\n")
    return Path(handle.name)


def load_puzzle_from_text(text: str) -> tuple[PushWorldPuzzle | None, Path | None, str | None]:
    try:
        path = write_temp_puzzle(text)
        return PushWorldPuzzle(str(path)), path, None
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def rollout_policy(
    model: torch.nn.Module,
    puzzle: PushWorldPuzzle,
    puzzle_key: str,
    height: int,
    width: int,
    device: torch.device,
    max_steps: int,
    beam_width: int,
    beam_depth: int,
    top_k: int,
    repeat_penalty: float,
    distance_target: str,
    distance_max_steps: int,
    beam_score: str,
    distance_weight: float,
    beam_length_normalization: float,
) -> dict[str, object]:
    state = puzzle.initial_state
    frames = [puzzle.render(state)]
    rows = []
    seen = {state}
    repeated_states = 0
    encode_cache: dict[tuple[str, tuple[tuple[int, int], ...]], torch.Tensor] = {}

    start = time.perf_counter()
    with torch.inference_mode():
        for step_idx in range(max_steps):
            if puzzle.is_goal_state(state):
                break
            achieved_before = puzzle.count_achieved_goals(state)
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
                puzzle_key=puzzle_key,
                encode_cache=encode_cache,
                max_cache_entries=50_000,
                seen_states=seen,
                repeat_penalty=repeat_penalty,
                distance_target=distance_target,
                distance_max_steps=distance_max_steps,
                beam_score=beam_score,
                distance_weight=distance_weight,
                beam_length_normalization=beam_length_normalization,
            )
            next_state = puzzle.get_next_state(state, action)
            no_op = next_state == state
            repeated = next_state in seen
            repeated_states += int(repeated)
            rows.append(
                {
                    "step": step_idx + 1,
                    "action": ACTION_CHARS[action],
                    "action_name": ["left", "right", "up", "down"][action],
                    "no_op": no_op,
                    "repeated_state": repeated,
                    "goals_before": achieved_before,
                    "goals_after": puzzle.count_achieved_goals(next_state),
                }
            )
            state = next_state
            seen.add(state)
            frames.append(puzzle.render(state))
    elapsed = time.perf_counter() - start
    return {
        "solved": puzzle.is_goal_state(state),
        "steps": len(rows),
        "actions": "".join(row["action"] for row in rows),
        "rows": rows,
        "frames": frames,
        "repeated_states": repeated_states,
        "repeat_penalty": repeat_penalty,
        "distance_target": distance_target,
        "beam_score": beam_score,
        "distance_weight": distance_weight,
        "beam_length_normalization": beam_length_normalization,
        "time_s": elapsed,
    }


def rollout_plan(puzzle: PushWorldPuzzle, plan: str) -> dict[str, object]:
    state = puzzle.initial_state
    frames = [puzzle.render(state)]
    rows = []
    seen = {state}
    repeated_states = 0
    for step_idx, action_char in enumerate(plan):
        action = Actions.FROM_CHAR[action_char]
        achieved_before = puzzle.count_achieved_goals(state)
        next_state = puzzle.get_next_state(state, action)
        no_op = next_state == state
        repeated = next_state in seen
        repeated_states += int(repeated)
        rows.append(
            {
                "step": step_idx + 1,
                "action": action_char,
                "action_name": ["left", "right", "up", "down"][action],
                "no_op": no_op,
                "repeated_state": repeated,
                "goals_before": achieved_before,
                "goals_after": puzzle.count_achieved_goals(next_state),
            }
        )
        state = next_state
        seen.add(state)
        frames.append(puzzle.render(state))
    return {
        "solved": puzzle.is_goal_state(state),
        "steps": len(rows),
        "actions": plan,
        "rows": rows,
        "frames": frames,
        "repeated_states": repeated_states,
    }


def run_rgd(planner: Path, puzzle: PushWorldPuzzle, puzzle_path: Path, time_limit: float) -> dict[str, object]:
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [str(planner), "N+RGD", str(puzzle_path)],
            capture_output=True,
            text=True,
            timeout=time_limit + 1.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "solved": False,
            "plan": "",
            "time_s": time.perf_counter() - start,
            "status": "timeout",
            "frames": [],
            "rows": [],
            "repeated_states": 0,
        }

    elapsed = time.perf_counter() - start
    plan = result.stdout.strip()
    solved = result.returncode == 0 and bool(plan) and set(plan).issubset(set(ACTION_CHARS))
    rollout = rollout_plan(puzzle, plan) if solved else {"frames": [], "rows": [], "repeated_states": 0}
    return {
        "solved": solved,
        "plan": plan if solved else "",
        "time_s": elapsed,
        "status": "solved" if solved else result.stdout.strip() or result.stderr.strip() or "failed",
        **rollout,
    }


def render_rollout_view(result: dict[str, object], fps: float, key_prefix: str) -> None:
    frames = result["frames"]
    if not frames:
        return
    playing_key = f"{key_prefix}_playing"
    frame_key = f"{key_prefix}_frame"
    st.session_state.setdefault(playing_key, False)
    st.session_state.setdefault(frame_key, 0)

    frame_placeholder = st.empty()
    max_idx = len(frames) - 1
    selected = st.slider(
        "Step",
        min_value=0,
        max_value=max_idx,
        value=min(int(st.session_state[frame_key]), max_idx),
        key=f"{key_prefix}_step_slider",
    )
    st.session_state[frame_key] = selected
    frame_placeholder.image(frames[selected], caption=f"Step {selected}", use_container_width=False)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Play rollout", key=f"{key_prefix}_play"):
            st.session_state[playing_key] = True
            st.rerun()
    with col_b:
        if st.button("Stop", key=f"{key_prefix}_stop"):
            st.session_state[playing_key] = False
            st.rerun()
    with col_c:
        st.metric("Playback FPS", f"{fps:.1f}")

    if st.session_state[playing_key]:
        delay = 1.0 / max(fps, 0.1)
        start_idx = min(int(st.session_state[frame_key]), max_idx)
        for idx in range(start_idx, len(frames)):
            if not st.session_state.get(playing_key, False):
                break
            st.session_state[frame_key] = idx
            frame_placeholder.image(frames[idx], caption=f"Step {idx}", use_container_width=False)
            time.sleep(delay)
        st.session_state[playing_key] = False


def main() -> None:
    st.set_page_config(page_title="PushWorld Policy Demo", layout="wide")
    st.markdown(
        """
        <style>
        textarea {
            font-family: "JetBrains Mono", "Fira Code", "Cascadia Mono",
                         "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
            font-size: 14px !important;
            line-height: 1.35 !important;
            white-space: pre !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("PushWorld Planner-Imitation Policy Demo")

    if "selected_puzzle" not in st.session_state:
        st.session_state.selected_puzzle = None
    if "puzzle_text" not in st.session_state:
        default = list_puzzles(LEVEL0_BASE_TEST)[0]
        set_editor_text(default)

    with st.sidebar:
        st.header("Model")
        checkpoint_name = st.selectbox("Checkpoint", list(CHECKPOINT_OPTIONS), index=0)
        checkpoint = CHECKPOINT_OPTIONS[checkpoint_name]
        st.caption(str(checkpoint.relative_to(PROJECT_ROOT) if checkpoint.is_relative_to(PROJECT_ROOT) else checkpoint))
        device_default = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = st.selectbox("Device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"], index=0 if device_default == "cuda" else 0)

        st.header("Puzzle")
        split = st.selectbox("Split", ["Level 0 test", "Level 1"])
        if split == "Level 0 test":
            level0_variant = st.selectbox("Level 0 set", list(LEVEL0_TEST_DIRS), index=0)
            puzzle_dir = LEVEL0_TEST_DIRS[level0_variant]
        else:
            puzzle_dir = LEVEL1
        puzzles = list_puzzles(puzzle_dir)
        names = [path.name for path in puzzles]

        if st.button("Random puzzle"):
            path = random.choice(puzzles)
            set_editor_text(path)
            st.rerun()

        selected_name = st.selectbox("Puzzle", names, index=names.index(st.session_state.selected_puzzle.name) if st.session_state.selected_puzzle in puzzles else 0)
        selected_path = puzzle_dir / selected_name
        if selected_path != st.session_state.selected_puzzle:
            set_editor_text(selected_path)
            st.rerun()

        st.header("Rollout")
        max_steps = st.number_input("Max steps", min_value=1, max_value=500, value=100, step=10)
        beam_width = st.number_input("Beam width", min_value=1, max_value=64, value=8, step=1)
        beam_depth = st.number_input("Beam depth", min_value=1, max_value=32, value=8, step=1)
        top_k = st.number_input("Top-k actions", min_value=1, max_value=4, value=3, step=1)
        repeat_penalty = st.number_input(
            "Repeat penalty",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.5,
        )
        beam_score = st.selectbox("Beam score", BEAM_SCORE_MODES, index=BEAM_SCORE_MODES.index("policy_distance"))
        distance_weight = st.number_input(
            "Distance weight",
            min_value=0.0,
            max_value=10.0,
            value=0.15,
            step=0.05,
        )
        beam_length_normalization = st.number_input(
            "Length normalization",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
        )
        fps = st.slider("Playback FPS", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

        st.header("Planner")
        compare_rgd = st.checkbox("Compare with RGD", value=True)
        planner_timeout = st.number_input("RGD timeout, sec", min_value=0.1, max_value=30.0, value=3.0, step=0.5)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Editable `.pwp`")
        with st.expander("Cell legend", expanded=True):
            st.table(pd.DataFrame([{"token": key, "meaning": value} for key, value in LEGEND.items()]))
            st.caption("Cells are whitespace-separated. Keep object and goal ids consistent when editing.")
        puzzle_text = st.text_area(
            "Puzzle text",
            value=st.session_state.puzzle_text,
            height=260,
            key=f"puzzle_text_area_{st.session_state.get('editor_version', 0)}",
        )
        col_fmt, col_load = st.columns(2)
        with col_fmt:
            if st.button("Align cells"):
                set_editor_text_value(puzzle_text)
                st.rerun()
        with col_load:
            if st.button("Load edited puzzle"):
                set_editor_text_value(puzzle_text)
                st.rerun()

    puzzle, temp_path, error = load_puzzle_from_text(st.session_state.puzzle_text)
    if error is not None or puzzle is None or temp_path is None:
        st.error(f"Could not parse puzzle: {error}")
        return

    with right:
        st.subheader("Initial State")
        st.image(puzzle.render(puzzle.initial_state), use_container_width=False)
        st.caption(f"Dimensions: {puzzle.dimensions[0]}x{puzzle.dimensions[1]}, goals: {len(puzzle.goal_state)}")

    run_clicked = st.button("Run model rollout", type="primary")
    if run_clicked:
        model, height, width, device, checkpoint_args = load_policy(str(checkpoint), device_name)
        distance_target = str(checkpoint_args.get("distance_target", "linear"))
        distance_max_steps = int(checkpoint_args.get("max_steps", max_steps))
        if puzzle.dimensions[1] > height or puzzle.dimensions[0] > width:
            st.error(
                f"Puzzle is larger than checkpoint padding: puzzle={puzzle.dimensions}, "
                f"checkpoint_width_height=({width}, {height})"
            )
            return

        with st.spinner("Running policy rollout..."):
            result = rollout_policy(
                model,
                puzzle,
                str(temp_path),
                height,
                width,
                device,
                int(max_steps),
                int(beam_width),
                int(beam_depth),
                int(top_k),
                float(repeat_penalty),
                distance_target,
                distance_max_steps,
                str(beam_score),
                float(distance_weight),
                float(beam_length_normalization),
            )
        result["checkpoint_args"] = checkpoint_args
        st.session_state.last_rollout = result
        st.session_state.last_temp_path = temp_path

        if compare_rgd:
            with st.spinner("Running RGD comparison..."):
                st.session_state.last_rgd = run_rgd(DEFAULT_PLANNER, puzzle, temp_path, float(planner_timeout))
        else:
            st.session_state.last_rgd = None

    if st.session_state.get("last_rollout") is not None:
        result = st.session_state.last_rollout
        st.subheader("Model Rollout")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Solved", "yes" if result["solved"] else "no")
        col2.metric("Steps", result["steps"])
        col3.metric("Repeated states", result["repeated_states"])
        col4.metric("Repeat penalty", f"{float(result.get('repeat_penalty', 0.0)):.1f}")
        col5.metric("Inference time", f"{result['time_s']:.2f}s")

        st.code(result["actions"] or "(no actions)", language="text")
        render_rollout_view(result, float(fps), "model")
        st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True)

    if st.session_state.get("last_rgd") is not None:
        rgd = st.session_state.last_rgd
        st.subheader("RGD Comparison")
        model_result = st.session_state.get("last_rollout")
        both_solved = bool(model_result and model_result.get("solved") and rgd.get("solved"))
        length_delta = int(model_result["steps"]) - len(rgd["plan"]) if both_solved else None
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("RGD status", rgd["status"])
        col2.metric("Plan length", len(rgd["plan"]))
        col3.metric("Repeated states", rgd.get("repeated_states", 0))
        col4.metric("Model - RGD", "n/a" if length_delta is None else f"{length_delta:+d}")
        col5.metric("Solve time", f"{rgd['time_s']:.3f}s")
        if length_delta is not None:
            if length_delta < 0:
                st.success(f"Model found a shorter solved rollout by {-length_delta} steps.")
            elif length_delta > 0:
                st.info(f"RGD plan is shorter by {length_delta} steps.")
            else:
                st.info("Model and RGD used the same number of steps.")
        st.code(rgd["plan"] or "(no plan)", language="text")
        if rgd.get("frames"):
            render_rollout_view(rgd, float(fps), "rgd")
            st.dataframe(pd.DataFrame(rgd["rows"]), use_container_width=True)


if __name__ == "__main__":
    main()
