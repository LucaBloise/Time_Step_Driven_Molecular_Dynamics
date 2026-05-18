#!/usr/bin/env python3
"""Run a short high-density simulation and render a zoomed overlap animation.

This script is focused on visually checking particle overlap/contact by drawing
particles with their physical radius as edge-only circles.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle

FRESH_COLOR = "#2ca02c"
USED_COLOR = "#d62728"


def parse_header_value(value: str) -> float:
    value = value.strip()
    token = value.split()[0] if value else value
    return float(token)


def parse_header(lines: list[str]) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in lines:
        entry = line[1:].strip()
        if "=" in entry:
            key, value = entry.split("=", 1)
            header[key.strip()] = value.strip()
    return header


def load_frames(path: Path, stride: int, max_frames: int | None) -> tuple[dict[str, str], list[float], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    header_lines: list[str] = []
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    times: list[float] = []

    with path.open("r", encoding="ascii") as handle:
        first_data_line = None
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header_lines.append(line)
                continue
            first_data_line = line
            break
        else:
            raise ValueError("No data rows found in state file.")

        header = parse_header(header_lines)
        n = int(header["N"])

        current_time = None
        x = np.zeros(n)
        y = np.zeros(n)
        state = np.zeros(n, dtype=int)
        filled = 0
        frame_index = 0

        def consume_row(row_line: str) -> bool:
            nonlocal current_time, filled, frame_index
            parts = row_line.split()
            t = float(parts[0])
            pid = int(parts[1])

            if current_time is None:
                current_time = t
            if t != current_time:
                if filled != n:
                    raise ValueError(f"Frame at t={current_time} has {filled} particles, expected {n}")

                if frame_index % stride == 0:
                    frames.append((x.copy(), y.copy(), state.copy()))
                    times.append(current_time)
                    if max_frames is not None and len(frames) >= max_frames:
                        return False

                frame_index += 1
                current_time = t
                filled = 0

            x[pid] = float(parts[2])
            y[pid] = float(parts[3])
            state[pid] = int(parts[6])
            filled += 1
            return True

        if not consume_row(first_data_line):
            return parse_header(header_lines), times, frames

        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not consume_row(line):
                break

        if max_frames is None or len(frames) < max_frames:
            if current_time is not None:
                if filled != n:
                    raise ValueError(f"Frame at t={current_time} has {filled} particles, expected {n}")
                if frame_index % stride == 0:
                    frames.append((x.copy(), y.copy(), state.copy()))
                    times.append(current_time)

    return parse_header(header_lines), times, frames


def compile_simulation(repo_root: Path, javac_cmd: str) -> None:
    simulation_dir = repo_root / "simulation"
    java_files = sorted(simulation_dir.glob("*.java"))
    if not java_files:
        raise FileNotFoundError(f"No Java files found in {simulation_dir}")

    cmd = [javac_cmd] + [str(path) for path in java_files]
    process = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            "Compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def run_simulation(repo_root: Path, args: argparse.Namespace, state_path: Path, props_path: Path) -> None:
    dt2 = args.dt if args.dt2 is None else args.dt2
    cmd = [
        args.java_cmd,
        "-cp",
        str(repo_root / "simulation"),
        "ScanningRateSimulation",
        "--n",
        str(args.n),
        "--l",
        str(args.l),
        "--r0",
        str(args.r0),
        "--r",
        str(args.r),
        "--m",
        str(args.m),
        "--k",
        str(args.k),
        "--v0",
        str(args.v0),
        "--tf",
        str(args.tf),
        "--dt",
        str(args.dt),
        "--dt2",
        str(dt2),
        "--seed",
        str(args.seed),
        "--out",
        str(state_path),
        "--properties-out",
        str(props_path),
        "--no-events",
    ]

    process = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            "Simulation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short simulation and export a zoomed overlap animation (edge-only circles)."
    )

    parser.add_argument("--n", type=int, default=900, help="Number of particles (high density recommended).")
    parser.add_argument("--tf", type=float, default=2.0, help="Final simulation time in seconds.")
    parser.add_argument("--dt", type=float, default=0.001, help="Integration timestep.")
    parser.add_argument("--dt2", type=float, default=None, help="Output timestep. Default: same as dt.")
    parser.add_argument("--k", type=float, default=1.0e3)
    parser.add_argument("--l", type=float, default=80.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--v0", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123456)

    parser.add_argument("--zoom-x", type=float, default=8.0, help="Zoom center x.")
    parser.add_argument("--zoom-y", type=float, default=0.0, help="Zoom center y.")
    parser.add_argument("--zoom-width", type=float, default=12.0, help="Zoom window width.")
    parser.add_argument("--zoom-height", type=float, default=12.0, help="Zoom window height.")

    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--java-cmd", type=str, default="java")
    parser.add_argument("--javac-cmd", type=str, default="javac")
    parser.add_argument("--compile", action="store_true", help="Compile Java sources before simulation.")

    parser.add_argument("--run-id", type=str, default="zoom_overlap", help="Output subfolder name.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs") / "zoom_overlap")
    parser.add_argument("--out", type=Path, default=None, help="Animation output path. Default inside run folder.")

    parser.add_argument("--reuse-existing-run", action="store_true", help="Reuse existing states.txt if present.")
    parser.add_argument("--only-plot", action="store_true", help="Do not run simulation; only animate existing states.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    run_dir = (repo_root / args.outputs_root / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    state_path = run_dir / "states.txt"
    props_path = run_dir / "properties.txt"

    if args.compile and not args.only_plot:
        print("Compiling simulation sources...")
        compile_simulation(repo_root, args.javac_cmd)

    if args.only_plot:
        if not state_path.exists():
            raise FileNotFoundError(f"Missing state file for only-plot mode: {state_path}")
    elif args.reuse_existing_run and state_path.exists():
        print(f"Reusing existing run: {state_path}")
    else:
        dt2 = args.dt if args.dt2 is None else args.dt2
        print(f"Running simulation (N={args.n}, tf={args.tf}, dt={args.dt}, dt2={dt2})...")
        run_simulation(repo_root, args, state_path, props_path)

    header, times, frames = load_frames(state_path, args.stride, args.max_frames)
    if not frames:
        raise ValueError("No animation frames loaded. Check stride/max-frames/dt2.")

    r = parse_header_value(header["r"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", "box")

    x_min = args.zoom_x - args.zoom_width / 2.0
    x_max = args.zoom_x + args.zoom_width / 2.0
    y_min = args.zoom_y - args.zoom_height / 2.0
    y_max = args.zoom_y + args.zoom_height / 2.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Posicion x (m)")
    ax.set_ylabel("Posicion y (m)")
    ax.grid(True, alpha=0.18)

    x0, y0, state0 = frames[0]
    particles: list[Circle] = []
    for i in range(x0.size):
        edge_color = FRESH_COLOR if state0[i] == 1 else USED_COLOR
        particle = Circle((x0[i], y0[i]), r, fill=False, edgecolor=edge_color, linewidth=1.0)
        ax.add_patch(particle)
        particles.append(particle)

    title_text = ax.set_title("", pad=12)

    def update(frame_idx: int):
        x, y, state = frames[frame_idx]
        for i, particle in enumerate(particles):
            particle.center = (x[i], y[i])
            particle.set_edgecolor(FRESH_COLOR if state[i] == 1 else USED_COLOR)
        title_text.set_text(f"t={times[frame_idx]:.4f} s | N={x.size} | dt={args.dt:g}")
        return particles + [title_text]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / args.fps,
        blit=False,
    )

    out_path = args.out if args.out is not None else (run_dir / "zoom_overlap.gif")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=args.fps)
        anim.save(str(out_path), writer=writer, dpi=200)
    else:
        try:
            writer = animation.FFMpegWriter(fps=args.fps)
            anim.save(str(out_path), writer=writer, dpi=200)
        except FileNotFoundError:
            fallback = out_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=args.fps)
            anim.save(str(fallback), writer=writer, dpi=200)
            out_path = fallback

    plt.close(fig)
    print(f"Animation saved to: {out_path}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
