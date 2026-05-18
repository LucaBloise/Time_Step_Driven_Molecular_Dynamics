#!/usr/bin/env python3
"""Run a dt sweep for ScanningRateSimulation and plot energy vs time for each dt."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def parse_list_float(raw: str) -> List[float]:
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0:
            raise ValueError("All dt values must be > 0")
        values.append(value)

    if not values:
        raise ValueError("At least one dt value is required")

    return values


def parse_header_value(value: str) -> float:
    value = value.strip()
    token = value.split()[0] if value else value
    return float(token)


def parse_header(lines: Sequence[str]) -> Dict[str, str]:
    header: Dict[str, str] = {}
    for line in lines:
        entry = line[1:].strip()
        if "=" in entry:
            key, val = entry.split("=", 1)
            header[key.strip()] = val.strip()
    return header


def compute_frame_energy(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    m: float,
    k: float,
    r: float,
    r0: float,
    outer_radius: float,
) -> float:
    kinetic = 0.5 * m * np.sum(vx * vx + vy * vy)

    potential = 0.0
    n = x.size
    min_dist = 2.0 * r

    for i in range(n):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n):
            dx = xi - x[j]
            dy = yi - y[j]
            dist2 = dx * dx + dy * dy
            if dist2 < min_dist * min_dist:
                dist = math.sqrt(dist2)
                overlap = min_dist - dist
                potential += 0.5 * k * overlap * overlap

    for i in range(n):
        dist = math.sqrt(x[i] * x[i] + y[i] * y[i])
        overlap_obstacle = r0 + r - dist
        if overlap_obstacle > 0.0:
            potential += 0.5 * k * overlap_obstacle * overlap_obstacle
        overlap_wall = r + dist - outer_radius
        if overlap_wall > 0.0:
            potential += 0.5 * k * overlap_wall * overlap_wall

    return float(kinetic + potential)


def load_frames(path: Path) -> Tuple[Dict[str, str], List[str]]:
    header_lines: List[str] = []
    rows: List[str] = []

    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header_lines.append(line)
            else:
                rows.append(line)

    return parse_header(header_lines), rows


def compute_energy_series(header: Dict[str, str], rows: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    n = int(header["N"])
    m = parse_header_value(header["m"])
    k = parse_header_value(header["k"])
    r = parse_header_value(header["r"])
    r0 = parse_header_value(header["r0"])
    l = parse_header_value(header["L"])
    outer_radius = l / 2.0

    times: List[float] = []
    energies: List[float] = []

    x = np.zeros(n)
    y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)

    current_time = None
    filled = 0

    for row in rows:
        parts = row.split()
        t = float(parts[0])
        pid = int(parts[1])

        if current_time is None:
            current_time = t

        if t != current_time:
            if filled != n:
                raise ValueError(f"Frame at t={current_time} has {filled} particles, expected {n}")
            energies.append(compute_frame_energy(x, y, vx, vy, m, k, r, r0, outer_radius))
            times.append(current_time)
            current_time = t
            filled = 0

        x[pid] = float(parts[2])
        y[pid] = float(parts[3])
        vx[pid] = float(parts[4])
        vy[pid] = float(parts[5])
        filled += 1

    if current_time is not None:
        if filled != n:
            raise ValueError(f"Frame at t={current_time} has {filled} particles, expected {n}")
        energies.append(compute_frame_energy(x, y, vx, vy, m, k, r, r0, outer_radius))
        times.append(current_time)

    return np.array(times), np.array(energies)


def build_plain_formatter(values: np.ndarray) -> FuncFormatter:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return FuncFormatter(lambda _v, _p: "0")

    span = float(np.max(finite) - np.min(finite))
    if span == 0.0:
        decimals = 6
    else:
        decimals = max(0, int(-math.floor(math.log10(span))) + 1)
        decimals = min(decimals, 6)

    def formatter(value: float, _pos: int) -> str:
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

    return FuncFormatter(formatter)


def compile_simulation(repo_root: Path, javac_cmd: str) -> None:
    simulation_dir = repo_root / "simulation"
    java_files = sorted(simulation_dir.glob("*.java"))
    if not java_files:
        raise FileNotFoundError(f"No Java files found in {simulation_dir}")

    cmd = [javac_cmd] + [str(path) for path in java_files]
    print("Compiling Java sources...")
    process = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            "Compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def format_dt_tag(dt: float) -> str:
    return f"{dt:.10g}".replace(".", "p").replace("-", "m")


def run_simulation_for_dt(
    repo_root: Path,
    java_cmd: str,
    dt: float,
    args: argparse.Namespace,
    state_path: Path,
    props_path: Path,
) -> None:
    cmd = [
        java_cmd,
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
        str(dt),
        "--dt2",
        str(args.dt2 if args.dt2 is not None else dt),
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
            f"dt={dt}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep dt and plot energy vs time for each dt (System 2)."
    )

    parser.add_argument("--dt-values", type=str, default="0.01,0.005,0.001,0.0005")
    parser.add_argument("--tf", type=float, default=200.0)
    parser.add_argument("--dt2", type=float, default=0.1, help="Output step for states. Larger means fewer saved frames (default: 0.1 s)")

    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--l", type=float, default=80.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=1.0e3)
    parser.add_argument("--v0", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123456)

    parser.add_argument("--java-cmd", type=str, default="java")
    parser.add_argument("--javac-cmd", type=str, default="javac")
    parser.add_argument("--compile", action="store_true", help="Compile simulation classes before running")

    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs") / "dt_sweep_energy",
        help="Base directory for generated states/properties",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs") / "dt_sweep_energy" / "energy_vs_time_by_dt.png")

    parser.add_argument("--relative", action="store_true", help="Plot (E - E0) / E0")
    parser.add_argument("--y-pad", type=float, default=0.15)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Run simulations even if existing states are already present.",
    )
    parser.add_argument("--only-plot", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    dt_values = parse_list_float(args.dt_values)

    if args.compile and not args.only_plot:
        compile_simulation(repo_root, args.javac_cmd)

    outputs_root = repo_root / args.outputs_root
    outputs_root.mkdir(parents=True, exist_ok=True)

    series: List[Tuple[float, np.ndarray, np.ndarray]] = []

    for dt in dt_values:
        run_dir = outputs_root / f"dt_{format_dt_tag(dt)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        state_path = run_dir / "states.txt"
        props_path = run_dir / "properties.txt"

        dt_out = args.dt2
        steps = int(math.ceil(args.tf / dt))
        frames = int(math.ceil(args.tf / dt_out)) + 1
        points = frames * args.n

        print(
            f"dt={dt:g}: steps={steps}, output_dt={dt_out:g}, approx_frames={frames}, approx_points={points}",
            flush=True,
        )
        if points > 2_000_000:
            print(
                "Warning: large output size. Consider increasing --dt2 or lowering --tf/--n.",
                flush=True,
            )

        if args.only_plot:
            if not state_path.exists():
                raise FileNotFoundError(f"Missing states file for dt={dt}: {state_path}")
        elif state_path.exists() and not args.force_rerun:
            print(f"Reusing existing run for dt={dt:g}: {state_path}", flush=True)
        else:
            print(f"Running simulation for dt={dt} ...")
            t0 = time.perf_counter()
            run_simulation_for_dt(repo_root, args.java_cmd, dt, args, state_path, props_path)
            elapsed = time.perf_counter() - t0
            print(f"Finished dt={dt:g} in {elapsed:.2f}s", flush=True)

        header, rows = load_frames(state_path)
        times, energies = compute_energy_series(header, rows)
        if times.size == 0:
            raise ValueError(f"No frames found for dt={dt} in {state_path}")

        if args.relative:
            baseline = energies[0]
            values = (energies - baseline) / baseline if baseline != 0.0 else energies
        else:
            values = energies

        series.append((dt, times, values))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 20,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    all_values = []
    for dt, times, values in series:
        all_values.append(values)
        ax.plot(times, values, linewidth=1.5, label=f"dt={dt:g}")

    y_values = np.concatenate(all_values) if all_values else np.array([0.0])

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Energia relativa (E - E0) / E0" if args.relative else "Energia total (J)")
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(build_plain_formatter(y_values))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    vmin = float(np.min(y_values))
    vmax = float(np.max(y_values))
    span = vmax - vmin
    if span <= 0.0:
        span = abs(vmax) if vmax != 0.0 else 1.0
    pad = span * args.y_pad
    ax.set_ylim(vmin - pad, vmax + pad)

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, len(series)),
        borderaxespad=0.0,
    )
    ax.margins(x=0.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(args.out, dpi=220)
    plt.close(fig)

    print(f"Figure saved to: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
