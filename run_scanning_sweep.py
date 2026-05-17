#!/usr/bin/env python3
"""Run a particle-count sweep for ScanningRateSimulation.

Default sweep:
- N from 100 to 1000 (step 50)
- 5 repetitions per N
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sweep of ScanningRateSimulation with multiple N values and repetitions."
    )
    parser.add_argument("--n-start", type=int, default=100)
    parser.add_argument("--n-end", type=int, default=1000)
    parser.add_argument("--n-step", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=5)

    parser.add_argument("--tf", type=float, default=2000.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--dt2", type=float, default=0.1)

    parser.add_argument("--k", type=float, default=1.0e3)
    parser.add_argument("--l", type=float, default=80.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--v0", type=float, default=1.0)

    parser.add_argument("--seed-base", type=int, default=900000)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs") / "sweep_n_100_1000")

    parser.add_argument(
        "--java-cmd",
        type=str,
        default="java",
        help="Java executable (default: java from PATH)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip javac compilation step.",
    )

    return parser.parse_args()


def compile_simulation(repo_root: Path, java_cmd: str) -> None:
    simulation_dir = repo_root / "simulation"
    java_files = sorted(simulation_dir.glob("*.java"))
    if not java_files:
        raise FileNotFoundError(f"No Java files found in {simulation_dir}")

    javac_cmd = java_cmd.replace("java", "javac") if java_cmd.endswith("java") else "javac"
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


def run_single_case(
    args: argparse.Namespace,
    repo_root: Path,
    cp_dir: Path,
    output_root: Path,
    n_particles: int,
    repetition: int,
) -> tuple[int, int, int]:
    seed = args.seed_base + n_particles * 1000 + repetition

    run_dir = output_root / f"n{n_particles}_rep{repetition}"
    run_dir.mkdir(parents=True, exist_ok=True)

    state_path = run_dir / "states.txt"
    events_path = run_dir / "events.txt"
    props_path = run_dir / "properties.txt"

    cmd = [
        args.java_cmd,
        "-cp",
        str(cp_dir),
        "ScanningRateSimulation",
        "--n",
        str(n_particles),
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
        str(args.dt2),
        "--seed",
        str(seed),
        "--out",
        str(state_path),
        "--events-out",
        str(events_path),
        "--properties-out",
        str(props_path),
    ]

    process = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)

    if process.returncode != 0:
        raise RuntimeError(
            "Simulation failed.\n"
            f"N={n_particles}, rep={repetition}, seed={seed}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    if not state_path.exists() or not events_path.exists() or not props_path.exists():
        raise FileNotFoundError(
            f"Missing output files in {run_dir}. Expected states.txt, events.txt, properties.txt"
        )

    return n_particles, repetition, seed


def run_sweep(args: argparse.Namespace, repo_root: Path) -> None:
    cp_dir = repo_root / "simulation"
    output_root = repo_root / args.outputs_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.n_step <= 0:
        raise ValueError("--n-step must be > 0")
    if args.n_end < args.n_start:
        raise ValueError("--n-end must be >= --n-start")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be > 0")
    if args.threads <= 0:
        raise ValueError("--threads must be > 0")

    n_values = list(range(args.n_start, args.n_end + 1, args.n_step))
    total_runs = len(n_values) * args.repetitions
    current_run = 0

    for n_particles in n_values:
        workers = min(args.threads, args.repetitions)
        print(f"Running N={n_particles} with {workers} thread(s) ...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_single_case,
                    args,
                    repo_root,
                    cp_dir,
                    output_root,
                    n_particles,
                    repetition,
                )
                for repetition in range(1, args.repetitions + 1)
            ]

            for future in as_completed(futures):
                n_done, rep_done, seed_done = future.result()
                current_run += 1
                print(f"[{current_run}/{total_runs}] Done N={n_done}, rep={rep_done}, seed={seed_done}")

    print("Sweep completed successfully.")
    print(f"Outputs saved in: {output_root}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    if not args.skip_compile:
        compile_simulation(repo_root, args.java_cmd)

    run_sweep(args, repo_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
