"""Run k-variation sweep for TP4 1.4.

Genera corridas para:
- k values: 1e2, 1e3, 1e4
- N values: configurable
- repetitions per (k, N)

Output:
outputs/k_variation_tp4/k1e02/n100_rep1/
    states.txt
    events.txt
    properties.txt
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path


def parse_csv_floats(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("Debe haber al menos un valor de k")
    return values


def format_k(k_value: float) -> str:
    return f"{k_value:.0e}".replace("+", "").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TP4 1.4 sweep over k, N and repetitions."
    )

    parser.add_argument("--k-values", type=str, default="1e2,1e3,1e4")

    parser.add_argument("--n-start", type=int, default=100)
    parser.add_argument("--n-end", type=int, default=1000)
    parser.add_argument("--n-step", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)

    parser.add_argument(
        "--threads",
        type=int,
        default=5,
        help="Threads per (k, N), usually <= repetitions.",
    )

    parser.add_argument("--tf", type=float, default=2000.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--dt2", type=float, default=0.1)

    parser.add_argument("--l", type=float, default=80.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--v0", type=float, default=1.0)

    parser.add_argument("--seed-base", type=int, default=700000)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs") / "k_variation_tp4",
    )

    parser.add_argument(
        "--java-cmd",
        type=str,
        default="java",
        help="Java executable, e.g. C:/Program Files/Java/jdk-25.0.2/bin/java.exe",
    )

    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip javac compilation step.",
    )

    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Do not write states.txt. Faster, but disables Jin analysis.",
    )

    return parser.parse_args()


def compile_simulation(repo_root: Path, java_cmd: str) -> None:
    simulation_dir = repo_root / "simulation"
    java_files = sorted(simulation_dir.glob("*.java"))
    if not java_files:
        raise FileNotFoundError(f"No Java files found in {simulation_dir}")

    if java_cmd.endswith("java.exe"):
        javac_cmd = java_cmd[:-len("java.exe")] + "javac.exe"
    elif java_cmd.endswith("java"):
        javac_cmd = java_cmd[:-len("java")] + "javac"
    else:
        javac_cmd = "javac"

    cmd = [javac_cmd] + [str(path) for path in java_files]

    print("Compiling Java sources...")
    process = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

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
    k_value: float,
    n_particles: int,
    repetition: int,
) -> tuple[float, int, int, int]:
    seed = args.seed_base + int(k_value) * 10_000_000 + n_particles * 1000 + repetition

    run_dir = output_root / f"k{format_k(k_value)}" / f"n{n_particles}_rep{repetition}"
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
        str(k_value),
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
        "--events-out",
        str(events_path),
        "--properties-out",
        str(props_path),
    ]

    if args.no_state:
        cmd.append("--no-state")
    else:
        cmd.extend(["--out", str(state_path)])

    process = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            "Simulation failed.\n"
            f"k={k_value}, N={n_particles}, rep={repetition}, seed={seed}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    if not events_path.exists() or not props_path.exists():
        raise FileNotFoundError(
            f"Missing events.txt/properties.txt in {run_dir}"
        )

    if not args.no_state and not state_path.exists():
        raise FileNotFoundError(f"Missing states.txt in {run_dir}")

    return k_value, n_particles, repetition, seed


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

    k_values = parse_csv_floats(args.k_values)
    n_values = list(range(args.n_start, args.n_end + 1, args.n_step))

    total_runs = len(k_values) * len(n_values) * args.repetitions
    current_run = 0

    for k_value in k_values:
        for n_particles in n_values:
            workers = min(args.threads, args.repetitions)
            print(f"Running k={k_value:.1e}, N={n_particles} with {workers} thread(s) ...")

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        run_single_case,
                        args,
                        repo_root,
                        cp_dir,
                        output_root,
                        k_value,
                        n_particles,
                        repetition,
                    )
                    for repetition in range(1, args.repetitions + 1)
                ]

                for future in as_completed(futures):
                    k_done, n_done, rep_done, seed_done = future.result()
                    current_run += 1
                    print(
                        f"[{current_run}/{total_runs}] Done "
                        f"k={k_done:.1e}, N={n_done}, rep={rep_done}, seed={seed_done}"
                    )

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