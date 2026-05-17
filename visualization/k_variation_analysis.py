#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunResult:
    k_value: float
    n_particles: int
    repetition: int
    seed: int
    scanning_rate_j: float
    jin_s2: float
    rho_s2: float
    v_s2: float
    run_dir: Path


@dataclass(frozen=True)
class AggregatedPoint:
    k_value: float
    n_particles: int
    j_mean: float
    j_std: float
    jin_mean: float
    jin_std: float
    rho_mean: float
    rho_std: float
    v_mean: float
    v_std: float


@dataclass(frozen=True)
class ScalarSummary:
    k_value: float
    max_j: float
    n_star_j: int
    max_jin: float
    n_star_jin: int


def parse_list_float(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_list_int(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def fit_line(x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[float, float, float]:
    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)

    var_x = sum((x - mean_x) ** 2 for x in x_values)
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))

    if var_x <= 1.0e-15:
        slope = 0.0
        intercept = mean_y
    else:
        slope = cov_xy / var_x
        intercept = mean_y - slope * mean_x

    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(x_values, y_values))
    ss_tot = sum((y - mean_y) ** 2 for y in y_values)

    r2 = 1.0 if ss_tot <= 1.0e-15 else 1.0 - ss_res / ss_tot
    return slope, intercept, r2


def parse_events_to_cfc(events_path: Path) -> Tuple[List[float], List[int]]:
    times = [0.0]
    cfc_values = [0]
    cumulative = 0

    with events_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                t = float(parts[0])
                event_type = parts[2]
            except ValueError:
                continue

            if event_type == "FRESH_TO_USED":
                cumulative += 1
                times.append(t)
                cfc_values.append(cumulative)

    if len(times) < 2:
        raise ValueError(f"No se pudo reconstruir Cfc(t) desde {events_path}")

    return times, cfc_values


def compute_scanning_rate(events_path: Path, stationary_start: float) -> float:
    times, cfc = parse_events_to_cfc(events_path)

    filtered = [(t, c) for t, c in zip(times, cfc) if t >= stationary_start]
    if len(filtered) < 2:
        filtered = list(zip(times, cfc))

    x = [float(t) for t, _ in filtered]
    y = [float(c) for _, c in filtered]

    slope, _, _ = fit_line(x, y)
    return abs(slope)


def build_radial_bins(r_min: float, r_max: float, ds: float) -> Tuple[List[float], List[float]]:
    n_bins = int(math.floor((r_max - r_min) / ds))
    s_centers = []
    areas = []

    for k in range(n_bins):
        s_in = r_min + k * ds
        s_out = s_in + ds
        s_centers.append(0.5 * (s_in + s_out))
        areas.append(math.pi * (s_out * s_out - s_in * s_in))

    return s_centers, areas


def nearest_bin_index(values: Sequence[float], target: float) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i] - target))

def parse_state_file_s2(
    state_path: Path,
    r0: float,
    particle_radius: float,
    l: float,
    ds: float,
    target_s: float,
    stationary_start: float,
) -> Tuple[float, float, float]:
    r_min = r0 + particle_radius
    r_max = l / 2.0 - particle_radius

    s_centers, areas = build_radial_bins(r_min, r_max, ds)
    target_bin = nearest_bin_index(s_centers, target_s)

    total_count = 0
    total_sum_vr = 0.0
    frame_count = 0

    current_time: float | None = None
    current_particles: List[Tuple[float, float, float, float, str]] = []

    def is_fresh_state(state: str) -> bool:
        return state == "FRESH" or state == "1"

    def process_frame(
        time_value: float | None,
        particles: Sequence[Tuple[float, float, float, float, str]],
    ) -> None:
        nonlocal total_count, total_sum_vr, frame_count

        if time_value is None:
            return
        if time_value < stationary_start:
            return
        if not particles:
            return

        frame_count += 1

        for x, y, vx, vy, state in particles:
            if not is_fresh_state(state):
                continue

            radial = math.hypot(x, y)
            if radial < r_min or radial >= r_max:
                continue

            dot = x * vx + y * vy
            if dot >= 0.0:
                continue

            bin_index = int(math.floor((radial - r_min) / ds))
            if bin_index != target_bin:
                continue

            vr = dot / radial
            total_count += 1
            total_sum_vr += vr

    with state_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Ignora líneas raras tipo "0000 m"
            if len(parts) < 7:
                continue

            try:
                time_value = float(parts[0])
                # parts[1] = id
                x = float(parts[2])
                y = float(parts[3])
                vx = float(parts[4])
                vy = float(parts[5])
                state = parts[6]
            except ValueError:
                continue

            if current_time is None:
                current_time = time_value

            if abs(time_value - current_time) > 1.0e-12:
                process_frame(current_time, current_particles)
                current_particles = []
                current_time = time_value

            current_particles.append((x, y, vx, vy, state))

    process_frame(current_time, current_particles)

    if frame_count <= 0:
        raise ValueError(f"No hay frames válidos en {state_path}")

    area = areas[target_bin]
    rho = total_count / (area * frame_count)
    v = total_sum_vr / total_count if total_count > 0 else 0.0
    jin = rho * abs(v)

    return rho, v, jin


def run_single_simulation(
    repo_root: Path,
    run_dir: Path,
    java_cmd: str,
    n_particles: int,
    k_value: float,
    tf: float,
    dt: float,
    dt2: float,
    seed: int,
) -> Tuple[Path, Path]:
    sim_dir = repo_root / "simulation"
    run_dir.mkdir(parents=True, exist_ok=True)

    state_path = run_dir / "states.txt"
    events_path = run_dir / "events.txt"
    properties_path = run_dir / "properties.txt"

    cmd = [
        java_cmd,
        "-cp",
        str(sim_dir),
        "ScanningRateSimulation",
        "--n",
        str(n_particles),
        "--k",
        str(k_value),
        "--tf",
        str(tf),
        "--dt",
        str(dt),
        "--dt2",
        str(dt2),
        "--seed",
        str(seed),
        "--out",
        str(state_path),
        "--events-out",
        str(events_path),
        "--properties-out",
        str(properties_path),
    ]

    process = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"Falló simulación k={k_value}, N={n_particles}, seed={seed}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    if not state_path.exists():
        raise FileNotFoundError(f"No se encontró {state_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"No se encontró {events_path}")

    return state_path, events_path


def collect_results(args: argparse.Namespace) -> List[RunResult]:
    repo_root = Path(__file__).resolve().parent.parent
    n_values = parse_list_int(args.n_values)
    k_values = parse_list_float(args.k_values)

    java_cmd = args.java_cmd
    if not os.path.exists(java_cmd):
        raise FileNotFoundError(f"No se encontró Java: {java_cmd}")

    results: List[RunResult] = []

    for k_value in k_values:
        for n_particles in n_values:
            for repetition in range(1, args.repetitions + 1):
                seed = args.seed_base + int(k_value) * 10_000_000 + n_particles * 1000 + repetition
                run_dir = args.outputs_root / f"k{format_k(k_value)}" / f"n{n_particles}_rep{repetition}"

                if args.reuse_existing_runs:
                    state_path = run_dir / "states.txt"
                    events_path = run_dir / "events.txt"
                    if not state_path.exists() or not events_path.exists():
                        raise FileNotFoundError(f"Faltan archivos en {run_dir}")
                else:
                    state_path, events_path = run_single_simulation(
                        repo_root=repo_root,
                        run_dir=run_dir,
                        java_cmd=java_cmd,
                        n_particles=n_particles,
                        k_value=k_value,
                        tf=args.tf,
                        dt=args.dt,
                        dt2=args.dt2,
                        seed=seed,
                    )

                j = compute_scanning_rate(events_path, args.stationary_start)

                rho_s2, v_s2, jin_s2 = parse_state_file_s2(
                    state_path=state_path,
                    r0=args.r0,
                    particle_radius=args.particle_radius,
                    l=args.l,
                    ds=args.ds,
                    target_s=args.target_s,
                    stationary_start=args.stationary_start,
                )

                results.append(
                    RunResult(
                        k_value=k_value,
                        n_particles=n_particles,
                        repetition=repetition,
                        seed=seed,
                        scanning_rate_j=j,
                        jin_s2=jin_s2,
                        rho_s2=rho_s2,
                        v_s2=v_s2,
                        run_dir=run_dir,
                    )
                )

                print(
                    f"[OK] k={k_value:.1e} N={n_particles} rep={repetition} "
                    f"J={j:.5e} Jin={jin_s2:.5e}"
                )

    return results


def aggregate_results(results: Sequence[RunResult]) -> List[AggregatedPoint]:
    grouped: Dict[Tuple[float, int], List[RunResult]] = {}

    for result in results:
        grouped.setdefault((result.k_value, result.n_particles), []).append(result)

    points: List[AggregatedPoint] = []

    def mean_std(values: Sequence[float]) -> Tuple[float, float]:
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    for (k_value, n_particles), group in sorted(grouped.items()):
        j_mean, j_std = mean_std([x.scanning_rate_j for x in group])
        jin_mean, jin_std = mean_std([x.jin_s2 for x in group])
        rho_mean, rho_std = mean_std([x.rho_s2 for x in group])
        v_mean, v_std = mean_std([x.v_s2 for x in group])

        points.append(
            AggregatedPoint(
                k_value=k_value,
                n_particles=n_particles,
                j_mean=j_mean,
                j_std=j_std,
                jin_mean=jin_mean,
                jin_std=jin_std,
                rho_mean=rho_mean,
                rho_std=rho_std,
                v_mean=v_mean,
                v_std=v_std,
            )
        )

    return points


def summarize_scalars(points: Sequence[AggregatedPoint]) -> List[ScalarSummary]:
    grouped: Dict[float, List[AggregatedPoint]] = {}

    for point in points:
        grouped.setdefault(point.k_value, []).append(point)

    summaries: List[ScalarSummary] = []

    for k_value, group in sorted(grouped.items()):
        best_j = max(group, key=lambda x: x.j_mean)
        best_jin = max(group, key=lambda x: x.jin_mean)

        summaries.append(
            ScalarSummary(
                k_value=k_value,
                max_j=best_j.j_mean,
                n_star_j=best_j.n_particles,
                max_jin=best_jin.jin_mean,
                n_star_jin=best_jin.n_particles,
            )
        )

    return summaries


def format_k(k_value: float) -> str:
    return f"{k_value:.0e}".replace("+", "").replace(".", "p")


def write_results_csv(results: Sequence[RunResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "k_value",
                "n_particles",
                "repetition",
                "seed",
                "scanning_rate_j",
                "jin_s2",
                "rho_s2",
                "v_s2",
                "run_dir",
            ]
        )

        for r in results:
            writer.writerow(
                [
                    f"{r.k_value:.10e}",
                    r.n_particles,
                    r.repetition,
                    r.seed,
                    f"{r.scanning_rate_j:.10e}",
                    f"{r.jin_s2:.10e}",
                    f"{r.rho_s2:.10e}",
                    f"{r.v_s2:.10e}",
                    str(r.run_dir),
                ]
            )


def read_results_csv(input_csv: Path) -> List[RunResult]:
    results: List[RunResult] = []

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            results.append(
                RunResult(
                    k_value=float(row["k_value"]),
                    n_particles=int(row["n_particles"]),
                    repetition=int(row["repetition"]),
                    seed=int(row["seed"]),
                    scanning_rate_j=float(row["scanning_rate_j"]),
                    jin_s2=float(row["jin_s2"]),
                    rho_s2=float(row["rho_s2"]),
                    v_s2=float(row["v_s2"]),
                    run_dir=Path(row["run_dir"]),
                )
            )

    return results


def plot_j_vs_n(points: Sequence[AggregatedPoint], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = sorted({p.k_value for p in points})

    for k_value in k_values:
        group = [p for p in points if p.k_value == k_value]
        ns = [p.n_particles for p in group]
        means = [p.j_mean for p in group]
        stds = [p.j_std for p in group]

        ax.errorbar(
            ns,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=5,
            label=rf"$k={k_value:.0e}$",
        )

    ax.set_xlabel("N")
    ax.set_ylabel(r"$\langle J \rangle$ $(s^{-1})$")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Constante elástica")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_jin_vs_n(points: Sequence[AggregatedPoint], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = sorted({p.k_value for p in points})

    for k_value in k_values:
        group = [p for p in points if p.k_value == k_value]
        ns = [p.n_particles for p in group]
        means = [p.jin_mean for p in group]
        stds = [p.jin_std for p in group]

        ax.errorbar(
            ns,
            means,
            yerr=stds,
            marker="s",
            linewidth=2,
            capsize=5,
            label=rf"$k={k_value:.0e}$",
        )

    ax.set_xlabel("N")
    ax.set_ylabel(r"$\langle J_{in}|_{S\sim2} \rangle$")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Constante elástica")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_scalars_vs_k(summaries: Sequence[ScalarSummary], output_path: Path) -> None:
    k_values = [s.k_value for s in summaries]
    max_j = [s.max_j for s in summaries]
    max_jin = [s.max_jin for s in summaries]
    n_star_j = [s.n_star_j for s in summaries]
    n_star_jin = [s.n_star_jin for s in summaries]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(k_values, max_j, marker="o", linewidth=2, label=r"$\max \langle J \rangle$")
    axes[0].plot(k_values, max_jin, marker="s", linewidth=2, label=r"$\max \langle J_{in} \rangle$")
    axes[0].set_ylabel("Máximo de la curva")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(k_values, n_star_j, marker="o", linewidth=2, label=r"$N^*_J$")
    axes[1].plot(k_values, n_star_jin, marker="s", linewidth=2, label=r"$N^*_{Jin}$")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$k$ $(N/m)$")
    axes[1].set_ylabel(r"$N^*$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[0].set_xscale("log")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_scalars_csv(summaries: Sequence[ScalarSummary], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k_value", "max_j", "n_star_j", "max_jin", "n_star_jin"])

        for s in summaries:
            writer.writerow(
                [
                    f"{s.k_value:.10e}",
                    f"{s.max_j:.10e}",
                    s.n_star_j,
                    f"{s.max_jin:.10e}",
                    s.n_star_jin,
                ]
            )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="TP4 1.4: variación de k")

    parser.add_argument("--k-values", type=str, default="1e2,1e3,1e4")
    parser.add_argument("--n-values", type=str, default="100,200,300,400,500,600,700,800,900,1000")
    parser.add_argument("--repetitions", type=int, default=5)

    parser.add_argument("--tf", type=float, default=1500.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--dt2", type=float, default=0.1)
    parser.add_argument("--stationary-start", type=float, default=0.0)

    parser.add_argument("--l", type=float, default=80.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--particle-radius", type=float, default=1.0)
    parser.add_argument("--ds", type=float, default=0.2)
    parser.add_argument("--target-s", type=float, default=2.1)

    parser.add_argument("--seed-base", type=int, default=700000)

    parser.add_argument(
        "--java-cmd",
        type=str,
        default="C:/Program Files/JetBrains/IntelliJ IDEA 2025.3.3/jbr/bin/java.exe",
    )

    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=repo_root / "outputs" / "k_variation_tp4",
    )

    parser.add_argument(
        "--out-csv",
        type=Path,
        default=repo_root / "outputs" / "k_variation_tp4" / "k_variation_results.csv",
    )

    parser.add_argument(
        "--scalars-csv",
        type=Path,
        default=repo_root / "outputs" / "k_variation_tp4" / "k_variation_scalars.csv",
    )

    parser.add_argument(
        "--j-figure",
        type=Path,
        default=repo_root / "outputs" / "k_variation_tp4" / "j_vs_n_by_k.png",
    )

    parser.add_argument(
        "--jin-figure",
        type=Path,
        default=repo_root / "outputs" / "k_variation_tp4" / "jin_s2_vs_n_by_k.png",
    )

    parser.add_argument(
        "--scalars-figure",
        type=Path,
        default=repo_root / "outputs" / "k_variation_tp4" / "scalars_vs_k.png",
    )

    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--reuse-existing-runs", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.only_plot:
        results = read_results_csv(args.out_csv)
    else:
        results = collect_results(args)
        write_results_csv(results, args.out_csv)
        print(f"Resultados guardados en {args.out_csv}")

    points = aggregate_results(results)
    summaries = summarize_scalars(points)

    plot_j_vs_n(points, args.j_figure)
    print(f"Figura J vs N guardada en {args.j_figure}")

    plot_jin_vs_n(points, args.jin_figure)
    print(f"Figura Jin vs N guardada en {args.jin_figure}")

    write_scalars_csv(summaries, args.scalars_csv)
    print(f"Escalares guardados en {args.scalars_csv}")

    plot_scalars_vs_k(summaries, args.scalars_figure)
    print(f"Figura escalares vs k guardada en {args.scalars_figure}")


if __name__ == "__main__":
    main()