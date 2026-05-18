#!/usr/bin/env python3
"""TP4 1.5 - Tiempo de retorno de la primera partícula usada.

Para cada corrida:
- busca la primera partícula que cambia FRESH -> USED,
- luego busca cuándo esa misma partícula cambia USED -> FRESH,
- calcula delta_t = t_return - t_used.

Analiza delta_t en función de N y k.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FirstReturnRecord:
    k_value: float
    n_particles: int
    repetition: int
    seed: int
    particle_id: int
    t_used: float
    t_return: float
    delta_t: float
    run_dir: Path


@dataclass(frozen=True)
class FirstReturnStats:
    k_value: float
    n_particles: int
    mean_delta_t: float
    std_delta_t: float
    sample_count: int


def parse_list_float(raw: str) -> List[float]:
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("Debe haber al menos un valor de k")
    return values


def parse_list_int(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("Debe haber al menos un valor de N")
    return values


def format_k(k_value: float) -> str:
    return f"{k_value:.0e}".replace("+", "").replace(".", "p")


def parse_event_line(parts: List[str]) -> Tuple[float, int, str] | None:

    if len(parts) < 3:
        return None

    try:
        time_value = float(parts[0])
    except ValueError:
        return None

    event_type = parts[2]

    try:
        particle_id = int(parts[1])
    except ValueError:
        # fallback por si el id está en otra columna
        particle_id = -1

    return time_value, particle_id, event_type


def compute_first_used_return_time(events_path: Path) -> Tuple[int, float, float, float]:
    if not events_path.exists():
        raise FileNotFoundError(f"No se encontró events.txt en {events_path}")

    first_particle_id: int | None = None
    t_used: float | None = None

    with events_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            parsed = parse_event_line(parts)
            if parsed is None:
                continue

            time_value, particle_id, event_type = parsed

            if first_particle_id is None:
                if event_type == "FRESH_TO_USED":
                    first_particle_id = particle_id
                    t_used = time_value
                continue

            if particle_id == first_particle_id and event_type == "USED_TO_FRESH":
                t_return = time_value
                delta_t = t_return - t_used
                if delta_t < 0:
                    raise ValueError(
                        f"Delta t negativo en {events_path}: "
                        f"t_used={t_used}, t_return={t_return}"
                    )
                return first_particle_id, t_used, t_return, delta_t

    if first_particle_id is None or t_used is None:
        raise ValueError(f"No se encontró ningún evento FRESH_TO_USED en {events_path}")

    raise ValueError(
        f"La primera partícula usada id={first_particle_id} en t={t_used} "
        f"no volvió al borde exterior durante la simulación: {events_path}"
    )


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
) -> Path:
    sim_dir = repo_root / "simulation"
    run_dir.mkdir(parents=True, exist_ok=True)

    events_path = run_dir / "events.txt"
    properties_path = run_dir / "properties.txt"

    # Para 1.5 sólo necesitamos eventos, no estados.
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
        "--no-state",
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

    if not events_path.exists():
        raise FileNotFoundError(f"No se encontró {events_path}")

    return events_path


def collect_records(args: argparse.Namespace) -> List[FirstReturnRecord]:
    repo_root = Path(__file__).resolve().parent.parent
    k_values = parse_list_float(args.k_values)
    n_values = parse_list_int(args.n_values)

    java_cmd = args.java_cmd
    if not os.path.exists(java_cmd):
        raise FileNotFoundError(f"No se encontró Java: {java_cmd}")

    records: List[FirstReturnRecord] = []

    for k_value in k_values:
        for n_particles in n_values:
            for repetition in range(1, args.repetitions + 1):
                seed = (
                    args.seed_base
                    + int(k_value) * 10_000_000
                    + n_particles * 1000
                    + repetition
                )

                run_dir = (
                    args.outputs_root
                    / f"k{format_k(k_value)}"
                    / f"n{n_particles}_rep{repetition}"
                )

                if args.reuse_existing_runs:
                    events_path = run_dir / "events.txt"
                    if not events_path.exists():
                        raise FileNotFoundError(f"No se encontró events.txt en {run_dir}")
                else:
                    events_path = run_single_simulation(
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

                try:
                    particle_id, t_used, t_return, delta_t = compute_first_used_return_time(events_path)
                except ValueError as exc:
                    if args.skip_missing_returns:
                        print(f"[SKIP] k={k_value:.1e} N={n_particles} rep={repetition}: {exc}")
                        continue
                    raise

                records.append(
                    FirstReturnRecord(
                        k_value=k_value,
                        n_particles=n_particles,
                        repetition=repetition,
                        seed=seed,
                        particle_id=particle_id,
                        t_used=t_used,
                        t_return=t_return,
                        delta_t=delta_t,
                        run_dir=run_dir,
                    )
                )

                print(
                    f"[OK] k={k_value:.1e} N={n_particles} rep={repetition} "
                    f"id={particle_id} dt_return={delta_t:.6f} s"
                )

    return records


def write_records_csv(records: Sequence[FirstReturnRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "k_value",
                "n_particles",
                "repetition",
                "seed",
                "particle_id",
                "t_used",
                "t_return",
                "delta_t",
                "run_dir",
            ]
        )

        for record in records:
            writer.writerow(
                [
                    f"{record.k_value:.10e}",
                    record.n_particles,
                    record.repetition,
                    record.seed,
                    record.particle_id,
                    f"{record.t_used:.10f}",
                    f"{record.t_return:.10f}",
                    f"{record.delta_t:.10f}",
                    str(record.run_dir),
                ]
            )


def read_records_csv(input_csv: Path) -> List[FirstReturnRecord]:
    records: List[FirstReturnRecord] = []

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                FirstReturnRecord(
                    k_value=float(row["k_value"]),
                    n_particles=int(row["n_particles"]),
                    repetition=int(row["repetition"]),
                    seed=int(row["seed"]),
                    particle_id=int(row["particle_id"]),
                    t_used=float(row["t_used"]),
                    t_return=float(row["t_return"]),
                    delta_t=float(row["delta_t"]),
                    run_dir=Path(row["run_dir"]),
                )
            )

    if not records:
        raise ValueError(f"No hay datos en {input_csv}")

    return records


def aggregate_records(records: Sequence[FirstReturnRecord]) -> List[FirstReturnStats]:
    grouped: Dict[Tuple[float, int], List[float]] = {}

    for record in records:
        grouped.setdefault((record.k_value, record.n_particles), []).append(record.delta_t)

    stats: List[FirstReturnStats] = []

    for (k_value, n_particles), values in sorted(grouped.items()):
        mean_delta_t = statistics.fmean(values)
        std_delta_t = statistics.stdev(values) if len(values) > 1 else 0.0

        stats.append(
            FirstReturnStats(
                k_value=k_value,
                n_particles=n_particles,
                mean_delta_t=mean_delta_t,
                std_delta_t=std_delta_t,
                sample_count=len(values),
            )
        )

    return stats
def plot_delta_t_vs_n(
    stats: Sequence[FirstReturnStats],
    output_path: Path,
) -> None:

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
        }
    )

    k_values = sorted({entry.k_value for entry in stats})

    fig, ax = plt.subplots(figsize=(10, 6))

    # offsets horizontales
    offsets = np.linspace(-12, 12, len(k_values))

    for offset, k_value in zip(offsets, k_values):

        group = [entry for entry in stats if entry.k_value == k_value]

        ns = np.array([entry.n_particles for entry in group], dtype=float)
        means = [entry.mean_delta_t for entry in group]
        stds = [entry.std_delta_t for entry in group]

        shifted_ns = ns + offset

        ax.errorbar(
            shifted_ns,
            means,
            yerr=stds,
            fmt="o-",
            linewidth=1.8,
            markersize=5,
            capsize=3,
            elinewidth=0.9,
            capthick=0.9,
            label=rf"$k={k_value:.0f}$"
        )

    ax.set_xlabel("Número de partículas (N)", fontsize=22)

    ax.set_ylabel(
        r"$\Delta t_{\mathrm{retorno}}$ (s)",
        fontsize=22,
    )

    ax.tick_params(axis="both", labelsize=16)

    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=14)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=200)

    plt.close(fig)
def plot_delta_t_vs_k(
    stats: Sequence[FirstReturnStats],
    output_path: Path,
) -> None:
    """Promedia sobre N para obtener un escalar por k."""

    grouped: Dict[float, List[float]] = {}

    for entry in stats:
        grouped.setdefault(entry.k_value, []).append(entry.mean_delta_t)

    k_values = sorted(grouped.keys())

    means = [statistics.fmean(grouped[k]) for k in k_values]

    stds = [
        statistics.stdev(grouped[k]) if len(grouped[k]) > 1 else 0.0
        for k in k_values
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        k_values,
        means,
        yerr=stds,
        fmt="o-",
        linewidth=1.8,
        markersize=5,
        capsize=3,
        elinewidth=0.9,
        capthick=0.9,
    )

    ax.set_xscale("log")

    ax.set_xlabel(r"$k$ (N/m)", fontsize=20)

    ax.set_ylabel(
        r"$\langle \Delta t_{\mathrm{retorno}} \rangle_N$ (s)",
        fontsize=20,
    )

    ax.tick_params(axis="both", labelsize=16)

    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=200)

    plt.close(fig)

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="TP4 1.5: tiempo de retorno de la primera partícula usada."
    )

    parser.add_argument("--k-values", type=str, default="1e2,1e3,1e4,1e5")
    parser.add_argument(
        "--n-values",
        type=str,
        default="100,200,300,400,500,600,700,800,900,1000",
    )
    parser.add_argument("--repetitions", type=int, default=5)

    parser.add_argument("--tf", type=float, default=2000.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--dt2", type=float, default=0.1)
    parser.add_argument("--seed-base", type=int, default=800000)

    parser.add_argument(
        "--java-cmd",
        type=str,
        default="C:/Program Files/JetBrains/IntelliJ IDEA 2025.3.3/jbr/bin/java.exe",
    )

    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=repo_root / "outputs" / "first_return_tp4",
    )

    parser.add_argument(
        "--out-csv",
        type=Path,
        default=repo_root / "outputs" / "first_return_tp4" / "first_return_results.csv",
    )

    parser.add_argument(
        "--figure-n",
        type=Path,
        default=repo_root / "outputs" / "first_return_tp4" / "first_return_vs_n.png",
    )

    parser.add_argument(
        "--figure-k",
        type=Path,
        default=repo_root / "outputs" / "first_return_tp4" / "first_return_vs_k.png",
    )

    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--reuse-existing-runs", action="store_true")
    parser.add_argument(
        "--skip-missing-returns",
        action="store_true",
        help="Si la primera partícula usada no vuelve antes de tf, saltea esa corrida.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.only_plot:
        records = read_records_csv(args.out_csv)
    else:
        records = collect_records(args)
        write_records_csv(records, args.out_csv)
        print(f"Resultados guardados en {args.out_csv}")

    stats = aggregate_records(records)

    plot_delta_t_vs_n(stats, args.figure_n)
    print(f"Figura Δt vs N guardada en {args.figure_n}")

    plot_delta_t_vs_k(stats, args.figure_k)
    print(f"Figura Δt vs k guardada en {args.figure_k}")


if __name__ == "__main__":
    main()