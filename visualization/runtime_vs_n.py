#!/usr/bin/env python3
"""TP4 1.1 - Runtime versus particle count N (Time-Step Driven Molecular Dynamics).

Wrapper script (renamed from tp4_runtime_vs_n.py).
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RuntimeRecord:
    n_particles: int
    repetition: int
    seed: int
    execution_time_s: float
    run_dir: Path


@dataclass(frozen=True)
class RuntimeStats:
    n_particles: int
    mean_s: float
    std_s: float
    sample_count: int


def parse_properties(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_n_values(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("Todos los valores de N deben ser > 0")
        values.append(value)

    if not values:
        raise ValueError("Debe haber al menos un valor de N")

    return values


def run_single_simulation_tp4(
    repo_root: Path,
    run_dir: Path,
    n_particles: int,
    tf_seconds: float,
    seed: int,
) -> RuntimeRecord:
    java_cmd = (
        "C:/Program Files/JetBrains/IntelliJ IDEA 2025.3.3/jbr/bin/java.exe"
    )
    if not os.path.exists(java_cmd):
        raise FileNotFoundError(f"Java executable not found: {java_cmd}")

    sim_dir = repo_root / "simulation"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = run_dir / "properties.txt"

    cmd = [
        java_cmd,
        "-cp",
        str(sim_dir),
        "ScanningRateSimulation",
        "--n",
        str(n_particles),
        "--tf",
        str(tf_seconds),
        "--seed",
        str(seed),
        "--no-output",
        "--properties-out",
        str(metadata_path),
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
            f"Fallo al ejecutar simulacion TP4 para N={n_particles}, seed={seed}, run_dir={run_dir}.\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontro properties.txt en {run_dir}")

    properties = parse_properties(metadata_path)
    if "execution_time_s" not in properties:
        raise KeyError(f"execution_time_s no encontrado en {metadata_path}")

    execution_time_s = float(properties["execution_time_s"])

    return RuntimeRecord(
        n_particles=n_particles,
        repetition=0,  # Set later
        seed=seed,
        execution_time_s=execution_time_s,
        run_dir=run_dir,
    )


def collect_runtime_records_tp4(
    repo_root: Path,
    n_values: Sequence[int],
    repetitions: int,
    tf_seconds: float,
    seed_base: int,
    outputs_base_dir: Path,
    run_prefix: str,
) -> List[RuntimeRecord]:
    outputs_base_dir.mkdir(parents=True, exist_ok=True)

    records: List[RuntimeRecord] = []
    for n_particles in n_values:
        for repetition in range(1, repetitions + 1):
            seed = seed_base + n_particles * 1000 + repetition
            run_dir = outputs_base_dir / f"{run_prefix}_n{n_particles}_rep{repetition}"

            record = run_single_simulation_tp4(
                repo_root=repo_root,
                run_dir=run_dir,
                n_particles=n_particles,
                tf_seconds=tf_seconds,
                seed=seed,
            )

            records.append(
                RuntimeRecord(
                    n_particles=record.n_particles,
                    repetition=repetition,
                    seed=seed,
                    execution_time_s=record.execution_time_s,
                    run_dir=record.run_dir,
                )
            )
            print(
                f"[OK] TP4 N={n_particles:4d} rep={repetition:2d} seed={seed} "
                f"runtime={record.execution_time_s:.6f} s"
            )

    return records


def write_runtime_csv(records: Sequence[RuntimeRecord], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["n_particles", "repetition", "seed", "execution_time_s", "run_dir"]
        )
        for record in records:
            writer.writerow(
                [
                    record.n_particles,
                    record.repetition,
                    record.seed,
                    f"{record.execution_time_s:.10f}",
                    str(record.run_dir),
                ]
            )


def aggregate_stats(records: Sequence[RuntimeRecord]) -> List[RuntimeStats]:
    grouped: Dict[int, List[float]] = {}
    for record in records:
        grouped.setdefault(record.n_particles, []).append(record.execution_time_s)

    stats: List[RuntimeStats] = []
    for n_particles in sorted(grouped.keys()):
        values = grouped[n_particles]
        mean_s = statistics.fmean(values)
        std_s = statistics.stdev(values) if len(values) > 1 else 0.0
        stats.append(
            RuntimeStats(
                n_particles=n_particles,
                mean_s=mean_s,
                std_s=std_s,
                sample_count=len(values),
            )
        )

    return stats


def load_tp3_runtime_stats_if_available(
    tp3_benchmark_csv: Path,
) -> List[RuntimeStats] | None:
    if not tp3_benchmark_csv.exists():
        return None
    try:
        records = []
        with tp3_benchmark_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(
                    RuntimeRecord(
                        n_particles=int(row["n_particles"]),
                        repetition=int(row["repetition"]),
                        seed=int(row["seed"]),
                        execution_time_s=float(row["execution_time_s"]),
                        run_dir=Path(row["run_dir"]),
                    )
                )
        return aggregate_stats(records)
    except Exception as e:
        print(f"Advertencia: No se pudo cargar datos TP3: {e}")
        return None


def plot_runtime_vs_n(
    tp4_stats: Sequence[RuntimeStats],
    tp3_stats: Sequence[RuntimeStats] | None,
    output_figure_path: Path,
    log_y: bool = False,
) -> None:
    tp4_ns = [s.n_particles for s in tp4_stats]
    tp4_means = [s.mean_s for s in tp4_stats]
    tp4_stds = [s.std_s for s in tp4_stats]

    import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        tp4_ns,
        tp4_means,
        yerr=tp4_stds,
        fmt="o-",
        capsize=5,
        capthick=2,
        label="TP4 (Time-Step)",
        linewidth=2,
        markersize=8,
    )

    if tp3_stats is not None:
        tp3_ns = [s.n_particles for s in tp3_stats]
        tp3_means = [s.mean_s for s in tp3_stats]
        tp3_stds = [s.std_s for s in tp3_stats]
        ax.errorbar(
            tp3_ns,
            tp3_means,
            yerr=tp3_stds,
            fmt="s--",
            capsize=5,
            capthick=2,
            label="TP3 (Event-Driven)",
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Número de partículas (N)")
    ax.set_ylabel("Tiempo de ejecución (s)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    # Force X ticks every 50
    try:
        min_n = min(tp4_ns)
        max_n = max(tp4_ns)
        xticks = list(range(int(min_n), int(max_n) + 1, 50))
        if xticks:
            ax.set_xticks(xticks)
    except Exception:
        pass
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_figure_path, dpi=200)
    print(f"Figura guardada en {output_figure_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="TP4 1.1: Runtime vs N for Time-Step Driven Molecular Dynamics."
    )

    parser.add_argument(
        "--n-values",
        type=str,
        default="100,150,200,250,300,350,400,450,500,550,600,650,700,750",
        help="Lista de N separada por comas.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Cantidad de corridas por cada N.",
    )
    parser.add_argument(
        "--tf",
        type=float,
        default=500.0,
        help="Tiempo final de simulacion en segundos.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=400000,
        help="Base para semillas reproducibles.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "runtime",
        help="Carpeta raiz donde se crean benchmarks.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="runtime",
        help="Prefijo de corrida.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Archivo CSV para guardar resultados TP4.",
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default="runtime_vs_tp3.png",
        help="Archivo de salida de la figura.",
    )
    parser.add_argument(
        "--tp3-csv",
        type=Path,
        default=None,
        help="CSV con datos TP3 para comparacion (opcional).",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Usar escala logaritmica en Y.",
    )

    args = parser.parse_args()

    if args.out_csv is None:
        args.out_csv = args.outputs_root / "results.csv"

    repo_root = Path(__file__).resolve().parent.parent

    n_values = parse_n_values(args.n_values)

    print(f"Benchmark dir: {args.outputs_root.resolve()}")
    print(f"Run prefix: {args.run_prefix}")
    print(f"Total corridas objetivo: {len(n_values) * args.repetitions}")
    print(f"tf={args.tf}s (tf para la simulacion)")

    records_tp4 = collect_runtime_records_tp4(
        repo_root=repo_root,
        n_values=n_values,
        repetitions=args.repetitions,
        tf_seconds=args.tf,
        seed_base=args.seed_base,
        outputs_base_dir=args.outputs_root,
        run_prefix=args.run_prefix,
    )

    write_runtime_csv(records_tp4, args.out_csv)
    print(f"Resultados guardados en {args.out_csv}")

    stats_tp4 = aggregate_stats(records_tp4)

    stats_tp3 = None
    if args.tp3_csv and args.tp3_csv.exists():
        stats_tp3 = load_tp3_runtime_stats_if_available(args.tp3_csv)
        if stats_tp3:
            print(f"Datos TP3 cargados desde {args.tp3_csv}")

    plot_runtime_vs_n(
        tp4_stats=stats_tp4,
        tp3_stats=stats_tp3,
        output_figure_path=Path(args.out_figure),
        log_y=args.log_y,
    )


if __name__ == "__main__":
    main()
