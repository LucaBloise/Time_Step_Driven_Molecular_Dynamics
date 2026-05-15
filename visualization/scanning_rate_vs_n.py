#!/usr/bin/env python3
"""TP4 1.2 - Scanning rate J versus particle count N (Time-Step Driven Molecular Dynamics).

Wrapper script (renamed from tp4_scanning_rate_vs_n.py).
"""

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
class ScanningRecord:
    n_particles: int
    repetition: int
    seed: int
    scanning_rate_j_s_inv: float
    run_dir: Path


@dataclass(frozen=True)
class ScanningStats:
    n_particles: int
    mean_j_s_inv: float
    std_j_s_inv: float
    sample_count: int


@dataclass(frozen=True)
class DiagnosticFit:
    n_particles: int
    repetition: int
    seed: int
    run_dir: Path
    times: Tuple[float, ...]
    cfc_values: Tuple[int, ...]
    fit_slope: float
    fit_intercept: float
    fit_r2: float


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


def fit_line(
    x_values: Sequence[float], y_values: Sequence[float]
) -> Tuple[float, float, float]:
    if len(x_values) != len(y_values):
        raise ValueError("x_values e y_values deben tener la misma longitud")
    if len(x_values) < 2:
        raise ValueError("Se necesitan al menos 2 puntos para un ajuste lineal")

    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)

    var_x = 0.0
    cov_xy = 0.0
    for x_val, y_val in zip(x_values, y_values):
        dx = x_val - mean_x
        var_x += dx * dx
        cov_xy += dx * (y_val - mean_y)

    if var_x <= 1.0e-15:
        slope = 0.0
        intercept = mean_y
    else:
        slope = cov_xy / var_x
        intercept = mean_y - slope * mean_x

    ss_res = 0.0
    ss_tot = 0.0
    for x_val, y_val in zip(x_values, y_values):
        estimate = intercept + slope * x_val
        residual = y_val - estimate
        ss_res += residual * residual

        centered = y_val - mean_y
        ss_tot += centered * centered

    if ss_tot <= 1.0e-15:
        r_squared = 1.0 if ss_res <= 1.0e-15 else 0.0
    else:
        r_squared = 1.0 - (ss_res / ss_tot)

    return slope, intercept, r_squared


def parse_events_file_to_cfc(
    events_path: Path, dt: float
) -> Tuple[List[float], List[int]]:
    if not events_path.exists():
        raise FileNotFoundError(f"No se encontro archivo de eventos en {events_path}")

    events: List[Tuple[float, str]] = []

    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
                event = parts[2]
                events.append((t, event))
            except (ValueError, IndexError):
                continue

    if not events:
        raise ValueError(f"No se encontraron eventos en {events_path}")

    events.sort(key=lambda x: x[0])

    times: List[float] = []
    cfc_values: List[int] = []

    cumulative_cfc = 0
    current_dt_bin = 0.0
    event_idx = 0

    while event_idx < len(events):
        t_event, event_type = events[event_idx]

        while current_dt_bin + dt < t_event - 1e-12:
            times.append(current_dt_bin)
            cfc_values.append(cumulative_cfc)
            current_dt_bin += dt

        while (
            event_idx < len(events)
            and events[event_idx][0] <= current_dt_bin + dt + 1e-12
        ):
            _, evt = events[event_idx]
            if evt == "FRESH_TO_USED":
                cumulative_cfc += 1
            event_idx += 1

        times.append(current_dt_bin)
        cfc_values.append(cumulative_cfc)
        current_dt_bin += dt

    return times, cfc_values


def run_single_simulation_tp4(
    repo_root: Path,
    run_dir: Path,
    n_particles: int,
    tf_seconds: float,
    dt_seconds: float,
    seed: int,
) -> ScanningRecord:
    java_cmd = (
        "C:/Program Files/JetBrains/IntelliJ IDEA 2025.3.3/jbr/bin/java.exe"
    )
    if not os.path.exists(java_cmd):
        raise FileNotFoundError(f"Java executable not found: {java_cmd}")

    sim_dir = repo_root / "simulation"
    run_dir.mkdir(parents=True, exist_ok=True)

    events_path = run_dir / "events.txt"

    cmd = [
        java_cmd,
        "-cp",
        str(sim_dir),
        "ScanningRateSimulation",
        "--n",
        str(n_particles),
        "--tf",
        str(tf_seconds),
        "--dt",
        str(dt_seconds),
        "--seed",
        str(seed),
        "--no-state",
        "--events-out",
        str(events_path),
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

    if not events_path.exists():
        raise FileNotFoundError(f"No se encontro archivo de eventos en {run_dir}")

    times, cfc_values = parse_events_file_to_cfc(events_path, dt_seconds)

    if len(times) < 2:
        raise ValueError(f"Muy pocos datos de Cfc(t) para N={n_particles}")

    x_vals = [float(t) for t in times]
    y_vals = [float(c) for c in cfc_values]

    slope_j, _, r2 = fit_line(x_vals, y_vals)

    if slope_j < 0:
        print(
            f"Advertencia: pendiente negativa J={slope_j:.6e} para N={n_particles}. "
            f"r2={r2:.6f}. Usando valor absoluto."
        )
        slope_j = abs(slope_j)

    return ScanningRecord(
        n_particles=n_particles,
        repetition=0,
        seed=seed,
        scanning_rate_j_s_inv=slope_j,
        run_dir=run_dir,
    )


def collect_scanning_records_tp4(
    repo_root: Path,
    n_values: Sequence[int],
    repetitions: int,
    tf_seconds: float,
    dt_seconds: float,
    seed_base: int,
    outputs_base_dir: Path,
    run_prefix: str,
) -> List[ScanningRecord]:
    outputs_base_dir.mkdir(parents=True, exist_ok=True)

    records: List[ScanningRecord] = []
    for n_particles in n_values:
        for repetition in range(1, repetitions + 1):
            seed = seed_base + n_particles * 1000 + repetition
            run_dir = (
                outputs_base_dir / f"{run_prefix}_n{n_particles}_rep{repetition}"
            )

            record = run_single_simulation_tp4(
                repo_root=repo_root,
                run_dir=run_dir,
                n_particles=n_particles,
                tf_seconds=tf_seconds,
                dt_seconds=dt_seconds,
                seed=seed,
            )

            records.append(
                ScanningRecord(
                    n_particles=record.n_particles,
                    repetition=repetition,
                    seed=seed,
                    scanning_rate_j_s_inv=record.scanning_rate_j_s_inv,
                    run_dir=record.run_dir,
                )
            )
            print(
                f"[OK] TP4 N={n_particles:4d} rep={repetition:2d} seed={seed} "
                f"J={record.scanning_rate_j_s_inv:.6e} s^-1"
            )

    return records


def write_scanning_csv(records: Sequence[ScanningRecord], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["n_particles", "repetition", "seed", "scanning_rate_j_s_inv", "run_dir"]
        )
        for record in records:
            writer.writerow(
                [
                    record.n_particles,
                    record.repetition,
                    record.seed,
                    f"{record.scanning_rate_j_s_inv:.10e}",
                    str(record.run_dir),
                ]
            )


def read_scanning_csv(csv_path: Path) -> List[ScanningRecord]:
    records: List[ScanningRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                ScanningRecord(
                    n_particles=int(row["n_particles"]),
                    repetition=int(row["repetition"]),
                    seed=int(row["seed"]),
                    scanning_rate_j_s_inv=float(row["scanning_rate_j_s_inv"]),
                    run_dir=Path(row["run_dir"]),
                )
            )
    if not records:
        raise ValueError(f"No hay datos en {csv_path}")
    return records


def aggregate_stats(records: Sequence[ScanningRecord]) -> List[ScanningStats]:
    grouped: Dict[int, List[float]] = {}
    for record in records:
        grouped.setdefault(record.n_particles, []).append(
            record.scanning_rate_j_s_inv
        )

    stats: List[ScanningStats] = []
    for n_particles in sorted(grouped.keys()):
        values = grouped[n_particles]
        mean_j = statistics.fmean(values)
        std_j = statistics.stdev(values) if len(values) > 1 else 0.0
        stats.append(
            ScanningStats(
                n_particles=n_particles,
                mean_j_s_inv=mean_j,
                std_j_s_inv=std_j,
                sample_count=len(values),
            )
        )

    return stats


def load_tp3_scanning_stats_if_available(
    tp3_benchmark_csv: Path,
) -> List[ScanningStats] | None:
    if not tp3_benchmark_csv.exists():
        return None
    try:
        records = read_scanning_csv(tp3_benchmark_csv)
        return aggregate_stats(records)
    except Exception as e:
        print(f"Advertencia: No se pudo cargar datos TP3: {e}")
        return None


def plot_scanning_rate_vs_n(
    tp4_stats: Sequence[ScanningStats],
    tp3_stats: Sequence[ScanningStats] | None,
    output_figure_path: Path,
) -> None:
    tp4_ns = [s.n_particles for s in tp4_stats]
    tp4_means = [s.mean_j_s_inv for s in tp4_stats]
    tp4_stds = [s.std_j_s_inv for s in tp4_stats]

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
        tp3_means = [s.mean_j_s_inv for s in tp3_stats]
        tp3_stds = [s.std_j_s_inv for s in tp3_stats]
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
    ax.set_ylabel("Velocidad de escaneo J (s⁻¹)")
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
    fig.tight_layout()
    fig.savefig(output_figure_path, dpi=200)
    print(f"Figura guardada en {output_figure_path}")
    plt.close(fig)


def plot_cfc_fit(
    fit_data: DiagnosticFit,
    output_figure_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
        }
    )

    times = list(fit_data.times)
    cfc_values = list(fit_data.cfc_values)
    fit_values = [fit_data.fit_intercept + fit_data.fit_slope * t for t in times]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, cfc_values, color="#1f77b4", linewidth=1.8, label=r"$C_{fc}(t)$")
    ax.plot(
        times,
        fit_values,
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label=rf"Ajuste lineal: $J={fit_data.fit_slope:.4e}\,s^{{-1}}$",
    )

    ax.set_xlabel("Tiempo t (s)")
    ax.set_ylabel(r"$C_{fc}(t)$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax.set_title(f"N={fit_data.n_particles}")

    fig.tight_layout()
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure_path, dpi=200)
    print(f"Figura de Cfc(t) guardada en {output_figure_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="TP4 1.2: Scanning rate J vs N for Time-Step Driven Molecular Dynamics."
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
        default=1500.0,
        help="Tiempo final de simulacion en segundos.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Integration step in seconds.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=500000,
        help="Base para semillas reproducibles.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "scanning_rate_tp4",
        help="Carpeta raiz donde se crean benchmarks.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="scanning",
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
        default="scanning_rate_tp4_vs_tp3.png",
        help="Archivo de salida de la figura.",
    )
    parser.add_argument(
        "--tp3-csv",
        type=Path,
        default=None,
        help="CSV con datos TP3 para comparacion (opcional).",
    )
    parser.add_argument(
        "--cfc-figure",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "scanning_rate_tp4" / "cfc_fit_example.png",
        help="Figura diagnostica de Cfc(t) con ajuste lineal para una corrida representativa.",
    )
    parser.add_argument(
        "--cfc-low-n",
        type=int,
        default=100,
        help="N bajo a usar para la figura diagnostica de Cfc(t).",
    )
    parser.add_argument(
        "--cfc-high-n",
        type=int,
        default=900,
        help="N alto a usar para la figura diagnostica de Cfc(t).",
    )
    parser.add_argument(
        "--cfc-low-figure",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "scanning_rate_tp4" / "cfc_fit_n100.png",
        help="Figura diagnostica de Cfc(t) para el N bajo.",
    )
    parser.add_argument(
        "--cfc-high-figure",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "scanning_rate_tp4" / "cfc_fit_n900.png",
        help="Figura diagnostica de Cfc(t) para el N alto.",
    )

    args = parser.parse_args()

    if args.out_csv is None:
        args.out_csv = args.outputs_root / "results.csv"

    repo_root = Path(__file__).resolve().parent.parent

    n_values = parse_n_values(args.n_values)

    print(f"Benchmark dir: {args.outputs_root.resolve()}")
    print(f"Run prefix: {args.run_prefix}")
    print(f"Total corridas objetivo: {len(n_values) * args.repetitions}")
    print(f"tf={args.tf}s, dt={args.dt}s")

    records_tp4 = collect_scanning_records_tp4(
        repo_root=repo_root,
        n_values=n_values,
        repetitions=args.repetitions,
        tf_seconds=args.tf,
        dt_seconds=args.dt,
        seed_base=args.seed_base,
        outputs_base_dir=args.outputs_root,
        run_prefix=args.run_prefix,
    )

    write_scanning_csv(records_tp4, args.out_csv)
    print(f"Resultados guardados en {args.out_csv}")

    stats_tp4 = aggregate_stats(records_tp4)

    def generate_diagnostic_figure(target_n: int, output_path: Path) -> None:
        matching_records = [record for record in records_tp4 if record.n_particles == target_n]
        if not matching_records:
            print(f"Advertencia: no hay corrida TP4 para N={target_n}, no se genera {output_path}")
            return

        representative = matching_records[0]
        representative_events = representative.run_dir / "events.txt"
        times, cfc_values = parse_events_file_to_cfc(representative_events, args.dt)
        fit_slope, fit_intercept, fit_r2 = fit_line(
            [float(t) for t in times],
            [float(c) for c in cfc_values],
        )
        plot_cfc_fit(
            fit_data=DiagnosticFit(
                n_particles=representative.n_particles,
                repetition=representative.repetition,
                seed=representative.seed,
                run_dir=representative.run_dir,
                times=tuple(times),
                cfc_values=tuple(cfc_values),
                fit_slope=fit_slope,
                fit_intercept=fit_intercept,
                fit_r2=fit_r2,
            ),
            output_figure_path=output_path,
        )

    if records_tp4:
        generate_diagnostic_figure(args.cfc_low_n, Path(args.cfc_low_figure))
        generate_diagnostic_figure(args.cfc_high_n, Path(args.cfc_high_figure))

    stats_tp3 = None
    if args.tp3_csv and args.tp3_csv.exists():
        stats_tp3 = load_tp3_scanning_stats_if_available(args.tp3_csv)
        if stats_tp3:
            print(f"Datos TP3 cargados desde {args.tp3_csv}")

    plot_scanning_rate_vs_n(
        tp4_stats=stats_tp4,
        tp3_stats=stats_tp3,
        output_figure_path=Path(args.out_figure),
    )


if __name__ == "__main__":
    main()
