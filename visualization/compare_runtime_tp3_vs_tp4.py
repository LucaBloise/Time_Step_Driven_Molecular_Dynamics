#!/usr/bin/env python3
"""Compare TP3 vs TP4 runtime benchmarks.

This script runs the TP3 runtime benchmark (if available) and TP4 `runtime_vs_n.py`,
then produces a comparison figure overlaying both results.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def run_if_missing(script: Path, output_csv: Path, extra_args: list[str], cwd: Path) -> None:
    if output_csv.exists():
        print(f"Usando CSV existente: {output_csv}")
        return
    if not script.exists():
        raise FileNotFoundError(f"No se encontro el script requerido: {script}")

    cmd = [sys.executable, str(script), *extra_args]
    print(f"Ejecutando: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def read_runtime_csv(csv_path: Path) -> list[tuple[int, float]]:
    records: list[tuple[int, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append((int(row["n_particles"]), float(row["execution_time_s"])))
    if not records:
        raise ValueError(f"No hay datos en {csv_path}")
    return records


def aggregate_runtime(records: list[tuple[int, float]]) -> dict[int, tuple[float, float]]:
    grouped: dict[int, list[float]] = {}
    for n_particles, execution_time_s in records:
        grouped.setdefault(n_particles, []).append(execution_time_s)

    stats: dict[int, tuple[float, float]] = {}
    for n_particles, values in grouped.items():
        mean_s = statistics.fmean(values)
        std_s = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[n_particles] = (mean_s, std_s)
    return stats


def plot_runtime_comparison(
    tp4_csv: Path,
    tp3_csv: Path,
    figure_path: Path,
) -> None:
    tp4_stats = aggregate_runtime(read_runtime_csv(tp4_csv))
    tp3_stats = aggregate_runtime(read_runtime_csv(tp3_csv))

    tp4_ns = sorted(tp4_stats.keys())
    tp4_means = [tp4_stats[n][0] for n in tp4_ns]
    tp4_stds = [tp4_stats[n][1] for n in tp4_ns]

    tp3_ns = sorted(tp3_stats.keys())
    tp3_means = [tp3_stats[n][0] for n in tp3_ns]
    tp3_stds = [tp3_stats[n][1] for n in tp3_ns]

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
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    print(f"Figura guardada en {figure_path}")


def resolve_tp3_root(script_dir: Path) -> Path:
    candidates = [
        script_dir.parent.parent / "TP3" / "Event-Driven-Molecular-Dynamics",
        script_dir.parent.parent.parent / "TP3" / "Event-Driven-Molecular-Dynamics",
        script_dir.parent / "Event-Driven-Molecular-Dynamics",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No se encontro el repositorio TP3. Probados: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def main():
    script_dir = Path(__file__).resolve().parent
    tp4_root = script_dir.parent
    tp3_root = resolve_tp3_root(script_dir)

    parser = argparse.ArgumentParser(description="Compare runtime TP3 vs TP4")
    parser.add_argument(
        "--tp3-script",
        type=Path,
        default=tp3_root / "visualization" / "runtime_vs_n.py",
        help="Path to TP3 runtime script.",
    )
    parser.add_argument(
        "--tp3-out",
        type=Path,
        default=script_dir / "tp3_runtime_results.csv",
        help="CSV path where TP3 results are stored.",
    )
    parser.add_argument(
        "--tp4-script",
        type=Path,
        default=script_dir / "runtime_vs_n.py",
        help="TP4 runtime script.",
    )
    parser.add_argument(
        "--tp4-out",
        type=Path,
        default=tp4_root / "outputs" / "runtime" / "results.csv",
        help="TP4 output CSV.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=script_dir / "runtime_tp4_vs_tp3.png",
        help="Output comparison figure.",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Forzar la ejecucion aunque ya existan CSVs.",
    )

    args = parser.parse_args()

    if args.force_run and args.tp3_out.exists():
        args.tp3_out.unlink()
    if args.force_run and args.tp4_out.exists():
        args.tp4_out.unlink()
    if args.force_run and args.figure.exists():
        args.figure.unlink()

    run_if_missing(
        script=args.tp3_script,
        output_csv=args.tp3_out,
        extra_args=[
            "--n-values",
            "100,150,200,250,300,350,400,450,500,550,600,650,700,750",
            "--repetitions",
            "5",
            "--tf",
            "500",
            "--results-csv",
            str(args.tp3_out),
        ],
        cwd=tp3_root,
    )

    if not args.tp4_out.exists() or args.force_run:
        tp4_cmd = [
            sys.executable,
            str(args.tp4_script),
            "--out-csv",
            str(args.tp4_out),
            "--out-figure",
            str(args.figure),
            "--tp3-csv",
            str(args.tp3_out),
        ]
        print(f"Ejecutando TP4 benchmark/plot: {args.tp4_script}")
        subprocess.run(tp4_cmd, cwd=tp4_root, check=True)
    else:
        print(f"Usando CSV existente: {args.tp4_out}")

    plot_runtime_comparison(args.tp4_out, args.tp3_out, args.figure)

    print(f"Comparación generada: {args.figure}")


if __name__ == "__main__":
    main()
