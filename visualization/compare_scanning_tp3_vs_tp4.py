#!/usr/bin/env python3
"""Compare TP3 vs TP4 scanning-rate (J) benchmarks.

Runs TP3 scanning_rate_vs_n.py (if available) and TP4 `scanning_rate_vs_n.py`,
then produces an overlay figure.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

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


def parse_tp4_events_to_cfc(events_path: Path) -> Tuple[List[float], List[int]]:
    if not events_path.exists():
        raise FileNotFoundError(f"No se encontro archivo de eventos en {events_path}")

    times: List[float] = [0.0]
    cfc_values: List[int] = [0]
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
                t_value = float(parts[0])
                event = parts[2]
            except (ValueError, IndexError):
                continue

            if event == "FRESH_TO_USED":
                cumulative += 1
                times.append(t_value)
                cfc_values.append(cumulative)

    if len(times) < 2:
        raise ValueError(f"No se pudo reconstruir Cfc(t) desde {events_path}")

    return times, cfc_values


def read_first_run_dir_for_n(csv_path: Path, target_n: int) -> Path | None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["n_particles"]) == target_n:
                return Path(row["run_dir"])
    return None


def plot_tp4_cfc_combined(
    series_data: Sequence[Tuple[int, Sequence[float], Sequence[int], float, float]],
    output_path: Path,
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

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#1f77b4", "#ff7f0e"]

    for index, (n_value, times, cfc_values, slope, intercept) in enumerate(series_data):
        color = colors[index % len(colors)]
        x_values = [float(x) for x in times]
        y_values = [float(y) for y in cfc_values]
        fit_values = [intercept + slope * x for x in x_values]

        ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=1.8,
            label=f"N={n_value}",
        )
        ax.plot(
            x_values,
            fit_values,
            color=color,
            linewidth=2.0,
            linestyle="--",
            label="_nolegend_",
        )

    # Use fixed X-axis ticks from 0 to 1500 every 250.
    fixed_ticks = list(range(0, 1501, 250))
    ax.set_xlim(0.0, 1500.0)
    ax.set_xticks(fixed_ticks)

    ax.set_xlabel("Tiempo t (s)")
    ax.set_ylabel(r"$C_{fc}(t)$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Figura de Cfc(t) guardada en {output_path}")


def build_tp4_cfc_figures(
    tp4_csv: Path,
    tp4_root: Path,
    output_dir: Path,
) -> None:
    series_data: List[Tuple[int, List[float], List[int], float, float]] = []

    for n_value in (100, 700):
        tp4_run_dir = read_first_run_dir_for_n(tp4_csv, n_value)

        if tp4_run_dir is None:
            print(f"Advertencia: no hay corrida representativa TP4 para N={n_value} en el CSV.")
            continue

        tp4_events = tp4_run_dir / "events.txt"

        if not tp4_events.exists():
            fallback_tp4 = tp4_root / "outputs" / "scanning_rate_tp4" / f"scanning_n{n_value}_rep1" / "events.txt"
            if fallback_tp4.exists():
                tp4_events = fallback_tp4

        if not tp4_events.exists():
            print(f"Advertencia: no se encontro events.txt de TP4 para N={n_value}.")
            continue

        times_tp4, cfc_tp4 = parse_tp4_events_to_cfc(tp4_events)
        x_values = [float(x) for x in times_tp4]
        y_values = [float(y) for y in cfc_tp4]
        slope, intercept, _ = fit_line(x_values, y_values)
        series_data.append((n_value, times_tp4, cfc_tp4, slope, intercept))

    if not series_data:
        print("Advertencia: no se pudo generar figura Cfc(t) porque faltan corridas de TP4.")
        return

    plot_tp4_cfc_combined(
        series_data=series_data,
        output_path=output_dir / "cfc_tp4_compare_n100_n700.png",
    )


def read_scanning_csv(csv_path: Path) -> list[tuple[int, float]]:
    records: list[tuple[int, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append((int(row["n_particles"]), float(row["scanning_rate_j_s_inv"])))
    if not records:
        raise ValueError(f"No hay datos en {csv_path}")
    return records


def aggregate_scanning(records: list[tuple[int, float]]) -> dict[int, tuple[float, float]]:
    grouped: dict[int, list[float]] = {}
    for n_particles, j_value in records:
        grouped.setdefault(n_particles, []).append(j_value)

    stats: dict[int, tuple[float, float]] = {}
    for n_particles, values in grouped.items():
        mean_j = statistics.fmean(values)
        std_j = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[n_particles] = (mean_j, std_j)
    return stats


def plot_scanning_comparison(
    tp4_csv: Path,
    tp3_csv: Path,
    figure_path: Path,
) -> None:
    tp4_stats = aggregate_scanning(read_scanning_csv(tp4_csv))
    tp3_stats = aggregate_scanning(read_scanning_csv(tp3_csv))

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

    parser = argparse.ArgumentParser(description="Compare scanning rate TP3 vs TP4")
    parser.add_argument(
        "--tp3-script",
        type=Path,
        default=tp3_root / "visualization" / "scanning_rate_vs_n.py",
        help="Path to TP3 scanning script.",
    )
    parser.add_argument(
        "--tp3-out",
        type=Path,
        default=script_dir / "tp3_scanning_results.csv",
        help="CSV path where TP3 results are stored.",
    )
    parser.add_argument(
        "--tp4-script",
        type=Path,
        default=script_dir / "scanning_rate_vs_n.py",
        help="TP4 scanning script.",
    )
    parser.add_argument(
        "--tp4-out",
        type=Path,
        default=tp4_root / "outputs" / "scanning_rate_tp4" / "results.csv",
        help="TP4 output CSV.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=script_dir / "scanning_tp4_vs_tp3.png",
        help="Output comparison figure.",
    )
    parser.add_argument(
        "--cfc-output-dir",
        type=Path,
        default=script_dir,
        help="Directorio de salida para figuras Cfc(t) de TP4.",
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
            "1500",
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

    plot_scanning_comparison(args.tp4_out, args.tp3_out, args.figure)
    build_tp4_cfc_figures(
        tp4_csv=args.tp4_out,
        tp4_root=tp4_root,
        output_dir=args.cfc_output_dir,
    )

    print(f"Comparación generada: {args.figure}")


if __name__ == "__main__":
    main()
