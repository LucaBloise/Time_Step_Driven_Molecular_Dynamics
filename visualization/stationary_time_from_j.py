#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import matplotlib as mpl

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LocalJCurve:
    n_particles: int
    repetition: int
    times: Tuple[float, ...]
    j_values: Tuple[float, ...]


@dataclass(frozen=True)
class LocalJStats:
    n_particles: int
    times: Tuple[float, ...]
    mean_j: Tuple[float, ...]
    std_j: Tuple[float, ...]


def parse_n_values(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("Debe haber al menos un valor de N")
    return values


def read_fresh_to_used_times(events_path: Path) -> List[float]:
    if not events_path.exists():
        raise FileNotFoundError(f"No se encontró {events_path}")

    times: List[float] = []

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
                times.append(t)

    return sorted(times)


def count_events_until(event_times: Sequence[float], t: float) -> int:
    # Búsqueda binaria simple
    lo = 0
    hi = len(event_times)

    while lo < hi:
        mid = (lo + hi) // 2
        if event_times[mid] <= t:
            lo = mid + 1
        else:
            hi = mid

    return lo


def compute_local_j_curve(
    event_times: Sequence[float],
    tf: float,
    window: float,
    step: float,
) -> Tuple[List[float], List[float]]:
    times: List[float] = []
    j_values: List[float] = []

    t = 0.0
    while t + window <= tf + 1.0e-12:
        c0 = count_events_until(event_times, t)
        c1 = count_events_until(event_times, t + window)

        j = (c1 - c0) / window

        times.append(t + 0.5 * window)
        j_values.append(j)

        t += step

    return times, j_values


def collect_curves(
    outputs_root: Path,
    run_prefix: str,
    n_values: Sequence[int],
    repetitions: int,
    tf: float,
    window: float,
    step: float,
) -> List[LocalJCurve]:
    curves: List[LocalJCurve] = []

    run_prefix = run_prefix.strip()

    for n_particles in n_values:
        for repetition in range(1, repetitions + 1):
            if run_prefix:
                run_dir = outputs_root / f"{run_prefix}_n{n_particles}_rep{repetition}"
            else:
                run_dir = outputs_root / f"n{n_particles}_rep{repetition}"

            events_path = run_dir / "events.txt"
            event_times = read_fresh_to_used_times(events_path)

            times, j_values = compute_local_j_curve(
                event_times=event_times,
                tf=tf,
                window=window,
                step=step,
            )

            curves.append(
                LocalJCurve(
                    n_particles=n_particles,
                    repetition=repetition,
                    times=tuple(times),
                    j_values=tuple(j_values),
                )
            )

            print(
                f"[OK] N={n_particles} rep={repetition} "
                f"eventos={len(event_times)} ventanas={len(times)}"
            )

    return curves


def aggregate_curves(curves: Sequence[LocalJCurve]) -> List[LocalJStats]:
    grouped: Dict[int, List[LocalJCurve]] = {}

    for curve in curves:
        grouped.setdefault(curve.n_particles, []).append(curve)

    stats: List[LocalJStats] = []

    for n_particles in sorted(grouped.keys()):
        group = grouped[n_particles]
        reference_times = group[0].times
        n_points = len(reference_times)

        for curve in group:
            if curve.times != reference_times:
                raise ValueError(f"Tiempos inconsistentes para N={n_particles}")

        mean_j: List[float] = []
        std_j: List[float] = []

        for i in range(n_points):
            values = [curve.j_values[i] for curve in group]
            mean_j.append(statistics.fmean(values))
            std_j.append(statistics.stdev(values) if len(values) > 1 else 0.0)

        stats.append(
            LocalJStats(
                n_particles=n_particles,
                times=reference_times,
                mean_j=tuple(mean_j),
                std_j=tuple(std_j),
            )
        )

    return stats


def estimate_stationary_time(
    times: Sequence[float],
    values: Sequence[float],
    tolerance_fraction: float,
    min_consecutive_windows: int,
) -> float | None:
    if len(times) < min_consecutive_windows + 2:
        return None

    # Usamos como referencia el promedio del último 25% de la curva.
    tail_start = int(0.75 * len(values))
    tail_values = values[tail_start:]

    if not tail_values:
        return None

    reference = statistics.fmean(tail_values)

    if abs(reference) < 1.0e-15:
        return None

    consecutive = 0

    for i, value in enumerate(values):
        relative_error = abs(value - reference) / abs(reference)

        if relative_error <= tolerance_fraction:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= min_consecutive_windows:
            start_index = i - min_consecutive_windows + 1
            return times[start_index]

    return None


def write_stationary_summary(
    stats: Sequence[LocalJStats],
    output_csv: Path,
    tolerance_fraction: float,
    min_consecutive_windows: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n_particles",
                "stationary_time",
                "tolerance_fraction",
                "min_consecutive_windows",
                "reference_rule",
            ]
        )

        for stat in stats:
            t_est = estimate_stationary_time(
                times=stat.times,
                values=stat.mean_j,
                tolerance_fraction=tolerance_fraction,
                min_consecutive_windows=min_consecutive_windows,
            )

            writer.writerow(
                [
                    stat.n_particles,
                    "" if t_est is None else f"{t_est:.10f}",
                    tolerance_fraction,
                    min_consecutive_windows,
                    "mean of final 25 percent of J(t)",
                ]
            )


def plot_local_j_curves(
    stats,
    output_path,
    tolerance_fraction,
    min_consecutive_windows,
    manual_stationary_time=None,
):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 17,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    cmap = plt.colormaps["viridis"]
    n_values = [s.n_particles for s in stats]
    n_min = min(n_values)
    n_max = max(n_values)

    def color_for_n(n: int):
        if n_max == n_min:
            return cmap(0.5)
        return cmap((n - n_min) / (n_max - n_min))

    stationary_times: List[float] = []

    for stat in stats:
        color = color_for_n(stat.n_particles)
        times = list(stat.times)
        mean_j = list(stat.mean_j)
        std_j = list(stat.std_j)

        ax.plot(
            times,
            mean_j,
            linewidth=2.0,
            color=color,
        )

        t_est = estimate_stationary_time(
            times=times,
            values=mean_j,
            tolerance_fraction=tolerance_fraction,
            min_consecutive_windows=min_consecutive_windows,
        )

        if t_est is not None:
            stationary_times.append(t_est)

    if manual_stationary_time is not None:
        ax.axvline(
            manual_stationary_time,
            color="black",
            linestyle="--",
            linewidth=2.0,
        )

    norm = mpl.colors.Normalize(vmin=n_min, vmax=n_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("N", fontsize=24)

    ax.set_xlabel("Tiempo t (s)", fontsize=24)
    ax.set_ylabel(r"$J(t)$ local $(s^{-1})$", fontsize=24)
    ax.grid(True, alpha=0.25)

#    ax.set_title(
#        rf"Tasa local de escaneo: ventana $\Delta T$; "
#        rf"criterio {100*tolerance_fraction:.0f}%"
#    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Estimar tiempo estacionario a partir de J(t) local."
    )

    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=repo_root / "outputs" / "sweep_n_100_1000",
        help="Carpeta con corridas nN_repR.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="",
        help="Prefijo de corridas. Vacío para n100_rep1.",
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="100,200,300,400,500,600,700,800,900,1000",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--tf", type=float, default=2000.0)

    parser.add_argument(
        "--window",
        type=float,
        default=100.0,
        help="Ancho temporal de ventana para calcular J(t).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=25.0,
        help="Paso entre ventanas sucesivas.",
    )
    parser.add_argument(
        "--tolerance-fraction",
        type=float,
        default=0.15,
        help="Tolerancia relativa respecto al promedio final.",
    )
    parser.add_argument(
        "--min-consecutive-windows",
        type=int,
        default=4,
        help="Cantidad de ventanas consecutivas dentro de tolerancia.",
    )

    parser.add_argument(
        "--out-figure",
        type=Path,
        default=repo_root / "outputs" / "stationary_time_from_j.png",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=repo_root / "outputs" / "stationary_time_from_j.csv",
    )

    parser.add_argument(
        "--stationary-time",
        type=float,
        default=None,
        help="Tiempo estacionario manual. Si se define, se dibuja esta línea.",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n_values = parse_n_values(args.n_values)

    curves = collect_curves(
        outputs_root=args.outputs_root,
        run_prefix=args.run_prefix,
        n_values=n_values,
        repetitions=args.repetitions,
        tf=args.tf,
        window=args.window,
        step=args.step,
    )

    stats = aggregate_curves(curves)

    write_stationary_summary(
        stats=stats,
        output_csv=args.out_csv,
        tolerance_fraction=args.tolerance_fraction,
        min_consecutive_windows=args.min_consecutive_windows,
    )

    plot_local_j_curves(
        stats=stats,
        output_path=args.out_figure,
        tolerance_fraction=args.tolerance_fraction,
        min_consecutive_windows=args.min_consecutive_windows,
        manual_stationary_time=args.stationary_time,
    )

    print(f"Figura guardada en {args.out_figure}")
    print(f"Resumen guardado en {args.out_csv}")


if __name__ == "__main__":
    main()