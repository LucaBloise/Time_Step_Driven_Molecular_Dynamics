#!/usr/bin/env python3
"""TP4 1.3 - Perfiles radiales de partículas frescas entrantes.

Calcula:
- <rho_fin>(S)
- |<v_fin>(S)|
- J_in(S) = <rho_fin>(S) |<v_fin>(S)|

para múltiples N y realizaciones.

El análisis se hace sobre archivos de estado producidos por la simulación TP4.
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


@dataclass(frozen=True)
class RadialProfileRun:
    n_particles: int
    repetition: int
    seed: int
    run_dir: Path
    frame_count: int
    s_centers: Tuple[float, ...]
    shell_areas: Tuple[float, ...]
    counts_fresh_inward: Tuple[int, ...]
    sum_vr_fresh_inward: Tuple[float, ...]


@dataclass(frozen=True)
class RadialProfileStats:
    n_particles: int
    s_centers: Tuple[float, ...]
    rho_mean: Tuple[float, ...]
    rho_std: Tuple[float, ...]
    v_mean: Tuple[float, ...]
    v_std: Tuple[float, ...]
    jin_mean: Tuple[float, ...]
    jin_std: Tuple[float, ...]
    total_frames: int
    repetitions: int


@dataclass(frozen=True)
class NearObstacleStats:
    n_particles: int
    s_min: float
    s_max: float
    rho_mean: float
    rho_std: float
    v_abs_mean: float
    v_abs_std: float
    jin_mean: float
    jin_std: float


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


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 17,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
        }
    )


def build_radial_bins(r_min: float, r_max: float, ds: float) -> Tuple[List[float], List[float]]:
    if ds <= 0.0:
        raise ValueError("ds debe ser > 0")
    if r_min >= r_max:
        raise ValueError("r_min debe ser menor que r_max")

    n_bins = int(math.floor((r_max - r_min) / ds))
    if n_bins <= 0:
        raise ValueError("No hay bins radiales válidos")

    s_centers: List[float] = []
    shell_areas: List[float] = []

    for k in range(n_bins):
        s_inner = r_min + k * ds
        s_outer = s_inner + ds
        s_centers.append(0.5 * (s_inner + s_outer))
        shell_areas.append(math.pi * (s_outer * s_outer - s_inner * s_inner))

    return s_centers, shell_areas


def empty_accumulators(n_bins: int) -> Tuple[List[int], List[float]]:
    return [0 for _ in range(n_bins)], [0.0 for _ in range(n_bins)]

def is_fresh_state(state: str) -> bool:
    return state == "FRESH" or state == "1"

def accumulate_frame(
    particles: Sequence[Tuple[float, float, float, float, str]],
    r_min: float,
    r_max: float,
    ds: float,
    total_counts: List[int],
    total_sum_vr: List[float],
) -> None:
    for x, y, vx, vy, state in particles:
        if not is_fresh_state(state):
            continue

        radial_distance = math.hypot(x, y)
        if radial_distance < r_min or radial_distance >= r_max:
            continue

        dot = x * vx + y * vy
        if dot >= 0.0:
            continue

        vr = dot / radial_distance
        bin_index = int(math.floor((radial_distance - r_min) / ds))

        if 0 <= bin_index < len(total_counts):
            total_counts[bin_index] += 1
            total_sum_vr[bin_index] += vr


def parse_state_file_radial_profiles(
    state_path: Path,
    obstacle_radius: float,
    particle_radius: float,
    system_diameter: float,
    ds: float,
    stationary_start_time: float,
) -> Tuple[List[float], List[float], List[int], List[float], int]:
    """Lee estados de la simulación y acumula perfiles radiales.

    Soporta dos formatos frecuentes:

    1) Formato por frames:
       FRAME ... time ...
       PARTICLE id x y vx vy state ...
       END_FRAME

    2) Formato plano:
       time id x y vx vy state
    """
    if not state_path.exists():
        raise FileNotFoundError(f"No se encontró archivo de estados: {state_path}")

    r_min = obstacle_radius
    r_max = system_diameter

    s_centers, shell_areas = build_radial_bins(r_min=r_min, r_max=r_max, ds=ds)
    total_counts, total_sum_vr = empty_accumulators(len(s_centers))
    frame_count = 0

    current_frame_time: float | None = None
    current_particles: List[Tuple[float, float, float, float, str]] = []

    # Para formato plano agrupado por tiempo
    last_flat_time: float | None = None
    flat_particles: List[Tuple[float, float, float, float, str]] = []

    def flush_frame(time_value: float | None, particles: List[Tuple[float, float, float, float, str]]) -> int:
        if time_value is None:
            return 0
        if time_value < stationary_start_time:
            return 0
        if not particles:
            return 0

        accumulate_frame(
            particles=particles,
            r_min=r_min,
            r_max=r_max,
            ds=ds,
            total_counts=total_counts,
            total_sum_vr=total_sum_vr,
        )
        return 1

    with state_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split()
            kind = tokens[0]

            # Formato por frames
            if kind == "FRAME":
                if current_frame_time is not None:
                    frame_count += flush_frame(current_frame_time, current_particles)
                current_particles = []

                # Busca el primer token que se pueda interpretar como tiempo.
                # En muchos outputs: FRAME frame_index step time
                parsed_time = None
                for token in tokens[1:]:
                    try:
                        parsed_time = float(token)
                    except ValueError:
                        continue
                if parsed_time is None:
                    raise ValueError(f"No se pudo leer tiempo de FRAME en {state_path}:{line_number}")

                current_frame_time = parsed_time
                continue

            if kind == "PARTICLE":
                if len(tokens) < 7:
                    raise ValueError(f"Línea PARTICLE inválida en {state_path}:{line_number}")

                # PARTICLE id x y vx vy state ...
                x = float(tokens[2])
                y = float(tokens[3])
                vx = float(tokens[4])
                vy = float(tokens[5])
                state = tokens[6]
                current_particles.append((x, y, vx, vy, state))
                continue

            if kind == "END_FRAME":
                frame_count += flush_frame(current_frame_time, current_particles)
                current_frame_time = None
                current_particles = []
                continue

            # Formato plano:
            # time id x y vx vy state
            try:
                time_value = float(tokens[0])
            except ValueError:
                continue

            if len(tokens) < 7:
                continue

            try:
                x = float(tokens[2])
                y = float(tokens[3])
                vx = float(tokens[4])
                vy = float(tokens[5])
                state = tokens[6]
            except ValueError:
                continue

            if last_flat_time is None:
                last_flat_time = time_value

            if abs(time_value - last_flat_time) > 1.0e-12:
                frame_count += flush_frame(last_flat_time, flat_particles)
                flat_particles = []
                last_flat_time = time_value

            flat_particles.append((x, y, vx, vy, state))

    if current_frame_time is not None:
        frame_count += flush_frame(current_frame_time, current_particles)

    if last_flat_time is not None and flat_particles:
        frame_count += flush_frame(last_flat_time, flat_particles)

    if frame_count <= 0:
        raise ValueError(
            f"No se encontraron frames válidos en {state_path} para t >= {stationary_start_time}"
        )

    return s_centers, shell_areas, total_counts, total_sum_vr, frame_count


def run_single_simulation_tp4_states(
    repo_root: Path,
    run_dir: Path,
    java_cmd: str,
    n_particles: int,
    tf_seconds: float,
    dt_seconds: float,
    dt2_seconds: float,
    seed: int,
    state_filename: str,
) -> Path:
    if not os.path.exists(java_cmd):
        raise FileNotFoundError(f"Java executable not found: {java_cmd}")

    sim_dir = repo_root / "simulation"
    run_dir.mkdir(parents=True, exist_ok=True)

    state_path = run_dir / state_filename
    events_path = run_dir / "events.txt"
    properties_path = run_dir / "properties.txt"

    # Ajustar estos flags si tu main Java usa otros nombres para salida de estados.
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
        "--dt2",
        str(dt2_seconds),
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
            f"Falló simulación TP4 N={n_particles}, seed={seed}, run_dir={run_dir}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    if not state_path.exists():
        raise FileNotFoundError(
            f"No se encontró archivo de estados {state_path}. "
            "Revisar si el simulador Java usa otro flag distinto de --out."
        )

    return state_path


def collect_radial_runs(
    repo_root: Path,
    java_cmd: str,
    n_values: Sequence[int],
    repetitions: int,
    tf_seconds: float,
    dt_seconds: float,
    dt2_seconds: float,
    seed_base: int,
    outputs_root: Path,
    run_prefix: str,
    reuse_existing_runs: bool,
    state_filename: str,
    obstacle_radius: float,
    particle_radius: float,
    system_diameter: float,
    ds: float,
    stationary_start_time: float,
) -> List[RadialProfileRun]:
    outputs_root.mkdir(parents=True, exist_ok=True)

    runs: List[RadialProfileRun] = []

    for n_particles in n_values:
        for repetition in range(1, repetitions + 1):
            seed = seed_base + n_particles * 1000 + repetition
            run_dir = outputs_root / f"{run_prefix}_n{n_particles}_rep{repetition}"

            if reuse_existing_runs:
                state_path = run_dir / state_filename
                if not state_path.exists():
                    raise FileNotFoundError(f"No se encontró corrida existente: {state_path}")
            else:
                state_path = run_single_simulation_tp4_states(
                    repo_root=repo_root,
                    java_cmd=java_cmd,
                    run_dir=run_dir,
                    n_particles=n_particles,
                    tf_seconds=tf_seconds,
                    dt_seconds=dt_seconds,
                    dt2_seconds=dt2_seconds,
                    seed=seed,
                    state_filename=state_filename,
                )

            s_centers, shell_areas, counts, sum_vr, frame_count = parse_state_file_radial_profiles(
                state_path=state_path,
                obstacle_radius=obstacle_radius,
                particle_radius=particle_radius,
                system_diameter=system_diameter,
                ds=ds,
                stationary_start_time=stationary_start_time,
            )

            runs.append(
                RadialProfileRun(
                    n_particles=n_particles,
                    repetition=repetition,
                    seed=seed,
                    run_dir=run_dir,
                    frame_count=frame_count,
                    s_centers=tuple(s_centers),
                    shell_areas=tuple(shell_areas),
                    counts_fresh_inward=tuple(counts),
                    sum_vr_fresh_inward=tuple(sum_vr),
                )
            )

            print(
                f"[OK] N={n_particles:4d} rep={repetition:2d} seed={seed} "
                f"frames={frame_count}"
            )

    return runs


def aggregate_radial_stats(runs: Sequence[RadialProfileRun]) -> List[RadialProfileStats]:
    grouped: Dict[int, List[RadialProfileRun]] = {}
    for run in runs:
        grouped.setdefault(run.n_particles, []).append(run)

    stats: List[RadialProfileStats] = []

    for n_particles in sorted(grouped.keys()):
        run_group = grouped[n_particles]
        first = run_group[0]
        n_bins = len(first.s_centers)

        rho_per_run: List[List[float]] = []
        v_per_run: List[List[float]] = []
        jin_per_run: List[List[float]] = []
        total_frames = 0

        for run in run_group:
            if run.s_centers != first.s_centers:
                raise ValueError(f"Bins inconsistentes para N={n_particles}")
            if run.shell_areas != first.shell_areas:
                raise ValueError(f"Áreas inconsistentes para N={n_particles}")

            total_frames += run.frame_count

            rho_run: List[float] = []
            v_run: List[float] = []
            jin_run: List[float] = []

            for k in range(n_bins):
                rho_k = run.counts_fresh_inward[k] / (run.shell_areas[k] * run.frame_count)
                v_k = (
                    run.sum_vr_fresh_inward[k] / run.counts_fresh_inward[k]
                    if run.counts_fresh_inward[k] > 0
                    else 0.0
                )
                jin_k = rho_k * abs(v_k)

                rho_run.append(rho_k)
                v_run.append(v_k)
                jin_run.append(jin_k)

            rho_per_run.append(rho_run)
            v_per_run.append(v_run)
            jin_per_run.append(jin_run)

        repetitions = len(run_group)

        def mean_and_std(values: Sequence[float]) -> Tuple[float, float]:
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            return mean, std

        rho_mean: List[float] = []
        rho_std: List[float] = []
        v_mean: List[float] = []
        v_std: List[float] = []
        jin_mean: List[float] = []
        jin_std: List[float] = []

        for k in range(n_bins):
            rho_values = [rho_per_run[r][k] for r in range(repetitions)]
            v_values = [v_per_run[r][k] for r in range(repetitions)]
            jin_values = [jin_per_run[r][k] for r in range(repetitions)]

            rho_m, rho_s = mean_and_std(rho_values)
            v_m, v_s = mean_and_std(v_values)
            jin_m, jin_s = mean_and_std(jin_values)

            rho_mean.append(rho_m)
            rho_std.append(rho_s)
            v_mean.append(v_m)
            v_std.append(v_s)
            jin_mean.append(jin_m)
            jin_std.append(jin_s)

        stats.append(
            RadialProfileStats(
                n_particles=n_particles,
                s_centers=first.s_centers,
                rho_mean=tuple(rho_mean),
                rho_std=tuple(rho_std),
                v_mean=tuple(v_mean),
                v_std=tuple(v_std),
                jin_mean=tuple(jin_mean),
                jin_std=tuple(jin_std),
                total_frames=total_frames,
                repetitions=repetitions,
            )
        )

    return stats


def write_radial_runs_csv(runs: Sequence[RadialProfileRun], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n_particles",
                "repetition",
                "seed",
                "run_dir",
                "frame_count",
                "bin_index",
                "s_center_m",
                "shell_area_m2",
                "count_fresh_inward",
                "sum_vr_fresh_inward_m_s",
            ]
        )

        for run in runs:
            for k, s_value in enumerate(run.s_centers):
                writer.writerow(
                    [
                        run.n_particles,
                        run.repetition,
                        run.seed,
                        str(run.run_dir),
                        run.frame_count,
                        k,
                        f"{s_value:.10f}",
                        f"{run.shell_areas[k]:.10f}",
                        run.counts_fresh_inward[k],
                        f"{run.sum_vr_fresh_inward[k]:.10f}",
                    ]
                )


def read_radial_runs_csv(input_csv: Path) -> List[RadialProfileRun]:
    grouped: Dict[Tuple[int, int, int, str, int], List[dict]] = {}

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (
                int(row["n_particles"]),
                int(row["repetition"]),
                int(row["seed"]),
                row["run_dir"],
                int(row["frame_count"]),
            )
            grouped.setdefault(key, []).append(row)

    runs: List[RadialProfileRun] = []

    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["bin_index"]))
        n_particles, repetition, seed, run_dir, frame_count = key

        runs.append(
            RadialProfileRun(
                n_particles=n_particles,
                repetition=repetition,
                seed=seed,
                run_dir=Path(run_dir),
                frame_count=frame_count,
                s_centers=tuple(float(row["s_center_m"]) for row in rows),
                shell_areas=tuple(float(row["shell_area_m2"]) for row in rows),
                counts_fresh_inward=tuple(int(row["count_fresh_inward"]) for row in rows),
                sum_vr_fresh_inward=tuple(float(row["sum_vr_fresh_inward_m_s"]) for row in rows),
            )
        )

    runs.sort(key=lambda run: (run.n_particles, run.repetition))
    return runs


def filter_positive_curve(
    x: Sequence[float],
    y: Sequence[float],
    y_std: Sequence[float],
    eps: float = 1.0e-15,
) -> Tuple[List[float], List[float], List[float]]:
    fx: List[float] = []
    fy: List[float] = []
    fs: List[float] = []

    for x_value, y_value, std_value in zip(x, y, y_std):
        if abs(y_value) > eps:
            fx.append(x_value)
            fy.append(y_value)
            fs.append(std_value)

    return fx, fy, fs


def plot_profiles_with_colorbar(
    stats: Sequence[RadialProfileStats],
    output_path: Path,
) -> None:
    configure_plot_style()

    ns = [entry.n_particles for entry in stats]
    norm = Normalize(vmin=min(ns), vmax=max(ns))
    cmap = plt.colormaps["viridis"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 14), sharex=True)

    for stat in stats:
        color = cmap(norm(stat.n_particles))
        s = list(stat.s_centers)

        rho = list(stat.rho_mean)
        rho_std = list(stat.rho_std)

        v_abs = [abs(v) for v in stat.v_mean]
        v_std = list(stat.v_std)

        jin = list(stat.jin_mean)
        jin_std = list(stat.jin_std)

        s_rho, rho_f, rho_s = filter_positive_curve(s, rho, rho_std)
        if s_rho:
            axes[0].plot(s_rho, rho_f, color=color, linewidth=1.8)
            axes[0].fill_between(
                s_rho,
                [max(0.0, m - sd) for m, sd in zip(rho_f, rho_s)],
                [m + sd for m, sd in zip(rho_f, rho_s)],
                color=color,
                alpha=0.15,
            )

        s_v, v_f, v_s = filter_positive_curve(s, v_abs, v_std)
        if s_v:
            axes[1].plot(s_v, v_f, color=color, linewidth=1.8)
            axes[1].fill_between(
                s_v,
                [max(0.0, m - sd) for m, sd in zip(v_f, v_s)],
                [m + sd for m, sd in zip(v_f, v_s)],
                color=color,
                alpha=0.15,
            )

        s_j, j_f, j_s = filter_positive_curve(s, jin, jin_std)
        if s_j:
            axes[2].plot(s_j, j_f, color=color, linewidth=1.8)
            axes[2].fill_between(
                s_j,
                [max(0.0, m - sd) for m, sd in zip(j_f, j_s)],
                [m + sd for m, sd in zip(j_f, j_s)],
                color=color,
                alpha=0.15,
            )

    axes[0].set_ylabel(r"$\langle \rho_{fin} \rangle(S)$")
    axes[1].set_ylabel(r"$|\langle v_{fin} \rangle(S)|$")
    axes[2].set_ylabel(r"$J_{in}(S)$")
    axes[2].set_xlabel("S (m)")

    for ax in axes:
        ax.grid(True, alpha=0.25)

        sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig.subplots_adjust(
        left=0.14,
        right=0.82,
        top=0.95,
        bottom=0.08,
        hspace=0.18,
    )

    cbar = fig.colorbar(
        sm,
        ax=axes,
        fraction=0.035,
        pad=0.04,
        aspect=35,
    )
    cbar.set_label("N")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_jin_zoom(
    stats: Sequence[RadialProfileStats],
    output_path: Path,
    s_min: float,
    s_max: float,
) -> None:
    configure_plot_style()

    ns = [entry.n_particles for entry in stats]
    norm = Normalize(vmin=min(ns), vmax=max(ns))
    cmap = plt.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for stat in stats:
        color = cmap(norm(stat.n_particles))
        s = list(stat.s_centers)
        jin = list(stat.jin_mean)
        jin_std = list(stat.jin_std)

        filtered = [
            (s_value, j_value, j_std)
            for s_value, j_value, j_std in zip(s, jin, jin_std)
            if s_min <= s_value <= s_max and abs(j_value) > 1.0e-15
        ]

        if not filtered:
            continue

        sf = [x[0] for x in filtered]
        jf = [x[1] for x in filtered]
        js = [x[2] for x in filtered]

        ax.plot(sf, jf, color=color, linewidth=2.0)
        ax.fill_between(
            sf,
            [max(0.0, m - sd) for m, sd in zip(jf, js)],
            [m + sd for m, sd in zip(jf, js)],
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel("S (m)")
    ax.set_ylabel(r"$J_{in}(S)$")
    ax.set_xlim(s_min, s_max)
    ax.grid(True, alpha=0.25)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("N")

    ax.set_title(rf"Detalle de $J_{{in}}(S)$ en $S\in[{s_min},{s_max}]$ m")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def average_near_obstacle(
    stats: Sequence[RadialProfileStats],
    s_min: float,
    s_max: float,
) -> List[NearObstacleStats]:
    output: List[NearObstacleStats] = []

    for stat in stats:
        indices = [
            i for i, s_value in enumerate(stat.s_centers)
            if s_min <= s_value <= s_max
        ]

        if not indices:
            continue

        def avg(values: Sequence[float]) -> float:
            selected = [values[i] for i in indices if abs(values[i]) > 1.0e-15]
            if not selected:
                return 0.0
            return statistics.fmean(selected)

        output.append(
            NearObstacleStats(
                n_particles=stat.n_particles,
                s_min=s_min,
                s_max=s_max,
                rho_mean=avg(stat.rho_mean),
                rho_std=avg(stat.rho_std),
                v_abs_mean=avg([abs(v) for v in stat.v_mean]),
                v_abs_std=avg(stat.v_std),
                jin_mean=avg(stat.jin_mean),
                jin_std=avg(stat.jin_std),
            )
        )

    return output

def plot_near_obstacle_vs_n(
    near_stats: Sequence[NearObstacleStats],
    output_path: Path,
) -> None:
    configure_plot_style()

    ns = [entry.n_particles for entry in near_stats]

    rho = [entry.rho_mean for entry in near_stats]
    rho_std = [entry.rho_std for entry in near_stats]

    v_abs = [entry.v_abs_mean for entry in near_stats]
    v_std = [entry.v_abs_std for entry in near_stats]

    jin = [entry.jin_mean for entry in near_stats]
    jin_std = [entry.jin_std for entry in near_stats]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    ax_top_rho = ax_top
    ax_top_jin = ax_top.twinx()

    rho_plot = ax_top_rho.errorbar(
        ns,
        rho,
        yerr=rho_std,
        fmt="o-",
        capsize=5,
        label=r"$\langle \rho_{fin} \rangle$",
    )

    jin_plot = ax_top_jin.errorbar(
        ns,
        jin,
        yerr=jin_std,
        fmt="s-",
        capsize=5,
        label=r"$J_{in}$",
    )

    ax_top_rho.set_ylabel(r"$\langle \rho_{fin} \rangle$")
    ax_top_jin.set_ylabel(r"$J_{in}$")

    ax_top_rho.grid(True, alpha=0.25)

    # leyenda combinada
    handles = [rho_plot, jin_plot]
    labels = [h.get_label() for h in handles]
    ax_top_rho.legend(handles, labels, loc="upper left")

    ax_bottom.errorbar(
        ns,
        v_abs,
        yerr=v_std,
        fmt="^-",
        capsize=5,
        label=r"$|\langle v_{fin} \rangle|$",
    )

    ax_bottom.set_ylabel(r"$|\langle v_{fin} \rangle|$")
    ax_bottom.set_xlabel("N")
    ax_bottom.grid(True, alpha=0.25)

    if near_stats:
        fig.suptitle(
            rf"Promedio cercano al obstáculo: "
            rf"$S\in[{near_stats[0].s_min},{near_stats[0].s_max}]$ m",
            y=0.98,
        )

    fig.subplots_adjust(
        left=0.12,
        right=0.88,
        top=0.93,
        bottom=0.08,
        hspace=0.18,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def write_near_obstacle_csv(near_stats: Sequence[NearObstacleStats], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n_particles",
                "s_min",
                "s_max",
                "rho_mean",
                "rho_std",
                "v_abs_mean",
                "v_abs_std",
                "jin_mean",
                "jin_std",
            ]
        )

        for entry in near_stats:
            writer.writerow(
                [
                    entry.n_particles,
                    f"{entry.s_min:.10f}",
                    f"{entry.s_max:.10f}",
                    f"{entry.rho_mean:.10e}",
                    f"{entry.rho_std:.10e}",
                    f"{entry.v_abs_mean:.10e}",
                    f"{entry.v_abs_std:.10e}",
                    f"{entry.jin_mean:.10e}",
                    f"{entry.jin_std:.10e}",
                ]
            )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="TP4 1.3: perfiles radiales de partículas frescas.")

    parser.add_argument(
        "--n-values",
        type=str,
        default="100,150,200,250,300,350,400,450,500,550,600,650,700,750",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--tf", type=float, default=1500.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--dt2", type=float, default=0.1)
    parser.add_argument("--seed-base", type=int, default=600000)

    parser.add_argument("--l", type=float, default=80.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--particle-radius", type=float, default=1.0)
    parser.add_argument("--ds", type=float, default=0.2)

    parser.add_argument("--stationary-start", type=float, default=0.0)

    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=repo_root / "outputs" / "radial_profiles_tp4",
    )
    parser.add_argument("--run-prefix", type=str, default="radial")
    parser.add_argument("--state-filename", type=str, default="states.txt")
    parser.add_argument("--reuse-existing-runs", action="store_true")

    parser.add_argument(
        "--java-cmd",
        type=str,
        default="C:/Program Files/JetBrains/IntelliJ IDEA 2025.3.3/jbr/bin/java.exe",
        help="Ruta al ejecutable de Java.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=repo_root / "outputs" / "radial_profiles_tp4" / "radial_runs.csv",
    )
    parser.add_argument(
        "--profiles-figure",
        type=Path,
        default=repo_root / "outputs" / "radial_profiles_tp4" / "radial_profiles_all_n.png",
    )
    parser.add_argument(
        "--jin-zoom-figure",
        type=Path,
        default=repo_root / "outputs" / "radial_profiles_tp4" / "jin_zoom_obstacle.png",
    )
    parser.add_argument(
        "--near-obstacle-figure",
        type=Path,
        default=repo_root / "outputs" / "radial_profiles_tp4" / "near_obstacle_vs_n.png",
    )
    parser.add_argument(
        "--near-obstacle-csv",
        type=Path,
        default=repo_root / "outputs" / "radial_profiles_tp4" / "near_obstacle_vs_n.csv",
    )

    parser.add_argument("--zoom-s-min", type=float, default=1.5)
    parser.add_argument("--zoom-s-max", type=float, default=5.0)
    parser.add_argument("--near-s-min", type=float, default=2.0)
    parser.add_argument("--near-s-max", type=float, default=2.2)

    parser.add_argument("--only-plot", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    n_values = parse_n_values(args.n_values)

    if args.only_plot:
        runs = read_radial_runs_csv(args.out_csv)
        print(f"Modo only-plot: {len(runs)} realizaciones leídas desde {args.out_csv}")
    else:
        runs = collect_radial_runs(
            repo_root=repo_root,
            java_cmd=args.java_cmd,
            n_values=n_values,
            repetitions=args.repetitions,
            tf_seconds=args.tf,
            dt_seconds=args.dt,
            dt2_seconds=args.dt2,
            seed_base=args.seed_base,
            outputs_root=args.outputs_root,
            run_prefix=args.run_prefix,
            reuse_existing_runs=args.reuse_existing_runs,
            state_filename=args.state_filename,
            obstacle_radius=args.r0,
            particle_radius=args.particle_radius,
            system_diameter=args.l,
            ds=args.ds,
            stationary_start_time=args.stationary_start,
        )
        write_radial_runs_csv(runs, args.out_csv)
        print(f"CSV radial guardado en {args.out_csv}")

    stats = aggregate_radial_stats(runs)

    plot_profiles_with_colorbar(
        stats=stats,
        output_path=args.profiles_figure,
    )
    print(f"Figura perfiles radiales guardada en {args.profiles_figure}")

    plot_jin_zoom(
        stats=stats,
        output_path=args.jin_zoom_figure,
        s_min=args.zoom_s_min,
        s_max=args.zoom_s_max,
    )
    print(f"Figura zoom Jin guardada en {args.jin_zoom_figure}")

    near_stats = average_near_obstacle(
        stats=stats,
        s_min=args.near_s_min,
        s_max=args.near_s_max,
    )

    write_near_obstacle_csv(near_stats, args.near_obstacle_csv)
    print(f"CSV cercano al obstáculo guardado en {args.near_obstacle_csv}")

    plot_near_obstacle_vs_n(
        near_stats=near_stats,
        output_path=args.near_obstacle_figure,
    )
    print(f"Figura cercana al obstáculo guardada en {args.near_obstacle_figure}")


if __name__ == "__main__":
    main()