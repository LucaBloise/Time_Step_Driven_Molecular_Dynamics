#!/usr/bin/env python3
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


def resolve_tp3_root(script_dir: Path) -> Path:
    candidates = [
        script_dir.parent.parent / "TP3" / "Event-Driven-Molecular-Dynamics",
        script_dir.parent.parent / "Event-Driven-Molecular-Dynamics",
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


def read_tp4_near_obstacle_csv(csv_path: Path) -> dict[int, tuple[float, float]]:
    data: dict[int, tuple[float, float]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n_particles"])
            jin = float(row["jin_mean"])
            jin_std = float(row.get("jin_std", 0.0))
            data[n] = (jin, jin_std)

    if not data:
        raise ValueError(f"No hay datos TP4 en {csv_path}")

    return data


def read_tp3_raw_radial_csv_at_layer(
    csv_path: Path,
    s_min: float,
    s_max: float,
) -> dict[int, tuple[float, float]]:
    rows_by_run: dict[tuple[int, int, int], list[dict]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n_particles"])
            rep = int(row["repetition"])
            seed = int(row["seed"])
            rows_by_run.setdefault((n, rep, seed), []).append(row)

    jin_by_n: dict[int, list[float]] = {}

    for (n, rep, seed), rows in rows_by_run.items():
        selected = [
            row for row in rows
            if s_min <= float(row["s_center_m"]) <= s_max
        ]

        if not selected:
            continue

        # Para S=[2.0,2.2] normalmente hay una sola capa, centro S=2.1.
        jin_values = []

        for row in selected:
            frame_count = int(row["frame_count"])
            count = int(row["count_fresh_inward"])
            shell_area = float(row["shell_area_m2"])
            sum_vr = float(row["sum_vr_fresh_inward_m_s"])

            if frame_count <= 0 or shell_area <= 0.0 or count <= 0:
                continue

            rho = count / (shell_area * frame_count)
            v = sum_vr / count
            jin = rho * abs(v)
            jin_values.append(jin)

        if not jin_values:
            continue

        jin_run = statistics.fmean(jin_values)
        jin_by_n.setdefault(n, []).append(jin_run)

    output: dict[int, tuple[float, float]] = {}

    for n, values in jin_by_n.items():
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        output[n] = (mean, std)

    if not output:
        raise ValueError(f"No se pudo calcular Jin TP3 desde {csv_path}")

    return output


def plot_jin_comparison(
    tp3_data: dict[int, tuple[float, float]],
    tp4_data: dict[int, tuple[float, float]],
    figure_path: Path,
) -> None:
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

    tp3_ns = sorted(tp3_data.keys())
    tp3_means = [tp3_data[n][0] for n in tp3_ns]
    tp3_stds = [tp3_data[n][1] for n in tp3_ns]

    tp4_ns = sorted(tp4_data.keys())
    tp4_means = [tp4_data[n][0] for n in tp4_ns]
    tp4_stds = [tp4_data[n][1] for n in tp4_ns]

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
    ax.set_ylabel(r"$J_{in}$ en $S \in [2.0,2.2]$ m")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    try:
        all_ns = sorted(set(tp3_ns + tp4_ns))
        min_n = min(all_ns)
        max_n = max(all_ns)
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


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    tp4_root = script_dir.parent
    tp3_root = resolve_tp3_root(script_dir)

    parser = argparse.ArgumentParser(description="Comparar Jin TP3 vs TP4 en S=[2.0,2.2]")

    parser.add_argument(
        "--tp3-script",
        type=Path,
        default=tp3_root / "visualization" / "radial_profiles_v2.py",
    )
    parser.add_argument(
        "--tp3-raw-csv",
        type=Path,
        default=script_dir / "tp3_radial_profiles_runs.csv",
    )
    parser.add_argument(
        "--tp4-script",
        type=Path,
        default=script_dir / "radial_profiles.py",
    )
    parser.add_argument(
        "--tp4-near-csv",
        type=Path,
        default=tp4_root / "outputs" / "radial_profiles_tp4" / "near_obstacle_vs_n.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=script_dir / "jin_tp4_vs_tp3.png",
    )

    parser.add_argument(
        "--n-values",
        type=str,
        default="100,150,200,250,300,350,400,450,500,550,600,650,700,750",
    )
    parser.add_argument("--repetitions", type=str, default="5")
    parser.add_argument("--tp3-tf", type=str, default="1500")
    parser.add_argument("--tp4-tf", type=str, default="1500")
    parser.add_argument("--s-min", type=float, default=2.0)
    parser.add_argument("--s-max", type=float, default=2.2)
    parser.add_argument("--force-run", action="store_true")

    args = parser.parse_args()

    if args.force_run:
        for path in [args.tp3_raw_csv, args.tp4_near_csv, args.figure]:
            if path.exists():
                path.unlink()

    run_if_missing(
        script=args.tp3_script,
        output_csv=args.tp3_raw_csv,
        extra_args=[
            "--n-values",
            args.n_values,
            "--repetitions",
            args.repetitions,
            "--tf",
            args.tp3_tf,
            "--results-csv",
            str(args.tp3_raw_csv),
        ],
        cwd=tp3_root,
    )

    run_if_missing(
        script=args.tp4_script,
        output_csv=args.tp4_near_csv,
        extra_args=[
            "--n-values",
            args.n_values,
            "--repetitions",
            args.repetitions,
            "--tf",
            args.tp4_tf,
            "--near-s-min",
            str(args.s_min),
            "--near-s-max",
            str(args.s_max),
            "--near-obstacle-csv",
            str(args.tp4_near_csv),
        ],
        cwd=tp4_root,
    )

    tp3_data = read_tp3_raw_radial_csv_at_layer(
        csv_path=args.tp3_raw_csv,
        s_min=args.s_min,
        s_max=args.s_max,
    )

    tp4_data = read_tp4_near_obstacle_csv(args.tp4_near_csv)

    plot_jin_comparison(
        tp3_data=tp3_data,
        tp4_data=tp4_data,
        figure_path=args.figure,
    )

    print(f"Comparación generada: {args.figure}")


if __name__ == "__main__":
    main()