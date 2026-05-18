#!/usr/bin/env python3
"""Plot used-particle fraction F_u(t) from a single run output.

Input can be either:
- A run directory containing states.txt, or
- A states file directly, including legacy scan_*.txt outputs.

By definition in this simulation:
- state = 1 -> fresh
- state = 0 -> used
So F_u(t) = N_u(t) / N where N_u counts particles with state == 0 at time t.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graficar F_u(t)=N_u(t)/N para una corrida individual o multiples N en una carpeta."
    )
    parser.add_argument(
        "run",
        type=Path,
        nargs="?",
        default=None,
        help="Ruta a carpeta de corrida o archivo de estados (states.txt / scan_*.txt).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Carpeta con corridas tipo nN_repR para superponer varias curvas en un mismo grafico.",
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="",
        help="Lista de N separada por comas para usar junto con --runs-dir (ej: 100,500,1000).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("visualization") / "outputs" / "used_fraction_vs_time.png",
        help="Ruta de salida de la figura.",
    )
    parser.add_argument(
        "--vline-time",
        type=float,
        default=1400.0,
        help="Tiempo (s) de la linea vertical de referencia.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Titulo opcional de la figura.",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default=r"Tiempo $t$ (s)",
        help="Texto del eje X.",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default=r"$F_u(t)$",
        help="Texto del eje Y.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Si se especifica, no abre ventana interactiva.",
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Activa zoom vertical con eje Y fijo en [0, 0.3].",
    )
    return parser.parse_args()


def resolve_states_file(path: Path) -> Path:
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"No existe la ruta indicada: {path}")

    states_path = path / "states.txt"
    if states_path.exists():
        return states_path

    candidates = sorted(
        [
            candidate
            for candidate in path.glob("*.txt")
            if candidate.is_file() and not candidate.name.endswith("_events.txt")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No se encontro states.txt ni archivos .txt compatibles en {path}"
        )

    return candidates[0]


def parse_n_from_header(raw_line: str) -> Optional[int]:
    line = raw_line.strip()
    if not line.startswith("#"):
        return None

    payload = line[1:].strip()
    if not payload.startswith("N="):
        return None

    raw_value = payload.split("=", 1)[1].strip()
    try:
        return int(raw_value)
    except ValueError:
        return None


def parse_n_values(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("Todos los N deben ser > 0")
        values.append(value)

    if not values:
        raise ValueError("Debes indicar al menos un N en --n-values")

    return values


def parse_used_fraction_timeseries(states_path: Path) -> Tuple[List[float], List[float], int]:
    times: List[float] = []
    fu_values: List[float] = []

    n_from_header: Optional[int] = None
    n_particles: Optional[int] = None

    current_t: Optional[float] = None
    frame_total = 0
    frame_used = 0

    with states_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            maybe_n = parse_n_from_header(raw_line)
            if maybe_n is not None:
                n_from_header = maybe_n

            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                t_value = float(parts[0])
                state = int(parts[6])
            except ValueError as exc:
                raise ValueError(
                    f"Linea invalida en {states_path}:{line_number}: {line}"
                ) from exc

            if state not in (0, 1):
                raise ValueError(
                    f"Estado invalido en {states_path}:{line_number}. Se esperaba 0 o 1 y llego {state}."
                )

            if current_t is None:
                current_t = t_value
            elif abs(t_value - current_t) > 1.0e-12:
                if frame_total <= 0:
                    raise ValueError(
                        f"Frame vacio detectado en {states_path}:{line_number}"
                    )

                if n_particles is None:
                    n_particles = frame_total
                elif frame_total != n_particles:
                    raise ValueError(
                        f"Cantidad de particulas inconsistente en t={current_t}: "
                        f"esperadas {n_particles}, encontradas {frame_total}."
                    )

                times.append(current_t)
                fu_values.append(frame_used / frame_total)

                current_t = t_value
                frame_total = 0
                frame_used = 0

            frame_total += 1
            if state == 0:
                frame_used += 1

    if current_t is None or frame_total <= 0:
        raise ValueError(f"No se encontraron datos de estados en {states_path}")

    if n_particles is None:
        n_particles = frame_total
    elif frame_total != n_particles:
        raise ValueError(
            f"Cantidad de particulas inconsistente en ultimo frame: "
            f"esperadas {n_particles}, encontradas {frame_total}."
        )

    times.append(current_t)
    fu_values.append(frame_used / frame_total)

    if n_from_header is not None and n_from_header != n_particles:
        raise ValueError(
            f"N en header ({n_from_header}) no coincide con datos ({n_particles}) en {states_path}."
        )

    for idx in range(1, len(times)):
        if times[idx] + 1.0e-12 < times[idx - 1]:
            raise ValueError(f"Serie temporal no monotona en {states_path}")

    return times, fu_values, n_particles


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 30,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )


def first_repetition_states_path(runs_dir: Path, n_particles: int) -> Path:
    run_dirs = sorted(
        [path for path in runs_dir.glob(f"n{n_particles}_rep*") if path.is_dir()],
        key=lambda p: p.name,
    )

    for run_dir in run_dirs:
        candidate = run_dir / "states.txt"
        if candidate.exists():
            return candidate

    scan_files = sorted(
        [
            path
            for path in runs_dir.glob(f"scan_N{n_particles}_*.txt")
            if path.is_file() and not path.name.endswith("_events.txt")
        ],
        key=lambda p: p.name,
    )
    if scan_files:
        return scan_files[0]

    raise FileNotFoundError(
        f"No se encontro corrida para N={n_particles} en {runs_dir}. "
        "Se esperaba nN_rep*/states.txt o scan_N*_*.txt"
    )


def plot_used_fraction(
    series: Sequence[Tuple[int, List[float], List[float]]],
    vline_time: float,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    zoom: bool,
    show: bool,
) -> None:
    configure_plot_style()

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for n_particles, times, fu_values in series:
        ax.plot(times, fu_values, linewidth=1.6, label=f"N={n_particles}")

    vline = ax.axvline(
        vline_time,
        color="#ff8c00",
        linestyle="--",
        linewidth=2.0,
        label=f"t = {vline_time:.0f}s",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if zoom:
        ax.set_ylim(0.0, 0.3)
    else:
        ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)

    if title:
        ax.set_title(title)
    else:
        if len(series) == 1:
            ax.set_title(f"Fraccion de particulas usadas (N={series[0][0]})")
        else:
            ns = ", ".join(str(n_particles) for n_particles, _, _ in series)
            ax.set_title(f"Fraccion de particulas usadas (N={ns})")

    ax.legend(handles=[vline], labels=[f"t = {vline_time:.0f}s"], loc="best", fontsize=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()

    series: List[Tuple[int, List[float], List[float]]] = []
    selected_inputs: List[Path] = []

    if args.runs_dir is not None:
        if not args.runs_dir.exists() or not args.runs_dir.is_dir():
            raise FileNotFoundError(f"No existe la carpeta indicada en --runs-dir: {args.runs_dir}")

        n_values = parse_n_values(args.n_values)
        for n_particles in n_values:
            states_path = first_repetition_states_path(args.runs_dir, n_particles)
            times, fu_values, n_detected = parse_used_fraction_timeseries(states_path)
            series.append((n_detected, times, fu_values))
            selected_inputs.append(states_path)
    else:
        if args.run is None:
            raise ValueError("Debes pasar una corrida (run) o usar --runs-dir con --n-values")
        states_path = resolve_states_file(args.run)
        times, fu_values, n_particles = parse_used_fraction_timeseries(states_path)
        series.append((n_particles, times, fu_values))
        selected_inputs.append(states_path)

    plot_used_fraction(
        series=series,
        vline_time=args.vline_time,
        output_path=args.out,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        zoom=args.zoom,
        show=not args.no_show,
    )

    for index, input_path in enumerate(selected_inputs, start=1):
        n_particles, times, _ = series[index - 1]
        print(f"[OK] Serie {index}: archivo={input_path.resolve()} N={n_particles} muestras={len(times)}")
    print(f"[OK] Figura guardada en: {args.out.resolve()}")


if __name__ == "__main__":
    main()
