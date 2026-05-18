#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, writers
from matplotlib.lines import Line2D


def parse_properties(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def find_states_by_n_tf(outputs_root: Path, n: int, tf: float, repetition: int | None = None) -> Path:
    if not outputs_root.exists():
        raise FileNotFoundError(f"No existe outputs-root: {outputs_root}")

    run_dirs = sorted(
        [p for p in outputs_root.glob(f"n{n}_rep*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if repetition is not None:
        target = outputs_root / f"n{n}_rep{repetition}"
        run_dirs = [target] if target.is_dir() else []

    if not run_dirs:
        raise FileNotFoundError(
            f"No se encontraron corridas para N={n} en {outputs_root}"
        )

    tf_tol = 1e-9
    for run_dir in run_dirs:
        states_path = run_dir / "states.txt"
        props_path = run_dir / "properties.txt"
        if not states_path.exists():
            continue

        props = parse_properties(props_path)
        if "tf" in props:
            try:
                tf_value = float(props["tf"])
            except ValueError:
                continue
            if abs(tf_value - tf) <= tf_tol:
                return states_path

    raise FileNotFoundError(
        f"No se encontró states.txt para N={n} con tf={tf} en {outputs_root}. "
        "Tip: revisá properties.txt o indicá --states manualmente."
    )


def resolve_input_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return repo_root / path


def resolve_output_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def run_simulation_for_animation(
    repo_root: Path,
    outputs_root: Path,
    n: int,
    tf: float,
    l: float,
    r0: float,
    r: float,
    m: float,
    k: float,
    v0: float,
    dt: float,
    dt2: float,
    seed: int | None,
    java_cmd: str,
) -> Path:
    outputs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tf_tag = str(tf).replace(".", "p")
    run_dir = outputs_root / f"run_n{n}_tf{tf_tag}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    states_path = run_dir / "states.txt"
    events_path = run_dir / "events.txt"
    props_path = run_dir / "properties.txt"

    cmd = [
        java_cmd,
        "-cp",
        str(repo_root / "simulation"),
        "ScanningRateSimulation",
        "--n",
        str(n),
        "--l",
        str(l),
        "--r0",
        str(r0),
        "--r",
        str(r),
        "--m",
        str(m),
        "--k",
        str(k),
        "--v0",
        str(v0),
        "--tf",
        str(tf),
        "--dt",
        str(dt),
        "--dt2",
        str(dt2),
        "--out",
        str(states_path),
        "--events-out",
        str(events_path),
        "--properties-out",
        str(props_path),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    process = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            "Falló la simulación para animar.\n"
            f"Comando: {' '.join(cmd)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    if not states_path.exists():
        raise FileNotFoundError(f"La simulación terminó pero no generó states.txt en {run_dir}")

    return states_path


def read_states(path: Path, max_frames: int | None = None):
    frames = []
    current_t = None
    current = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            t = float(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            state = int(parts[6])

            if current_t is None:
                current_t = t

            if abs(t - current_t) > 1e-12:
                frames.append((current_t, current))
                if max_frames is not None and len(frames) >= max_frames:
                    return frames
                current = []
                current_t = t

            current.append((x, y, state))

    if current:
        frames.append((current_t, current))

    return frames


def animate(states_path: Path, output_path: Path, L: float, r0: float, r: float, fps: int, max_frames: int | None, frame_step: int, dpi: int):
    frames = read_states(states_path, max_frames=max_frames)
    frames = frames[::frame_step]

    if not frames:
        raise ValueError(f"No se encontraron frames en {states_path}")

    outer_radius = L / 2.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-outer_radius - 1, outer_radius + 1)
    ax.set_ylim(-outer_radius - 1, outer_radius + 1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)

    outer = plt.Circle((0, 0), outer_radius, fill=False, linewidth=2)
    obstacle = plt.Circle((0, 0), r0, color="black", alpha=0.7)

    ax.add_patch(outer)
    ax.add_patch(obstacle)

    max_particles = max(len(particles) for _, particles in frames)
    particle_circles = []
    for _ in range(max_particles):
        circle = plt.Circle(
            (0.0, 0.0),
            r,
            facecolor="green",
            edgecolor="black",
            linewidth=0.3,
            visible=False,
        )
        ax.add_patch(circle)
        particle_circles.append(circle)

    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="green", markeredgecolor="black", markersize=8, label="Frescas"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="purple", markeredgecolor="black", markersize=8, label="Usadas"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    def update(frame_index: int):
        t, particles = frames[frame_index]

        for idx, circle in enumerate(particle_circles):
            if idx < len(particles):
                x, y, state = particles[idx]
                circle.center = (x, y)
                circle.set_facecolor("green" if state == 1 else "purple")
                circle.set_visible(True)
            else:
                circle.set_visible(False)

        time_text.set_text(f"t = {t:.2f} s")
        return (*particle_circles, time_text)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)

    elif suffix == ".mp4":
        if not writers.is_available("ffmpeg"):
            raise RuntimeError(
                "No se encontró ffmpeg. Instalalo con: winget install Gyan.FFmpeg "
                "o guardá como .gif."
            )

        animation.save(
            output_path,
            writer=FFMpegWriter(fps=fps, bitrate=1800),
            dpi=dpi,
        )

    else:
        raise ValueError("Formato no soportado. Usá .gif o .mp4.")

    plt.close(fig)
    print(f"Animación guardada en {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Animación de partículas TP4.")

    parser.add_argument(
        "--states",
        type=Path,
        default=None,
        help="Archivo states.txt generado por la simulación.",
    )
    parser.add_argument("--n", type=int, default=None, help="N de la corrida a animar (si no se usa --states).")
    parser.add_argument("--tf", type=float, default=None, help="tf de la corrida a animar (si no se usa --states).")
    parser.add_argument("--rep", type=int, default=None, help="Repetición específica (opcional).")
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Si se usa con --n y --tf, busca una corrida existente en vez de ejecutar una nueva.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs") / "videos",
        help="Carpeta raíz para corridas/animaciones (usada con --n y --tf).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/animation_tp4.gif"),
        help="Archivo de salida .gif o .mp4.",
    )
    parser.add_argument("--L", type=float, default=80.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=1.0e3)
    parser.add_argument("--v0", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--dt2", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--java-cmd", type=str, default="java", help="Ejecutable de Java.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limita la cantidad de frames para probar rápido.",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=80)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    states_path = args.states
    if states_path is None:
        if args.n is None or args.tf is None:
            parser.error("Debés pasar --states o bien --n y --tf (con --outputs-root opcional).")
        outputs_root = resolve_output_path(args.outputs_root, repo_root)
        if args.use_existing:
            states_path = find_states_by_n_tf(
                outputs_root=outputs_root,
                n=args.n,
                tf=args.tf,
                repetition=args.rep,
            )
            print(f"Usando corrida existente: {states_path}")
        else:
            states_path = run_simulation_for_animation(
                repo_root=repo_root,
                outputs_root=outputs_root,
                n=args.n,
                tf=args.tf,
                l=args.L,
                r0=args.r0,
                r=args.r,
                m=args.m,
                k=args.k,
                v0=args.v0,
                dt=args.dt,
                dt2=args.dt2,
                seed=args.seed,
                java_cmd=args.java_cmd,
            )
            print(f"Corrida generada: {states_path}")
    else:
        states_path = resolve_input_path(states_path, repo_root)

    if not states_path.exists():
        raise FileNotFoundError(f"No existe states file: {states_path}")

    output_path = resolve_output_path(args.out, repo_root)

    animate(
        states_path=states_path,
        output_path=output_path,
        L=args.L,
        r0=args.r0,
        r=args.r,
        fps=args.fps,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        dpi=args.dpi,
    )

if __name__ == "__main__":
    main()