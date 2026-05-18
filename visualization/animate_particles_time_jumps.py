#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.lines import Line2D


def parse_n_values(raw: str) -> list[int]:
    values: list[int] = []
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


def parse_properties(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def build_target_times(start_time: float, end_time: float, time_step: float) -> list[float]:
    if time_step <= 0.0:
        raise ValueError("time_step debe ser > 0")
    if end_time < start_time:
        return []

    times: list[float] = []
    t = start_time
    while t <= end_time + 1.0e-12:
        times.append(t)
        t += time_step
    return times


def sample_states_for_target_times(
    states_path: Path,
    n_particles: int,
    target_times: list[float],
    time_tolerance: float,
) -> list[tuple[float, list[tuple[float, float, int]]]]:
    sampled: list[tuple[float, list[tuple[float, float, int]]]] = []
    if not target_times:
        return sampled

    next_target_idx = 0
    current_t: float | None = None
    current_particles: list[tuple[float, float, int]] = []

    def maybe_store_frame(frame_t: float, frame_particles: list[tuple[float, float, int]]) -> None:
        nonlocal next_target_idx
        if len(frame_particles) != n_particles:
            raise ValueError(
                f"Frame en t={frame_t} tiene {len(frame_particles)} particulas; se esperaban {n_particles}."
            )

        # Advance targets using the first available frame at or after each target time.
        while next_target_idx < len(target_times):
            target_t = target_times[next_target_idx]
            if frame_t + time_tolerance < target_t:
                break
            sampled.append((frame_t, frame_particles.copy()))
            next_target_idx += 1

    with states_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
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

            if abs(t - current_t) > 1.0e-12:
                maybe_store_frame(current_t, current_particles)
                if next_target_idx >= len(target_times):
                    break
                current_particles = []
                current_t = t

            current_particles.append((x, y, state))

        if current_t is not None and next_target_idx < len(target_times):
            maybe_store_frame(current_t, current_particles)

    return sampled


def render_animation(
    sampled_frames: list[tuple[float, list[tuple[float, float, int]]]],
    out_path: Path,
    L: float,
    r0: float,
    r: float,
    fps: int,
    hold_seconds: float,
    dpi: int,
    n_value: int,
    rep: int,
) -> None:
    if not sampled_frames:
        raise ValueError("No hay frames para animar")

    repeat_count = max(1, int(round(hold_seconds * fps)))
    expanded_indices = [
        frame_idx
        for frame_idx in range(len(sampled_frames))
        for _ in range(repeat_count)
    ]

    outer_radius = L / 2.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-outer_radius - 1.0, outer_radius + 1.0)
    ax.set_ylim(-outer_radius - 1.0, outer_radius + 1.0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)

    boundary = plt.Circle((0.0, 0.0), outer_radius, fill=False, linewidth=1.8, color="black")
    obstacle = plt.Circle((0.0, 0.0), r0, color="black", alpha=0.7)
    ax.add_patch(boundary)
    ax.add_patch(obstacle)

    first_time, first_particles = sampled_frames[0]
    particle_circles = []
    for x, y, state in first_particles:
        circle = plt.Circle(
            (x, y),
            r,
            facecolor="green" if state == 1 else "purple",
            edgecolor="black",
            linewidth=0.3,
        )
        ax.add_patch(circle)
        particle_circles.append(circle)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="green", markeredgecolor="black", markersize=8, label="Frescas"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="purple", markeredgecolor="black", markersize=8, label="Usadas"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    title_text = ax.set_title(
        f"N={n_value} rep={rep} | t={first_time:.1f} s",
        pad=12,
    )

    def update(anim_idx: int):
        frame_idx = expanded_indices[anim_idx]
        t, particles = sampled_frames[frame_idx]

        for i, circle in enumerate(particle_circles):
            x, y, state = particles[i]
            circle.center = (x, y)
            circle.set_facecolor("green" if state == 1 else "purple")

        title_text.set_text(f"N={n_value} rep={rep} | t={t:.1f} s")
        return (*particle_circles, title_text)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(expanded_indices),
        interval=1000 / fps,
        blit=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".gif":
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    elif suffix == ".mp4":
        if not writers.is_available("ffmpeg"):
            raise RuntimeError("No se encontro ffmpeg para exportar mp4. Usa .gif o instala ffmpeg.")
        anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2000), dpi=dpi)
    else:
        raise ValueError("Formato no soportado. Usa .gif o .mp4")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Genera animaciones con saltos de tiempo grandes (10s, 20s, 30s, ...) "
            "desde corridas existentes del sweep."
        )
    )
    parser.add_argument("--sweep-root", type=Path, default=Path("sweep_n_100_1000_2"))
    parser.add_argument("--n-values", type=str, default="300,600,900")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--start-time", type=float, default=100.0)
    parser.add_argument("--time-step", type=float, default=100.0)
    parser.add_argument("--end-time", type=float, default=None, help="Default: usa tf del properties.txt")
    parser.add_argument("--time-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--hold-seconds", type=float, default=1.2)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "videos" / "time_jumps")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    sweep_root = args.sweep_root
    if not sweep_root.is_absolute():
        sweep_root = repo_root / sweep_root

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    n_values = parse_n_values(args.n_values)

    for n_value in n_values:
        run_dir = sweep_root / f"n{n_value}_rep{args.rep}"
        states_path = run_dir / "states.txt"
        props_path = run_dir / "properties.txt"

        if not states_path.exists():
            raise FileNotFoundError(f"No existe states.txt: {states_path}")
        if not props_path.exists():
            raise FileNotFoundError(f"No existe properties.txt: {props_path}")

        props = parse_properties(props_path)
        tf_value = float(props["tf"])
        L = float(props["L"])
        r0 = float(props["r0"])
        r = float(props["r"])

        end_time = tf_value if args.end_time is None else min(args.end_time, tf_value)
        target_times = build_target_times(args.start_time, end_time, args.time_step)

        sampled_frames = sample_states_for_target_times(
            states_path=states_path,
            n_particles=n_value,
            target_times=target_times,
            time_tolerance=args.time_tolerance,
        )

        out_name = f"n{n_value}_rep{args.rep}_jump{args.time_step:g}s.{args.format}"
        out_path = out_dir / out_name

        render_animation(
            sampled_frames=sampled_frames,
            out_path=out_path,
            L=L,
            r0=r0,
            r=r,
            fps=args.fps,
            hold_seconds=args.hold_seconds,
            dpi=args.dpi,
            n_value=n_value,
            rep=args.rep,
        )

        print(f"[OK] N={n_value} -> {out_path}")


if __name__ == "__main__":
    main()
