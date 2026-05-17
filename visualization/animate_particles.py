#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, writers


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

    fresh_scatter = ax.scatter([], [], s=(2 * r * 8) ** 2, c="green", label="Frescas")
    used_scatter = ax.scatter([], [], s=(2 * r * 8) ** 2, c="purple", label="Usadas")

    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax.legend(loc="upper right")

    def update(frame_index: int):
        t, particles = frames[frame_index]

        fresh_xy = [(x, y) for x, y, state in particles if state == 1]
        used_xy = [(x, y) for x, y, state in particles if state == 0]

        fresh_scatter.set_offsets(fresh_xy if fresh_xy else [[math.nan, math.nan]])
        used_scatter.set_offsets(used_xy if used_xy else [[math.nan, math.nan]])

        time_text.set_text(f"t = {t:.2f} s")
        return fresh_scatter, used_scatter, time_text

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
        required=True,
        help="Archivo states.txt generado por la simulación.",
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

    animate(
        states_path=args.states,
        output_path=args.out,
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