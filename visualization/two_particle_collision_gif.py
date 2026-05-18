#!/usr/bin/env python3
"""Render a GIF of two-particle collision using the same math as simulation.

Physics matched with simulation package:
- Contact force only when overlap > 0: F = k * overlap along collision normal
- Acceleration: a = F / m
- Integrator: Velocity Verlet (same update pattern)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle


def compute_accelerations(x: np.ndarray, y: np.ndarray, m: float, k: float, r: float) -> tuple[np.ndarray, np.ndarray]:
    ax = np.zeros(2)
    ay = np.zeros(2)

    min_dist = 2.0 * r
    dx = x[0] - x[1]
    dy = y[0] - y[1]
    dist2 = dx * dx + dy * dy

    if dist2 < min_dist * min_dist:
        dist = float(np.sqrt(dist2))
        nx = dx / dist if dist > 0.0 else 1.0
        ny = dy / dist if dist > 0.0 else 0.0
        overlap = min_dist - dist
        accel = (k / m) * overlap

        ax[0] += accel * nx
        ay[0] += accel * ny
        ax[1] -= accel * nx
        ay[1] -= accel * ny

    return ax, ay


def velocity_verlet_step(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    dt: float,
    m: float,
    k: float,
    r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_dt = 0.5 * dt
    dt2 = dt * dt

    x = x + vx * dt + 0.5 * ax * dt2
    y = y + vy * dt + 0.5 * ay * dt2
    vx = vx + ax * half_dt
    vy = vy + ay * half_dt

    ax_new, ay_new = compute_accelerations(x, y, m, k, r)

    vx = vx + ax_new * half_dt
    vy = vy + ay_new * half_dt

    return x, y, vx, vy, ax_new, ay_new


def simulate(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    x = np.array([args.x1, args.x2], dtype=float)
    y = np.array([args.y1, args.y2], dtype=float)
    vx = np.array([args.vx1, args.vx2], dtype=float)
    vy = np.array([args.vy1, args.vy2], dtype=float)

    ax, ay = compute_accelerations(x, y, args.m, args.k, args.r)

    t = 0.0
    next_out = 0.0
    steps = int(np.ceil(args.tf / args.dt))

    times: list[float] = []
    frames: list[tuple[np.ndarray, np.ndarray]] = []

    for step in range(steps + 1):
        if t + 1e-12 >= next_out:
            times.append(t)
            frames.append((x.copy(), y.copy()))
            next_out += args.dt2

        if step == steps:
            break

        x, y, vx, vy, ax, ay = velocity_verlet_step(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            ax=ax,
            ay=ay,
            dt=args.dt,
            m=args.m,
            k=args.k,
            r=args.r,
        )
        t += args.dt

    return np.array(times), frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a zoomed GIF of two particles colliding.")

    parser.add_argument("--r", type=float, default=1.0, help="Particle radius")
    parser.add_argument("--m", type=float, default=1.0, help="Particle mass")
    parser.add_argument("--k", type=float, default=1.0e3, help="Elastic constant")

    parser.add_argument("--x1", type=float, default=-3.0)
    parser.add_argument("--y1", type=float, default=0.0)
    parser.add_argument("--x2", type=float, default=3.0)
    parser.add_argument("--y2", type=float, default=0.0)

    parser.add_argument("--vx1", type=float, default=1.0)
    parser.add_argument("--vy1", type=float, default=0.0)
    parser.add_argument("--vx2", type=float, default=-1.0)
    parser.add_argument("--vy2", type=float, default=0.0)

    parser.add_argument("--tf", type=float, default=8.0, help="Final time")
    parser.add_argument("--dt", type=float, default=0.001, help="Integration time step")
    parser.add_argument("--dt2", type=float, default=0.002, help="Output frame time step")

    parser.add_argument("--zoom", type=float, default=4.0, help="Half-width of zoom window")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("outputs") / "two_particle_collision.gif")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times, frames = simulate(args)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", "box")
    ax.set_xlim(-args.zoom, args.zoom)
    ax.set_ylim(-args.zoom, args.zoom)
    ax.set_xlabel("Posicion x (m)")
    ax.set_ylabel("Posicion y (m)")
    ax.grid(True, alpha=0.2)

    p1 = Circle((frames[0][0][0], frames[0][1][0]), args.r, fill=False, edgecolor="#1f77b4", linewidth=2.0)
    p2 = Circle((frames[0][0][1], frames[0][1][1]), args.r, fill=False, edgecolor="#d62728", linewidth=2.0)
    ax.add_patch(p1)
    ax.add_patch(p2)

    info = ax.set_title("", fontsize=14)

    def update(i: int):
        x, y = frames[i]
        p1.center = (x[0], y[0])
        p2.center = (x[1], y[1])

        dx = x[0] - x[1]
        dy = y[0] - y[1]
        dist = float(np.sqrt(dx * dx + dy * dy))
        overlap = max(0.0, 2.0 * args.r - dist)

        info.set_text(f"t={times[i]:.4f} s | dist={dist:.4f} | overlap={overlap:.4f}")
        return p1, p2, info

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=args.fps)
    anim.save(str(args.out), writer=writer, dpi=200)
    plt.close(fig)

    print(f"GIF saved to: {args.out}")


if __name__ == "__main__":
    main()
