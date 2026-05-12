import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def parse_header_value(value):
    value = value.strip()
    token = value.split()[0] if value else value
    return float(token)


def parse_header(lines):
    header = {}
    for line in lines:
        entry = line[1:].strip()
        if "=" in entry:
            key, value = entry.split("=", 1)
            header[key.strip()] = value.strip()
    return header


def select_input_file(input_path, default_dir):
    if input_path:
        return Path(input_path)
    candidates = []
    for path in Path(default_dir).glob("*.txt"):
        if path.name.endswith("_events.txt"):
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No state files found in {default_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def compute_frame_energy(x, y, vx, vy, m, k, r, r0, outer_radius):
    kinetic = 0.5 * m * np.sum(vx * vx + vy * vy)

    potential = 0.0
    n = x.size
    min_dist = 2.0 * r

    for i in range(n):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n):
            dx = xi - x[j]
            dy = yi - y[j]
            dist2 = dx * dx + dy * dy
            if dist2 < min_dist * min_dist:
                dist = math.sqrt(dist2)
                overlap = min_dist - dist
                potential += 0.5 * k * overlap * overlap

    for i in range(n):
        dist = math.sqrt(x[i] * x[i] + y[i] * y[i])
        overlap_obstacle = r0 + r - dist
        if overlap_obstacle > 0.0:
            potential += 0.5 * k * overlap_obstacle * overlap_obstacle
        overlap_wall = r + dist - outer_radius
        if overlap_wall > 0.0:
            potential += 0.5 * k * overlap_wall * overlap_wall

    return kinetic + potential


def load_frames(path):
    header_lines = []
    rows = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header_lines.append(line)
                continue
            rows.append(line)

    header = parse_header(header_lines)
    return header, rows


def compute_energy_series(header, rows):
    n = int(header["N"])
    m = parse_header_value(header["m"])
    k = parse_header_value(header["k"])
    r = parse_header_value(header["r"])
    r0 = parse_header_value(header["r0"])
    l = parse_header_value(header["L"])
    outer_radius = l / 2.0

    times = []
    energies = []

    x = np.zeros(n)
    y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)

    current_time = None
    filled = 0

    for row in rows:
        parts = row.split()
        t = float(parts[0])
        pid = int(parts[1])
        if current_time is None:
            current_time = t
        if t != current_time:
            if filled != n:
                raise ValueError(f"Frame at t={current_time} has {filled} particles, expected {n}")
            energy = compute_frame_energy(x, y, vx, vy, m, k, r, r0, outer_radius)
            times.append(current_time)
            energies.append(energy)
            current_time = t
            filled = 0

        x[pid] = float(parts[2])
        y[pid] = float(parts[3])
        vx[pid] = float(parts[4])
        vy[pid] = float(parts[5])
        filled += 1

    if current_time is not None:
        if filled != n:
            raise ValueError(f"Frame at t={current_time} has {filled} particles, expected {n}")
        energy = compute_frame_energy(x, y, vx, vy, m, k, r, r0, outer_radius)
        times.append(current_time)
        energies.append(energy)

    return np.array(times), np.array(energies)


def build_plain_formatter(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return FuncFormatter(lambda _v, _p: "0")
    span = float(np.max(finite) - np.min(finite))
    if span == 0.0:
        decimals = 6
    else:
        decimals = max(0, int(-math.floor(math.log10(span))) + 1)
        decimals = min(decimals, 6)

    def formatter(value, _pos):
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

    return FuncFormatter(formatter)


def apply_axis_format(ax, values):
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total energy (J)")
    ax.grid(True, which="both", alpha=0.2)
    ax.yaxis.set_major_formatter(build_plain_formatter(values))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)


def apply_y_limits(ax, values, pad_fraction):
    if values.size == 0:
        return
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    if span <= 0.0:
        span = abs(vmax) if vmax != 0.0 else 1.0
    pad = span * pad_fraction
    ax.set_ylim(vmin - pad, vmax + pad)


def main():
    parser = argparse.ArgumentParser(description="Plot total energy vs time for System 2 outputs.")
    parser.add_argument("--input", default=None, help="State output file path")
    parser.add_argument("--input-dir", default="outputs/scanningRate", help="Default directory to search for outputs")
    parser.add_argument("--out", default="scanning_rate_energy.png", help="Output image path")
    parser.add_argument("--relative", action="store_true", help="Plot (E - E0) / E0 instead of E")
    parser.add_argument("--y-pad", type=float, default=0.5, help="Extra padding fraction for Y limits")
    parser.add_argument("--y-min", type=float, default=None, help="Fixed Y min")
    parser.add_argument("--y-max", type=float, default=None, help="Fixed Y max")
    args = parser.parse_args()

    input_path = select_input_file(args.input, args.input_dir)
    header, rows = load_frames(input_path)
    times, energies = compute_energy_series(header, rows)

    if times.size == 0:
        raise ValueError("No frames found in the selected output file.")

    if args.relative:
        baseline = energies[0]
        values = (energies - baseline) / baseline if baseline != 0.0 else energies
        ylabel = "Relative energy (E - E0) / E0"
    else:
        values = energies
        ylabel = "Total energy (J)"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, values, color="black", linewidth=1.5)
    ax.set_ylabel(ylabel)
    apply_axis_format(ax, values)
    if (args.y_min is not None) ^ (args.y_max is not None):
        raise ValueError("Provide both --y-min and --y-max, or neither.")
    if args.y_min is not None and args.y_max is not None:
        ax.set_ylim(args.y_min, args.y_max)
    else:
        apply_y_limits(ax, values, args.y_pad)
    ax.margins(x=0.0, y=0.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
