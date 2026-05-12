import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle

FRESH_COLOR = "#2ca02c"
USED_COLOR = "#d62728"


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


def load_frames(path, stride, max_frames, t_start, t_end):
    header_lines = []
    frames = []
    times = []

    with path.open("r", encoding="ascii") as handle:
        first_data_line = None
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header_lines.append(line)
                continue
            header = parse_header(header_lines)
            n = int(header["N"])
            first_data_line = line
            break
        else:
            raise ValueError("No data rows found in output file.")

        current_time = None
        x = np.zeros(n)
        y = np.zeros(n)
        state = np.zeros(n, dtype=int)
        filled = 0
        frame_index = 0

        def consume_row(row_line):
            nonlocal current_time, filled, frame_index
            parts = row_line.split()
            t = float(parts[0])
            pid = int(parts[1])

            if current_time is None:
                current_time = t
            if t != current_time:
                if filled != n:
                    raise ValueError(f"Frame at t={current_time} has {filled} particles, expected {n}")

                keep = (frame_index % stride == 0)
                in_window = True
                if t_start is not None and current_time < t_start:
                    in_window = False
                if t_end is not None and current_time > t_end:
                    in_window = False

                if keep and in_window:
                    frames.append((x.copy(), y.copy(), state.copy()))
                    times.append(current_time)
                    if max_frames is not None and len(frames) >= max_frames:
                        return False

                frame_index += 1
                current_time = t
                filled = 0

            x[pid] = float(parts[2])
            y[pid] = float(parts[3])
            state[pid] = int(parts[6])
            filled += 1
            return True

        if first_data_line is not None:
            if not consume_row(first_data_line):
                header = parse_header(header_lines)
                return header, times, frames

        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if not consume_row(line):
                break

        if max_frames is None or len(frames) < max_frames:
            if current_time is not None:
                if filled != n:
                    raise ValueError(f"Frame at t={current_time} has {filled} particles, expected {n}")
                keep = (frame_index % stride == 0)
                in_window = True
                if t_start is not None and current_time < t_start:
                    in_window = False
                if t_end is not None and current_time > t_end:
                    in_window = False
                if keep and in_window:
                    frames.append((x.copy(), y.copy(), state.copy()))
                    times.append(current_time)

    header = parse_header(header_lines)
    return header, times, frames


def main():
    parser = argparse.ArgumentParser(description="Animate System 2 outputs.")
    parser.add_argument("--input", default=None, help="State output file path")
    parser.add_argument("--input-dir", default="outputs/scanningRate", help="Default directory to search for outputs")
    parser.add_argument("--out", default="scanning_rate_animation.mp4", help="Output animation path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to keep")
    parser.add_argument("--t-start", type=float, default=None, help="Start time for animation window")
    parser.add_argument("--t-end", type=float, default=None, help="End time for animation window")
    args = parser.parse_args()

    input_path = select_input_file(args.input, args.input_dir)
    header, times, frames = load_frames(input_path, args.stride, args.max_frames, args.t_start, args.t_end)

    if not frames:
        raise ValueError("No frames selected for animation. Adjust stride or time window.")

    l = parse_header_value(header["L"])
    r0 = parse_header_value(header["r0"])

    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", "box")
    ax.set_xlim(-l / 2.0, l / 2.0)
    ax.set_ylim(-l / 2.0, l / 2.0)
    ax.set_xlabel("Posición x (m)")
    ax.set_ylabel("Posición y (m)")

    boundary = Circle((0.0, 0.0), l / 2.0, fill=False, color="black", linewidth=1.2)
    obstacle = Circle((0.0, 0.0), r0, fill=False, color="black", linewidth=1.2)
    ax.add_patch(boundary)
    ax.add_patch(obstacle)

    x0, y0, state0 = frames[0]
    colors = np.where(state0 == 1, FRESH_COLOR, USED_COLOR)
    scatter = ax.scatter(x0, y0, s=18, c=colors)

    fig.subplots_adjust(top=0.9)
    title_text = ax.set_title("", pad=12, fontsize=20)

    def update(frame_idx):
        x, y, state = frames[frame_idx]
        scatter.set_offsets(np.column_stack((x, y)))
        scatter.set_color(np.where(state == 1, FRESH_COLOR, USED_COLOR))
        title_text.set_text(f"Tiempo (s): {times[frame_idx]:.2f}")
        return scatter, title_text

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=True)

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=args.fps)
        anim.save(str(out_path), writer=writer, dpi=200)
    else:
        try:
            writer = animation.FFMpegWriter(fps=args.fps)
            anim.save(str(out_path), writer=writer, dpi=200)
        except FileNotFoundError:
            fallback = out_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=args.fps)
            anim.save(str(fallback), writer=writer, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
