import argparse
from datetime import datetime
import math
from pathlib import Path
import subprocess

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


def resolve_input_path(path, repo_root):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return repo_root / path


def resolve_output_path(path, repo_root):
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def parse_properties(path):
    data = {}
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def find_existing_state_by_n_tf(input_dir, n_value, tf_value):
    if not input_dir.exists():
        raise FileNotFoundError(f"No existe input-dir: {input_dir}")

    candidates = sorted(
        [p for p in input_dir.glob("*.txt") if not p.name.endswith("_events.txt")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No state files found in {input_dir}")

    tf_tol = 1e-9
    for state_path in candidates:
        props_path = state_path.with_name(state_path.stem + "_properties.txt")
        props = parse_properties(props_path)
        if "N" not in props or "tf" not in props:
            continue
        try:
            n_curr = int(float(props["N"]))
            tf_curr = float(props["tf"])
        except ValueError:
            continue
        if n_curr == n_value and abs(tf_curr - tf_value) <= tf_tol:
            return state_path

    raise FileNotFoundError(
        f"No se encontró salida existente con N={n_value} y tf={tf_value} en {input_dir}"
    )


def run_simulation_for_animation(
    repo_root,
    input_dir,
    n_value,
    tf_value,
    dt,
    dt2,
    l,
    r0,
    r,
    m,
    k,
    v0,
    seed,
    java_cmd,
):
    input_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tf_tag = str(tf_value).replace(".", "p")
    base = f"scan_n{n_value}_tf{tf_tag}_{stamp}"

    state_path = input_dir / f"{base}.txt"
    events_path = input_dir / f"{base}_events.txt"
    props_path = input_dir / f"{base}_properties.txt"

    cmd = [
        java_cmd,
        "-cp",
        str(repo_root / "simulation"),
        "ScanningRateSimulation",
        "--n",
        str(n_value),
        "--tf",
        str(tf_value),
        "--dt",
        str(dt),
        "--dt2",
        str(dt2),
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
        "--out",
        str(state_path),
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
            "Falló la simulación para animación.\n"
            f"Comando: {' '.join(cmd)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    if not state_path.exists():
        raise FileNotFoundError(f"La simulación terminó pero no generó {state_path}")

    return state_path


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
    parser.add_argument("--n", type=int, default=None, help="N para ejecutar/buscar corrida")
    parser.add_argument("--tf", type=float, default=None, help="tf para ejecutar/buscar corrida")
    parser.add_argument("--use-existing", action="store_true", help="Con --n y --tf, usar corrida existente en input-dir en vez de simular nueva")
    parser.add_argument("--java-cmd", default="java", help="Java executable")
    parser.add_argument("--dt", type=float, default=0.001, help="Integration step for auto-run")
    parser.add_argument("--dt2", type=float, default=0.1, help="Output step for auto-run")
    parser.add_argument("--l", type=float, default=80.0, help="Domain diameter for auto-run")
    parser.add_argument("--r0", type=float, default=1.0, help="Obstacle radius for auto-run")
    parser.add_argument("--r", type=float, default=1.0, help="Particle radius for auto-run")
    parser.add_argument("--m", type=float, default=1.0, help="Particle mass for auto-run")
    parser.add_argument("--k", type=float, default=1.0e3, help="Elastic constant for auto-run")
    parser.add_argument("--v0", type=float, default=1.0, help="Initial speed for auto-run")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for auto-run")
    parser.add_argument("--out", default="scanning_rate_animation.mp4", help="Output animation path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to keep")
    parser.add_argument("--t-start", type=float, default=None, help="Start time for animation window")
    parser.add_argument("--t-end", type=float, default=None, help="End time for animation window")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    input_dir = resolve_input_path(args.input_dir, repo_root)

    if args.input is not None:
        input_path = resolve_input_path(args.input, repo_root)
    elif args.n is not None and args.tf is not None:
        if args.use_existing:
            input_path = find_existing_state_by_n_tf(input_dir, args.n, args.tf)
            print(f"Usando corrida existente: {input_path}")
        else:
            input_path = run_simulation_for_animation(
                repo_root=repo_root,
                input_dir=input_dir,
                n_value=args.n,
                tf_value=args.tf,
                dt=args.dt,
                dt2=args.dt2,
                l=args.l,
                r0=args.r0,
                r=args.r,
                m=args.m,
                k=args.k,
                v0=args.v0,
                seed=args.seed,
                java_cmd=args.java_cmd,
            )
            print(f"Corrida generada: {input_path}")
    else:
        input_path = select_input_file(None, input_dir)

    header, times, frames = load_frames(input_path, args.stride, args.max_frames, args.t_start, args.t_end)

    if not frames:
        raise ValueError("No frames selected for animation. Adjust stride or time window.")

    l = parse_header_value(header["L"])
    r0 = parse_header_value(header["r0"])
    r = parse_header_value(header["r"])

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
    particles = []
    for i in range(x0.size):
        edge_color = FRESH_COLOR if state0[i] == 1 else USED_COLOR
        particle = Circle((x0[i], y0[i]), r, fill=False, edgecolor=edge_color, linewidth=0.9)
        ax.add_patch(particle)
        particles.append(particle)

    fig.subplots_adjust(top=0.9)
    title_text = ax.set_title("", pad=12, fontsize=20)

    def update(frame_idx):
        x, y, state = frames[frame_idx]
        for i, particle in enumerate(particles):
            particle.center = (x[i], y[i])
            particle.set_edgecolor(FRESH_COLOR if state[i] == 1 else USED_COLOR)
        title_text.set_text(f"Tiempo (s): {times[frame_idx]:.2f}")
        return particles + [title_text]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)

    out_path = resolve_output_path(args.out, repo_root)
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
