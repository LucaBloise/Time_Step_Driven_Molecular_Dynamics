import argparse
import math
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

METHOD_ORDER = ["euler", "verlet", "beeman", "gear5"]
LABELS = {
    "euler": "Euler",
    "verlet": "Verlet",
    "beeman": "Beeman",
    "gear5": "Gear 5",
    "analytic": "Analítica",
}
COLORS = {
    "euler": "#1f77b4",
    "verlet": "#ff7f0e",
    "beeman": "#2ca02c",
    "gear5": "#d62728",
    "analytic": "black",
}


def format_dt_tag(dt):
    return f"{dt:.10g}".replace(".", "p").replace("-", "m")


def compile_simulation_sources(repo_root, javac_cmd):
    simulation_dir = repo_root / "simulation"
    java_files = sorted(simulation_dir.glob("*.java"))
    if not java_files:
        raise FileNotFoundError(f"No Java files found in {simulation_dir}")

    cmd = [javac_cmd] + [str(path) for path in java_files]
    process = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            "Compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def run_oscillator_outputs(repo_root, input_dir, args):
    input_dir.mkdir(parents=True, exist_ok=True)

    if args.run_compile:
        print("Compiling Java sources...", flush=True)
        compile_simulation_sources(repo_root, args.javac_cmd)

    run_dt2 = args.run_dt if args.run_dt2 is None else args.run_dt2
    dt_tag = format_dt_tag(args.run_dt)

    for method in METHOD_ORDER:
        out_path = input_dir / f"{method}_dt{dt_tag}.txt"
        if out_path.exists() and not args.run_force:
            print(f"Reusing existing output for {method}: {out_path}", flush=True)
            continue

        cmd = [
            args.java_cmd,
            "-cp",
            str(repo_root / "simulation"),
            "OscillatorSimulation",
            "--method",
            method,
            "--dt",
            str(args.run_dt),
            "--dt2",
            str(run_dt2),
            "--tf",
            str(args.run_tf),
            "--m",
            str(args.run_m),
            "--k",
            str(args.run_k),
            "--gamma",
            str(args.run_gamma),
            "--r0",
            str(args.run_r0),
            "--out",
            str(out_path),
        ]
        if args.run_v0 is not None:
            cmd.extend(["--v0", str(args.run_v0)])

        print(f"Running {method} (dt={args.run_dt:g}) ...", flush=True)
        process = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            raise RuntimeError(
                f"Simulation failed for method={method}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{process.stdout}\n"
                f"STDERR:\n{process.stderr}"
            )


def parse_header_value(value):
    value = value.strip()
    token = value.split()[0] if value else value
    return float(token)


def read_output(path):
    header = {}
    rows = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                entry = line[1:].strip()
                if "=" in entry:
                    key, value = entry.split("=", 1)
                    header[key.strip()] = value.strip()
                continue
            rows.append([float(x) for x in line.split()])

    data = np.array(rows)
    if data.shape[1] < 3:
        raise ValueError(f"Unexpected data format in {path}")

    return {
        "path": path,
        "header": header,
        "t": data[:, 0],
        "r": data[:, 1],
        "v": data[:, 2],
    }


def analytic_r(t, m, k, gamma, r0, v0):
    beta = gamma / (2.0 * m)
    omega2 = k / m - beta * beta
    if omega2 > 0.0:
        omega = math.sqrt(omega2)
        return np.exp(-beta * t) * (
            r0 * np.cos(omega * t) + (v0 + beta * r0) * np.sin(omega * t) / omega
        )
    if omega2 < 0.0:
        lamb = math.sqrt(-omega2)
        return np.exp(-beta * t) * (
            r0 * np.cosh(lamb * t) + (v0 + beta * r0) * np.sinh(lamb * t) / lamb
        )
    return np.exp(-beta * t) * (r0 + (v0 + beta * r0) * t)


def find_common_dt(records, target_dt):
    by_method = {method: [] for method in METHOD_ORDER}
    for rec in records:
        method = rec["header"].get("method", "").lower()
        if method in by_method:
            by_method[method].append(rec)

    for method, items in by_method.items():
        if not items:
            raise ValueError(f"Missing outputs for method {method}")

    dt_map = {
        method: sorted({parse_header_value(item["header"].get("dt", "nan")) for item in items})
        for method, items in by_method.items()
    }
    common = set(dt_map[METHOD_ORDER[0]])
    for method in METHOD_ORDER[1:]:
        common &= set(dt_map[method])

    if not common:
        raise ValueError("No common dt across methods. Provide --dt explicitly.")

    if target_dt is not None:
        if target_dt not in common:
            raise ValueError(f"Requested dt={target_dt} not found for all methods.")
        dt_choice = target_dt
    else:
        if 0.001 in common:
            dt_choice = 0.001
        else:
            dt_choice = min(common)

    chosen = {}
    for method in METHOD_ORDER:
        candidates = [
            item
            for item in by_method[method]
            if parse_header_value(item["header"].get("dt", "nan")) == dt_choice
        ]
        chosen[method] = candidates[0]

    return dt_choice, chosen


def compute_sample_dt(t, fallback):
    if t.size < 2:
        return fallback
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return fallback
    return float(np.median(diffs))


def compute_zoom_window(t, errors, tf, zoom_points, center_bias):
    if errors.size == 0:
        return 0.0, tf
    idx = int(np.argmax(np.abs(errors)))
    center_by_error = t[idx]
    center = center_by_error if center_bias is None else center_bias
    sample_dt = compute_sample_dt(t, tf)
    width = max(sample_dt * zoom_points, sample_dt)
    start = max(0.0, center - width / 2.0)
    end = min(tf, center + width / 2.0)
    if end - start < 1e-12:
        return 0.0, tf
    return start, end


def compute_zoom_ylim(analytic_t, analytic_r_values, per_method, zoom_methods, start, end, pad_frac):
    values = []
    mask = (analytic_t >= start) & (analytic_t <= end)
    if np.any(mask):
        values.append(analytic_r_values[mask])
    for method in zoom_methods:
        t = per_method[method]["t"]
        r = per_method[method]["r"]
        msk = (t >= start) & (t <= end)
        if np.any(msk):
            values.append(r[msk])
    if not values:
        return None
    all_values = np.concatenate(values)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    span = y_max - y_min
    pad = span * pad_frac if span > 0.0 else 1.0e-6
    return y_min - pad, y_max + pad


def apply_axis_format(ax):
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Posición (m)")
    ax.grid(True, alpha=0.2)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)


def add_external_legend(fig, ax):
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])


def main():
    parser = argparse.ArgumentParser(description="Plot analytic vs numerical oscillator solutions.")
    parser.add_argument("--input-dir", default="outputs/oscillatorOutputs", help="Input directory with oscillator outputs")
    parser.add_argument("--dt", type=float, default=None, help="Select outputs with this dt")
    parser.add_argument("--run", action="store_true", help="Run oscillator simulations before plotting")
    parser.add_argument("--run-force", action="store_true", help="When used with --run, overwrite existing outputs")
    parser.add_argument("--run-compile", action="store_true", help="When used with --run, compile Java sources before simulation")
    parser.add_argument("--java-cmd", default="java", help="Java executable")
    parser.add_argument("--javac-cmd", default="javac", help="javac executable")
    parser.add_argument("--run-dt", type=float, default=0.001, help="dt used by --run")
    parser.add_argument("--run-dt2", type=float, default=None, help="dt2 used by --run (default: run-dt)")
    parser.add_argument("--run-tf", type=float, default=5.0, help="tf used by --run")
    parser.add_argument("--run-m", type=float, default=70.0, help="mass used by --run")
    parser.add_argument("--run-k", type=float, default=1.0e4, help="spring constant used by --run")
    parser.add_argument("--run-gamma", type=float, default=100.0, help="damping used by --run")
    parser.add_argument("--run-r0", type=float, default=1.0, help="initial position used by --run")
    parser.add_argument("--run-v0", type=float, default=None, help="initial velocity used by --run")
    parser.add_argument("--zoom", action="store_true", help="Save zoomed figure")
    parser.add_argument("--full", action="store_true", help="Save full-range figure")
    parser.add_argument("--out-prefix", default="oscillator_comparison", help="Output file prefix")
    parser.add_argument("--zoom-exclude-euler", action="store_true", help="Exclude Euler from zoom plot")
    parser.add_argument("--zoom-start", type=float, default=None, help="Zoom window start time (s)")
    parser.add_argument("--zoom-end", type=float, default=None, help="Zoom window end time (s)")
    parser.add_argument("--zoom-center", type=float, default=None, help="Zoom window center time (s)")
    parser.add_argument("--zoom-window", type=float, default=None, help="Zoom window width (s)")
    parser.add_argument("--zoom-points", type=int, default=4, help="Zoom window width in sample steps")
    parser.add_argument("--zoom-center-bias", type=float, default=0.75, help="Default zoom center as fraction of tf")
    parser.add_argument("--zoom-ymin", type=float, default=None, help="Zoom Y min (m)")
    parser.add_argument("--zoom-ymax", type=float, default=None, help="Zoom Y max (m)")
    parser.add_argument("--zoom-pad", type=float, default=0.1, help="Padding fraction for auto Y limits")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = repo_root / out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = repo_root / input_dir

    if args.run:
        run_oscillator_outputs(repo_root, input_dir, args)
        if args.dt is None:
            args.dt = args.run_dt

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    records = [read_output(path) for path in input_dir.glob("*.txt")]
    if not records:
        raise FileNotFoundError(f"No output files found in {input_dir}")

    dt_choice, chosen = find_common_dt(records, args.dt)

    first = chosen[METHOD_ORDER[0]]
    header = first["header"]
    m = parse_header_value(header["m"])
    k = parse_header_value(header["k"])
    gamma = parse_header_value(header["gamma"])
    r0 = parse_header_value(header["r0"])
    v0 = parse_header_value(header["v0"])
    tf = parse_header_value(header["tf"])

    analytic_t = np.linspace(0.0, tf, 2000)
    analytic_r_values = analytic_r(analytic_t, m, k, gamma, r0, v0)

    per_method = {}
    all_errors = []
    for method, rec in chosen.items():
        t = rec["t"]
        r = rec["r"]
        r_ana = analytic_r(t, m, k, gamma, r0, v0)
        errors = r - r_ana
        ecm = float(np.mean(errors ** 2))
        per_method[method] = {"t": t, "r": r, "ecm": ecm, "errors": errors}
        all_errors.append(np.abs(errors))

    zoom_methods = METHOD_ORDER if not args.zoom_exclude_euler else METHOD_ORDER[1:]
    # Keep t and error arrays aligned: the zoom center must be computed from
    # one method's own time grid, not from concatenated errors of all methods.
    zoom_reference_method = zoom_methods[0]
    zoom_errors = np.abs(per_method[zoom_reference_method]["errors"])
    if (args.zoom_start is not None) ^ (args.zoom_end is not None):
        raise ValueError("Provide both --zoom-start and --zoom-end, or neither.")
    if (args.zoom_center is not None) ^ (args.zoom_window is not None):
        raise ValueError("Provide both --zoom-center and --zoom-window, or neither.")

    if args.zoom_start is not None and args.zoom_end is not None:
        zoom_start, zoom_end = args.zoom_start, args.zoom_end
    elif args.zoom_center is not None and args.zoom_window is not None:
        zoom_start = max(0.0, args.zoom_center - args.zoom_window / 2.0)
        zoom_end = min(tf, args.zoom_center + args.zoom_window / 2.0)
    else:
        center_bias = None
        if args.zoom_center_bias is not None:
            center_bias = max(0.0, min(tf, args.zoom_center_bias * tf))
        zoom_start, zoom_end = compute_zoom_window(
            per_method[zoom_reference_method]["t"],
            zoom_errors,
            tf,
            args.zoom_points,
            center_bias,
        )

    if zoom_end <= zoom_start:
        raise ValueError("Zoom window is empty. Check zoom parameters.")

    if args.zoom_ymin is not None and args.zoom_ymax is not None:
        zoom_ylim = (args.zoom_ymin, args.zoom_ymax)
    else:
        zoom_ylim = compute_zoom_ylim(
            analytic_t,
            analytic_r_values,
            per_method,
            zoom_methods,
            zoom_start,
            zoom_end,
            args.zoom_pad,
        )

    print(f"dt used: {dt_choice}")
    for method in METHOD_ORDER:
        ecm = per_method[method]["ecm"]
        print(f"ECM {method}: {ecm:.6e}")

    if args.full or not args.zoom:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(analytic_t, analytic_r_values, color=COLORS["analytic"], linewidth=2.0, label=LABELS["analytic"])
        for method in METHOD_ORDER:
            ax.plot(
                per_method[method]["t"],
                per_method[method]["r"],
                color=COLORS.get(method),
                label=LABELS.get(method, method),
            )
        apply_axis_format(ax)
        add_external_legend(fig, ax)
        fig.savefig(str(out_prefix.with_name(out_prefix.name + "_full.png")), dpi=200, bbox_inches="tight")
        plt.close(fig)

    if args.zoom or not args.full:
        fig, ax = plt.subplots(figsize=(9, 5))
        analytic_t_zoom = np.linspace(zoom_start, zoom_end, 2000)
        analytic_r_zoom = analytic_r(analytic_t_zoom, m, k, gamma, r0, v0)
        ax.plot(analytic_t_zoom, analytic_r_zoom, color=COLORS["analytic"], linewidth=2.0, label=LABELS["analytic"])
        for method in zoom_methods:
            t = per_method[method]["t"]
            r = per_method[method]["r"]
            msk = (t >= zoom_start) & (t <= zoom_end)
            ax.plot(t[msk], r[msk], color=COLORS.get(method), label=LABELS.get(method, method))
        ax.set_xlim(zoom_start, zoom_end)
        ax.margins(x=0.0, y=0.0)
        if zoom_ylim is not None:
            ax.set_ylim(zoom_ylim)
        apply_axis_format(ax)
        add_external_legend(fig, ax)
        fig.savefig(str(out_prefix.with_name(out_prefix.name + "_zoom.png")), dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
