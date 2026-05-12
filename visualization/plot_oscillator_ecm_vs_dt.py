import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation

METHOD_ORDER = ["euler", "verlet", "beeman", "gear5"]
LABELS = {
    "euler": "Euler",
    "verlet": "Verlet",
    "beeman": "Beeman",
    "gear5": "Gear 5",
}
MARKERS = {
    "euler": "o",
    "verlet": "s",
    "beeman": "^",
    "gear5": "D",
}


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


def apply_axis_format(ax, xticks):
    ax.set_xlabel("Paso temporal dt (s)")
    ax.set_ylabel("ECM posición (m^2)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax.invert_xaxis()


def main():
    parser = argparse.ArgumentParser(description="Plot ECM vs dt for oscillator outputs.")
    parser.add_argument("--input-dir", default="outputs/oscillatorOutputs", help="Input directory with oscillator outputs")
    parser.add_argument("--out", default="oscillator_ecm_vs_dt.png", help="Output image path")
    parser.add_argument("--common-only", action="store_true", help="Use only dt values present for all methods")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    records = [read_output(path) for path in input_dir.glob("*.txt")]
    if not records:
        raise FileNotFoundError(f"No output files found in {input_dir}")

    grouped = {method: {} for method in METHOD_ORDER}
    dt_sets = {method: set() for method in METHOD_ORDER}

    for rec in records:
        header = rec["header"]
        method = header.get("method", "").lower()
        if method not in grouped:
            continue
        dt = parse_header_value(header.get("dt", "nan"))
        dt2 = parse_header_value(header.get("dt2", str(dt)))
        if not math.isfinite(dt):
            continue

        m = parse_header_value(header["m"])
        k = parse_header_value(header["k"])
        gamma = parse_header_value(header["gamma"])
        r0 = parse_header_value(header["r0"])
        v0 = parse_header_value(header["v0"])

        t = rec["t"]
        r = rec["r"]
        r_ana = analytic_r(t, m, k, gamma, r0, v0)
        errors = r - r_ana
        ecm = float(np.mean(errors ** 2))

        grouped[method].setdefault(dt, []).append(ecm)
        dt_sets[method].add(dt)

        if abs(dt2 - dt) > 1e-12:
            print(f"Warning: dt2 != dt for {rec['path']} (dt={dt}, dt2={dt2}). ECM uses output times only.")

    if args.common_only:
        common = set.intersection(*(dt_sets[method] for method in METHOD_ORDER if dt_sets[method]))
        if not common:
            raise ValueError("No common dt values across methods.")
        for method in METHOD_ORDER:
            grouped[method] = {dt: grouped[method][dt] for dt in grouped[method] if dt in common}

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    all_dts = set()

    for method in METHOD_ORDER:
        entries = grouped[method]
        if not entries:
            continue
        dts = sorted(entries.keys(), reverse=True)
        ecms = [float(np.mean(entries[dt])) for dt in dts]
        ax.plot(dts, ecms, marker=MARKERS.get(method, "o"), label=LABELS.get(method, method))
        all_dts.update(dts)

        for dt, ecm in zip(dts, ecms):
            print(f"{method} dt={dt:.6g} ECM={ecm:.6e}")

    xticks = sorted(all_dts, reverse=True)
    apply_axis_format(ax, xticks)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
