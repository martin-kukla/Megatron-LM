#!/usr/bin/env python3
"""
Extract last validation loss from TensorBoard event files for scaling law runs
and produce a Llama-3-style IsoFLOP scaling law plot.

Usage:
    python extract_scaling_laws.py [--tb-root ~/tensorboard_logs] [--output scaling_laws.png]

Requirements:
    pip install tensorboard matplotlib numpy scipy
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for remote machines
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit(
        "ERROR: tensorboard is not installed. Run:  pip install tensorboard"
    )

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found — will skip curve fitting. Install with: pip install scipy")


# ---------------------------------------------------------------------------
#  Directory name parser
# ---------------------------------------------------------------------------

# Pattern: llama3_C{compute}_D{tokens}_N{params}_h{hidden}_l{layers}_s{steps}_fp8
DIR_PATTERN = re.compile(
    r"llama3_C(?P<C>[0-9e.]+)_D(?P<D>\d+)B_N(?P<N>\d+)(?P<N_unit>[MBT]?)_"
    r"h(?P<h>\d+)_l(?P<l>\d+)_s(?P<s>[0-9e.]+)_fp8"
)


def parse_si(value_str: str, unit: str) -> float:
    """Convert e.g. '460' + 'M' -> 460e6."""
    val = float(value_str)
    multiplier = {"M": 1e6, "B": 1e9, "T": 1e12, "": 1e6}.get(unit, 1)
    return val * multiplier


def parse_dir_name(dirname: str) -> dict | None:
    """Parse run metadata from directory name."""
    m = DIR_PATTERN.search(dirname)
    if not m:
        return None
    return {
        "C": float(m.group("C")),         # Compute budget in FLOPs
        "D_B": int(m.group("D")),          # Data tokens in billions (raw int from name)
        "N": parse_si(m.group("N"), m.group("N_unit")),  # Model params
        "hidden": int(m.group("h")),
        "layers": int(m.group("l")),
        "steps_str": m.group("s"),
        "expected_steps": int(float(m.group("s"))),  # e.g. '4.3e3' → 4300
        "dirname": dirname,
    }


def tokens_from_dir(meta: dict) -> float:
    """Approximate tokens from the directory name D field (in billions)."""
    return meta["D_B"] * 1e9


# ---------------------------------------------------------------------------
#  TensorBoard extraction
# ---------------------------------------------------------------------------

# Megatron-LM logs validation loss under these typical tag names.
# We try several patterns because different Megatron versions may differ.
VALIDATION_TAG_CANDIDATES = [
    "lm loss validation",
    "lm-loss-validation/lm loss validation",
    "lm loss validation vs samples",
]


def get_last_validation_loss(event_dir: str) -> tuple[float | None, int | None]:
    """
    Read TensorBoard events from *event_dir* and return
    (last_val_loss, step_at_last_val_loss).

    Returns (None, None) if nothing is found.
    """
    # EventAccumulator can take a long time on huge dirs; size_guidance limits
    # in-memory scalars to the most recent 0 (= all).
    ea = EventAccumulator(event_dir, size_guidance={"scalars": 0})
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])

    for tag in VALIDATION_TAG_CANDIDATES:
        if tag in available_tags:
            events = ea.Scalars(tag)
            if events:
                last = events[-1]
                return last.value, last.step
    
    # If none of the candidate tags matched, try a fuzzy match
    for tag in available_tags:
        if "validation" in tag.lower() and "loss" in tag.lower():
            events = ea.Scalars(tag)
            if events:
                last = events[-1]
                print(f"  [info] matched fallback tag: '{tag}' → loss={last.value:.4f} @ step {last.step}")
                return last.value, last.step

    # If still nothing, try any tag with 'validation' in it
    for tag in available_tags:
        if "validation" in tag.lower():
            events = ea.Scalars(tag)
            if events:
                last = events[-1]
                print(f"  [info] matched loose tag: '{tag}' → loss={last.value:.4f} @ step {last.step}")
                return last.value, last.step

    return None, None


# ---------------------------------------------------------------------------
#  Curve fitting (Chinchilla-style power law)
# ---------------------------------------------------------------------------

def power_law(D, a, b, E):
    """L(D) = a / D^b + E   (IsoFLOP power law in tokens)."""
    return a / np.power(D, b) + E


def fit_isoflop_curve(tokens_arr, loss_arr):
    """Fit a power-law to (tokens, loss) data for a single IsoFLOP slice.
    Returns (popt, D_opt) where D_opt is the token count that minimises loss.
    """
    if not HAS_SCIPY or len(tokens_arr) < 3:
        return None, None
    try:
        popt, _ = curve_fit(
            power_law,
            tokens_arr,
            loss_arr,
            p0=[1.0, 0.5, 0.5],
            maxfev=20000,
            bounds=([0, 0, 0], [np.inf, 2.0, np.inf]),
        )
        # Optimal D is where dL/dD = 0 → D_opt is at the minimum of the fitted curve
        # Since L = a/D^b + E is monotonically decreasing in D, the minimum is at
        # D→∞.  In practice we want the *valley* on the plot — which means the
        # minimum among the data points, or we can evaluate over a dense grid.
        D_dense = np.geomspace(tokens_arr.min() * 0.5, tokens_arr.max() * 2, 500)
        L_dense = power_law(D_dense, *popt)
        D_opt = D_dense[np.argmin(L_dense)]
        return popt, D_opt
    except Exception as e:
        print(f"  [warn] curve_fit failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def format_compute(c: float) -> str:
    """Pretty-print compute budget, e.g. 6e18 → '6e18'."""
    exp = int(np.floor(np.log10(c)))
    mantissa = c / 10**exp
    if abs(mantissa - round(mantissa)) < 0.01:
        mantissa = int(round(mantissa))
    return f"{mantissa}e{exp}"


def make_plot(data_by_C: dict, output_path: str):
    """
    Produce an IsoFLOP scaling law plot similar to the Llama-3 paper.
    
    X-axis: Training Tokens (log scale)
    Y-axis: Validation Loss
    Each IsoFLOP line is a different color (light → dark blue gradient).
    Diamond markers show the optimal (minimum loss) point per IsoFLOP.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Sort compute budgets
    sorted_C = sorted(data_by_C.keys())
    n = len(sorted_C)

    # Blue gradient from light (low C) to dark (high C) — matching the Llama3 paper
    cmap = matplotlib.colormaps["Blues"]
    colors = [cmap(0.25 + 0.65 * i / max(n - 1, 1)) for i in range(n)]

    legend_entries = []

    for idx, C in enumerate(sorted_C):
        runs = data_by_C[C]
        tokens = np.array([r["tokens"] for r in runs])
        losses = np.array([r["loss"] for r in runs])
        
        # Sort by tokens
        order = np.argsort(tokens)
        tokens = tokens[order]
        losses = losses[order]

        color = colors[idx]
        label = format_compute(C)

        # Plot data points
        ax.scatter(tokens, losses, color=color, s=40, zorder=5, alpha=0.9)

        # Fit and plot smooth curve
        popt, D_opt = fit_isoflop_curve(tokens, losses)
        if popt is not None:
            D_dense = np.geomspace(tokens.min() * 0.8, tokens.max() * 1.2, 300)
            L_dense = power_law(D_dense, *popt)
            ax.plot(D_dense, L_dense, color=color, linewidth=2, alpha=0.8)
            
            # Mark the optimal point (minimum loss) with a diamond
            min_idx = np.argmin(losses)
            ax.scatter(
                [tokens[min_idx]], [losses[min_idx]],
                color="#E91E63",  # Pink/magenta diamond like in the paper
                marker="D", s=80, zorder=10, edgecolors="white", linewidth=0.5,
            )
        else:
            # Just connect with lines if curve fitting failed
            ax.plot(tokens, losses, color=color, linewidth=2, alpha=0.8)
            # Still mark minimum
            min_idx = np.argmin(losses)
            ax.scatter(
                [tokens[min_idx]], [losses[min_idx]],
                color="#E91E63",
                marker="D", s=80, zorder=10, edgecolors="white", linewidth=0.5,
            )

        legend_entries.append((color, label))

    # X-axis log scale
    ax.set_xscale("log")
    ax.set_xlabel("Training Tokens", fontsize=14)
    ax.set_ylabel("Validation Loss", fontsize=14)
    ax.tick_params(labelsize=12)

    # Legend (manual, matching the paper style)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=c, linewidth=3, label=l) for c, l in legend_entries
    ]
    legend = ax.legend(
        handles=legend_handles,
        title="Compute",
        title_fontsize=12,
        fontsize=11,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )
    legend.get_title().set_fontweight("bold")

    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\n✅ Plot saved to: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract validation losses from TensorBoard and plot scaling laws."
    )
    parser.add_argument(
        "--tb-root",
        default=os.path.expanduser("~/tensorboard_logs"),
        help="Root directory containing per-run TensorBoard subdirectories.",
    )
    parser.add_argument(
        "--output",
        default="scaling_laws.png",
        help="Output path for the scaling law plot.",
    )
    parser.add_argument(
        "--csv",
        default="scaling_laws_data.csv",
        help="Output path for the extracted data CSV.",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="Just list available TensorBoard scalar tags for each run and exit.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.tb_root):
        sys.exit(f"ERROR: TensorBoard root directory not found: {args.tb_root}")

    # Discover runs
    subdirs = sorted(os.listdir(args.tb_root))
    runs = []
    for d in subdirs:
        meta = parse_dir_name(d)
        if meta is None:
            print(f"  [skip] Cannot parse directory name: {d}")
            continue
        meta["path"] = os.path.join(args.tb_root, d)
        runs.append(meta)

    if not runs:
        sys.exit("ERROR: No matching run directories found.")

    print(f"Found {len(runs)} run directories.\n")

    if args.list_tags:
        for run in runs:
            print(f"=== {run['dirname']} ===")
            ea = EventAccumulator(run["path"], size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            for t in sorted(tags):
                print(f"  {t}")
            print()
        return

    # Extract validation losses
    EVAL_INTERVAL = 100  # from --eval-interval in training script
    data_by_C = defaultdict(list)
    print(f"{'Directory':<65s} {'C':>10s} {'Tokens':>12s} {'Params':>10s} {'Val Loss':>10s} {'Step':>8s} {'Expected':>8s} {'Status':>12s}")
    print("-" * 140)

    csv_lines = ["directory,compute_C,tokens_D,params_N,hidden,layers,val_loss,step,expected_steps,complete"]
    incomplete_count = 0

    for run in runs:
        loss, step = get_last_validation_loss(run["path"])
        tokens = tokens_from_dir(run)
        expected = run["expected_steps"]

        # Check if the run is complete: last validation step must be
        # within one eval interval of the expected total steps.
        is_complete = (
            loss is not None
            and step is not None
            and step >= expected - EVAL_INTERVAL
        )

        if loss is not None and not is_complete:
            status_label = "INCOMPLETE"
            incomplete_count += 1
        elif loss is not None:
            status_label = "OK"
        else:
            status_label = "NO DATA"

        loss_str = f"{loss:.6f}" if loss is not None else "N/A"
        step_str = str(step) if step is not None else "N/A"

        print(
            f"{run['dirname']:<65s} {format_compute(run['C']):>10s} "
            f"{tokens:>12.2e} {run['N']:>10.0f} {loss_str:>10s} {step_str:>8s} "
            f"{expected:>8d} {status_label:>12s}"
        )

        csv_lines.append(
            f"{run['dirname']},{run['C']:.2e},{tokens:.2e},{run['N']:.0f},"
            f"{run['hidden']},{run['layers']},"
            f"{loss if loss is not None else ''},"
            f"{step if step is not None else ''},"
            f"{expected},{is_complete}"
        )

        # Only include completed runs in the plot
        if is_complete:
            data_by_C[run["C"]].append({
                "tokens": tokens,
                "loss": loss,
                "N": run["N"],
                "dirname": run["dirname"],
            })

    if incomplete_count > 0:
        print(f"\n⚠️  {incomplete_count} run(s) marked INCOMPLETE (last step far from expected total).")

    # Save CSV
    with open(args.csv, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\n📊 Data saved to: {args.csv}")

    # Check we have data to plot
    total_points = sum(len(v) for v in data_by_C.values())
    if total_points == 0:
        print("\n⚠️  No validation losses found. Runs may still be in progress.")
        print("   Try --list-tags to see what TensorBoard tags are available.")
        return

    print(f"\n📈 Plotting {total_points} data points across {len(data_by_C)} compute budgets...")
    make_plot(data_by_C, args.output)


if __name__ == "__main__":
    main()
