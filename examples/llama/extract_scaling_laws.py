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




# ---------------------------------------------------------------------------
#  Directory name parser
# ---------------------------------------------------------------------------

# Pattern: llama3_C{compute}_D{tokens}_N{params}_h{hidden}_l{layers}_s{steps}_fp8
DIR_PATTERN = re.compile(
    r"llama3_C(?P<C>[0-9e.]+)_D(?P<D>\d+)B_N(?P<N>\d+)(?P<N_unit>[MBT]?)_"
    r"h(?P<h>\d+)_l(?P<l>\d+)_s(?P<s>[0-9e.]+)_fp8"
)


# ---------------------------------------------------------------------------
#  Sequence length (constant across all runs) and per-compute GBS lookup
# ---------------------------------------------------------------------------

# Llama-3 sequence length used in every training run.
SEQ_LEN = 8192

# Global Batch Size is fixed per compute budget (IsoFLOP sweep).
# Maps the C value parsed from the directory name → GBS (sequences per step).
GBS_BY_COMPUTE: dict[float, int] = {
    6e18:  32,
    1e19:  40,
    3e19:  56,
    6e19:  72,
    1e20:  96,
    3e20: 136,
    6e20: 192,
}


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


def tokens_from_dir(meta: dict, actual_step: int | None = None) -> float:
    """Compute precise token count as GBS * steps * SEQ_LEN.

    Uses *actual_step* (the last step logged to TensorBoard) when available,
    because it is exact.  Falls back to *expected_steps* parsed from the
    directory name (e.g. 's1.1e5' → 110000) which can be off by up to ~3%.
    Falls further back to the coarser D_B * 1e9 estimate with a warning
    when the compute budget is not in GBS_BY_COMPUTE.
    """
    gbs = GBS_BY_COMPUTE.get(meta["C"])
    if gbs is None:
        print(
            f"  [warn] C={meta['C']:.2e} not found in GBS_BY_COMPUTE; "
            f"falling back to D_B*1e9 estimate for '{meta['dirname']}'"
        )
        return meta["D_B"] * 1e9
    steps = actual_step if actual_step is not None else meta["expected_steps"]
    return gbs * steps * SEQ_LEN


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


def get_last_validation_loss(event_dir: str) -> tuple[float | None, int | None, float | None]:
    """
    Read TensorBoard events from *event_dir* and return
    (last_val_loss, step_at_last_val_loss, duration_hours).

    duration_hours is the wall clock time from the first to last logged event
    across all scalar tags (not just validation).

    Returns (None, None, None) if nothing is found.
    """
    ea = EventAccumulator(event_dir, size_guidance={"scalars": 0})
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])

    # --- Compute wall clock duration from the most-logged tag ---
    duration_hours = None
    best_tag = None
    best_count = 0
    for tag in available_tags:
        events = ea.Scalars(tag)
        if len(events) > best_count:
            best_count = len(events)
            best_tag = tag
    if best_tag and best_count >= 2:
        events = ea.Scalars(best_tag)
        duration_sec = events[-1].wall_time - events[0].wall_time
        duration_hours = duration_sec / 3600.0

    # --- Find validation loss ---
    for tag in VALIDATION_TAG_CANDIDATES:
        if tag in available_tags:
            events = ea.Scalars(tag)
            if events:
                last = events[-1]
                return last.value, last.step, duration_hours

    # Fuzzy match
    for tag in available_tags:
        if "validation" in tag.lower() and "loss" in tag.lower():
            events = ea.Scalars(tag)
            if events:
                last = events[-1]
                print(f"  [info] matched fallback tag: '{tag}' → loss={last.value:.4f} @ step {last.step}")
                return last.value, last.step, duration_hours

    for tag in available_tags:
        if "validation" in tag.lower():
            events = ea.Scalars(tag)
            if events:
                last = events[-1]
                print(f"  [info] matched loose tag: '{tag}' → loss={last.value:.4f} @ step {last.step}")
                return last.value, last.step, duration_hours

    return None, None, duration_hours


# ---------------------------------------------------------------------------
#  Curve fitting (Llama-3 style: parabola in log-token space)
# ---------------------------------------------------------------------------

def fit_isoflop_parabola(tokens_arr, loss_arr):
    """Fit a second-degree polynomial (parabola) to loss vs log(tokens).

    As described in the Llama-3 paper: "We fit the measured loss values using
    a second-degree polynomial and identify the minimums of each parabola."

    Returns (coeffs, D_opt, L_opt) where:
      - coeffs: polynomial coefficients [a, b, c] for a*x^2 + b*x + c
      - D_opt:  token count at the parabola minimum (compute-optimal)
      - L_opt:  predicted loss at D_opt
    Returns (None, None, None) if fitting fails.
    """
    if len(tokens_arr) < 3:
        return None, None, None
    try:
        log_tokens = np.log10(tokens_arr)
        # Fit: L(log10(D)) = a*(log10(D))^2 + b*(log10(D)) + c
        coeffs = np.polyfit(log_tokens, loss_arr, 2)
        a, b, c = coeffs

        # Minimum of parabola: x_min = -b / (2a)  (only valid if a > 0)
        if a <= 0:
            # Parabola opens downward — no minimum; fall back to data minimum
            min_idx = np.argmin(loss_arr)
            return coeffs, tokens_arr[min_idx], loss_arr[min_idx]

        log_D_opt = -b / (2 * a)
        D_opt = 10 ** log_D_opt
        L_opt = np.polyval(coeffs, log_D_opt)

        return coeffs, D_opt, L_opt
    except Exception as e:
        print(f"  [warn] parabola fit failed: {e}")
        return None, None, None


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


def make_plot(data_by_C: dict, output_path: str, incomplete_C_budgets: set | None = None):
    """
    Produce an IsoFLOP scaling law plot matching the Llama-3 paper (Figure 2).

    - X-axis: Training Tokens (log scale)
    - Y-axis: Validation Loss
    - All data points are plotted as same-colored circles per IsoFLOP curve.
    - A second-degree polynomial (parabola) is fitted in log-token space.
    - The minimum of each parabola is marked with a pink diamond (compute-optimal).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Sort compute budgets
    sorted_C = sorted(data_by_C.keys())
    n = len(sorted_C)

    # Blue gradient from light (low C) to dark (high C) — matching the Llama3 paper
    cmap = matplotlib.colormaps["Blues"]
    colors = [cmap(0.25 + 0.65 * i / max(n - 1, 1)) for i in range(n)]

    legend_entries = []

    # Collect compute-optimal points for Figure 3
    optimal_points = []  # list of (C, D_opt, L_opt)

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

        # Plot ALL data points as circles
        ax.scatter(tokens, losses, color=color, s=40, zorder=5, alpha=0.9)

        # Fit parabola in log-token space and plot smooth curve
        coeffs, D_opt, L_opt = fit_isoflop_parabola(tokens, losses)
        if coeffs is not None:
            log_D_dense = np.linspace(
                np.log10(tokens.min()) - 0.15,
                np.log10(tokens.max()) + 0.15,
                300,
            )
            L_dense = np.polyval(coeffs, log_D_dense)
            ax.plot(10**log_D_dense, L_dense, color=color, linewidth=2, alpha=0.8)

            # Mark the parabola minimum (compute-optimal point) with a diamond
            ax.scatter(
                [D_opt], [L_opt],
                color="#E91E63",  # Pink/magenta diamond like in the paper
                marker="D", s=80, zorder=10, edgecolors="white", linewidth=0.5,
            )
            # Only include in Figure 3 power law fit if ALL runs for this C are complete
            if incomplete_C_budgets is None or C not in incomplete_C_budgets:
                optimal_points.append((C, D_opt, L_opt))
            else:
                print(f"  [info] C={format_compute(C)}: optimal point excluded from power law fit (incomplete sweep)")
        else:
            # Fallback: just connect with lines
            ax.plot(tokens, losses, color=color, linewidth=2, alpha=0.8)

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
    print(f"\n✅ IsoFLOP plot saved to: {output_path}")
    plt.close(fig)

    return optimal_points


def make_optimal_tokens_plot(optimal_points: list, output_path: str):
    """
    Produce a two-panel figure:
      Left  — D*(C): optimal training tokens vs compute  (Figure 3 of Llama-3 paper)
      Right — N*(C): optimal model params vs compute     (N* = C / (6 · D*))

    Both panels show:
      - Your fitted power law   D*(C) = A · C^α  (blue solid)
      - Llama-3 paper reference D*(C) = 0.29 · C^0.537  (orange dashed)
      - Compute-optimal data points from parabola minima  (pink diamonds)
    """
    if len(optimal_points) < 2:
        print("\n⚠️  Need at least 2 compute-optimal points for Figure 3. Skipping.")
        return None

    C_arr = np.array([p[0] for p in optimal_points])
    D_arr = np.array([p[1] for p in optimal_points])

    # Sort by compute
    order = np.argsort(C_arr)
    C_arr = C_arr[order]
    D_arr = D_arr[order]

    # Optimal params from Chinchilla identity: N* = C / (6 · D*)
    N_arr = C_arr / (6.0 * D_arr)

    # Fit power law in log-log space:  log₁₀(D*) = α · log₁₀(C) + log₁₀(A)
    log_C = np.log10(C_arr)
    log_D = np.log10(D_arr)
    coeffs = np.polyfit(log_C, log_D, 1)  # [α, log₁₀(A)]
    alpha = coeffs[0]
    A = 10 ** coeffs[1]

    print(f"\n📐 Power law fit:  D*(C) = {A:.3f} · C^{alpha:.3f}")

    # Compute range for smooth curves — extend to the Llama-3 405B budget
    PLOT_C_MAX = 3.8e25   # Llama-3 paper's extrapolation target
    PLOT_C_MIN = C_arr.min() * 0.3
    C_dense = np.geomspace(PLOT_C_MIN, PLOT_C_MAX, 500)
    C_ref   = C_dense  # same range for both fits

    # Pre-compute all curve quantities
    D_fitted  = A      * np.power(C_dense, alpha)
    N_fitted  = C_dense / (6.0 * D_fitted)
    D_paper   = PAPER_A * np.power(C_ref,   PAPER_ALPHA)
    N_paper   = C_ref   / (6.0 * D_paper)

    # Ratio D*/N* = 6·A²·C^(2α−1)  — measures tokens per parameter
    R_arr    = D_arr   / N_arr                        # data points
    R_fitted = D_fitted / N_fitted                    # = 6·A²·C_dense^(2α−1)
    R_paper  = D_paper  / N_paper                     # = 6·PAPER_A²·C_ref^(2·PAPER_ALPHA−1)

    # --- Figure with three subplots ---
    fig, (ax_D, ax_N, ax_R) = plt.subplots(1, 3, figsize=(21, 6))

    DIAMOND_KW = dict(color="#E91E63", marker="D", s=80, zorder=10,
                      edgecolors="white", linewidth=0.5)
    SWEEP_KW   = dict(color="#1976D2", linewidth=2.5, alpha=0.9)
    PAPER_KW   = dict(color="#FF6F00", linewidth=2.0, linestyle="--", alpha=0.85)

    def _style(ax, xlabel, ylabel, title):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(PLOT_C_MIN, PLOT_C_MAX * 1.5)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(labelsize=11)
        # Vertical reference at Llama-3 405B compute
        ax.axvline(PLOT_C_MAX, color="#9E9E9E", linewidth=1.2, linestyle=":", zorder=1)
        ax.text(PLOT_C_MAX * 1.08, 0.02, "Llama-3 405B\n3.8e25 FLOPs",
                fontsize=8, color="#616161", va="bottom", ha="left", rotation=90,
                transform=ax.get_xaxis_transform())
        ax.legend(fontsize=10, loc="upper left", frameon=True,
                  framealpha=0.9, edgecolor="#cccccc")
        ax.grid(True, which="both", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    # ── Left panel: D*(C) ──────────────────────────────────────────────────
    ax_D.scatter(C_arr, D_arr, **DIAMOND_KW)
    ax_D.plot(C_dense, D_fitted, **SWEEP_KW,
              label=rf"This sweep, $\alpha={alpha:.3f}$, $A={A:.3f}$")
    ax_D.plot(C_ref, D_paper, **PAPER_KW,
              label=rf"Llama-3 paper, $\alpha={PAPER_ALPHA:.3f}$, $A={PAPER_A:.3f}$")
    _style(ax_D, "Compute (FLOPs)", "Optimal Training Tokens  D*(C)",
           "Compute-optimal token count")
    # Human-readable y-axis labels for token counts
    _tok_ticks  = [1e8,    1e9,  1e10,  1e11,   1e12, 1e13,  1e14]
    _tok_labels = ["100M", "1B", "10B", "100B", "1T", "10T", "100T"]
    ax_D.set_yticks(_tok_ticks)
    ax_D.set_yticklabels(_tok_labels, fontsize=11)
    ax_D.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # ── Middle panel: N*(C) = C / (6·D*) ──────────────────────────────────
    ax_N.scatter(C_arr, N_arr, **DIAMOND_KW)
    ax_N.plot(C_dense, N_fitted, **SWEEP_KW,
              label=rf"This sweep, $\alpha={alpha:.3f}$, $A={A:.3f}$")
    ax_N.plot(C_ref, N_paper, **PAPER_KW,
              label=rf"Llama-3 paper, $\alpha={PAPER_ALPHA:.3f}$, $A={PAPER_A:.3f}$")
    _style(ax_N, "Compute (FLOPs)", "Optimal Model Parameters  N*(C)",
           "Compute-optimal parameter count")
    # Human-readable y-axis labels for parameter counts
    _param_ticks  = [1e7,   1e8,    1e9,  1e10,  1e11,   1e12]
    _param_labels = ["10M", "100M", "1B", "10B", "100B",  "1T"]
    ax_N.set_yticks(_param_ticks)
    ax_N.set_yticklabels(_param_labels, fontsize=11)
    ax_N.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # ── Right panel: D*/N* ratio (tokens per parameter) ───────────────────
    # Chinchilla (Hoffmann et al. 2022) recommends ≈20 tokens per parameter.
    # D*/N* = 6·A²·C^(2α−1); if α=0.5 this is constant; it drifts otherwise.
    ax_R.scatter(C_arr, R_arr, **DIAMOND_KW, label="Data points")
    ax_R.plot(C_dense, R_fitted, **SWEEP_KW,
              label=rf"This sweep ($\alpha={alpha:.3f}$)")
    ax_R.plot(C_ref, R_paper, **PAPER_KW,
              label=rf"Llama-3 paper ($\alpha={PAPER_ALPHA:.3f}$)")
    # Chinchilla 20× reference
    ax_R.axhline(20, color="#43A047", linewidth=1.8, linestyle=":",
                 label="Chinchilla 20× rule")
    _style(ax_R, "Compute (FLOPs)", "D*(C) / N*(C)  [tokens per param]",
           "Token-to-parameter ratio")
    # Force y-axis to show the 20× line clearly
    all_R = np.concatenate([R_arr, R_fitted, R_paper])
    ax_R.set_ylim(max(1, all_R.min() * 0.5), all_R.max() * 2)

    fig.suptitle("Compute-optimal allocation  —  D*(C),  N*(C),  and D*/N* ratio",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"✅ Optimal allocation plot (3 panels) saved to: {output_path}")
    plt.close(fig)
    return alpha, A



# ---------------------------------------------------------------------------
#  Optimal allocation comparison table
# ---------------------------------------------------------------------------

# Llama-3 paper power-law coefficients (Figure 3):  D*(C) = A · C^α
PAPER_ALPHA = 0.537
PAPER_A     = 0.29  # tokens


def _fmt_tokens(n: float) -> str:
    """Format a token count as e.g. '15.3B' or '1.26T'."""
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    return f"{n/1e6:.0f}M"


def _fmt_params(n: float) -> str:
    """Format a parameter count as e.g. '7.3B' or '405M'."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    return f"{n/1e6:.0f}M"


def print_optimal_allocation_table(
    alpha: float,
    A: float,
    compute_budgets: list[float],
) -> None:
    """
    Print a side-by-side table of compute-optimal token count D*(C) and
    parameter count N*(C) predicted by:
      - the power law fitted from *this* sweep  (D* = A · C^α)
      - the original Llama-3 paper coefficients (D* = 0.29 · C^0.537)

    N* is derived via the Chinchilla identity:  N* = C / (6 · D*)
    """
    print("\n" + "=" * 88)
    print("  Compute-optimal allocation:  D*(C) = A·C^α,  N*(C) = C / (6·D*)")
    print(f"  Your fit   : α = {alpha:.3f}, A = {A:.4f}")
    print(f"  Llama-3 paper: α = {PAPER_ALPHA:.3f}, A = {PAPER_A:.4f}")
    print("=" * 88)
    hdr = (
        f"  {'C (FLOPs)':>12s}  "
        f"{'Yours D*':>12s}  {'Paper D*':>12s}  "
        f"{'Yours N*':>10s}  {'Paper N*':>10s}  "
        f"{'D* ratio':>9s}  {'N* ratio':>9s}"
    )
    print(hdr)
    print("  " + "-" * 84)
    for C in compute_budgets:
        D_yours = A * C ** alpha
        D_paper = PAPER_A * C ** PAPER_ALPHA
        N_yours = C / (6.0 * D_yours)
        N_paper = C / (6.0 * D_paper)
        ratio_D = D_yours / D_paper
        ratio_N = N_yours / N_paper
        print(
            f"  {format_compute(C):>12s}  "
            f"{_fmt_tokens(D_yours):>12s}  {_fmt_tokens(D_paper):>12s}  "
            f"{_fmt_params(N_yours):>10s}  {_fmt_params(N_paper):>10s}  "
            f"{ratio_D:>8.2f}x  {ratio_N:>8.2f}x"
        )
    print("=" * 88)


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
    data_by_C = defaultdict(list)
    print(f"{'Directory':<65s} {'C':>10s} {'Tokens':>12s} {'Params':>10s} {'Val Loss':>10s} {'Step':>8s} {'Expected':>8s} {'Duration':>10s} {'Status':>12s}")
    print("-" * 155)

    csv_lines = ["directory,compute_C,tokens_D,params_N,hidden,layers,val_loss,step,expected_steps,duration_hours,complete"]
    incomplete_count = 0
    incomplete_C_budgets = set()  # C values with any incomplete or missing runs

    for run in runs:
        loss, step, duration_hours = get_last_validation_loss(run["path"])
        # Use the actual TensorBoard step when available (most precise);
        # tokens_from_dir falls back to expected_steps from the dir name.
        tokens = tokens_from_dir(run, actual_step=step)
        expected = run["expected_steps"]

        # Check if the run is complete: last validation step must be
        # at least 95% of expected total steps (we use a percentage because
        # the directory name rounds the step count, e.g. s1.8e4 for 17619).
        is_complete = (
            loss is not None
            and step is not None
            and step >= expected * 0.95
        )

        if loss is not None and not is_complete:
            status_label = "INCOMPLETE"
            incomplete_count += 1
            incomplete_C_budgets.add(run["C"])
        elif loss is not None:
            status_label = "OK"
        else:
            status_label = "NO DATA"

        loss_str = f"{loss:.6f}" if loss is not None else "N/A"
        step_str = str(step) if step is not None else "N/A"
        if duration_hours is not None:
            if duration_hours < 1:
                dur_str = f"{duration_hours * 60:.0f}m"
            else:
                dur_str = f"{duration_hours:.1f}h"
        else:
            dur_str = "N/A"

        print(
            f"{run['dirname']:<65s} {format_compute(run['C']):>10s} "
            f"{tokens:>12.2e} {run['N']:>10.0f} {loss_str:>10s} {step_str:>8s} "
            f"{expected:>8d} {dur_str:>10s} {status_label:>12s}"
        )

        csv_lines.append(
            f"{run['dirname']},{run['C']:.2e},{tokens:.2e},{run['N']:.0f},"
            f"{run['hidden']},{run['layers']},"
            f"{loss if loss is not None else ''},"
            f"{step if step is not None else ''},"
            f"{expected},{duration_hours if duration_hours is not None else ''},"
            f"{is_complete}"
        )

        # Only include completed runs in the plot
        if is_complete:
            data_by_C[run["C"]].append({
                "tokens": tokens,
                "loss": loss,
                "N": run["N"],
                "dirname": run["dirname"],
                "duration_hours": duration_hours,
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

    if incomplete_C_budgets:
        print(f"\n⚠️  Compute budgets with incomplete sweeps (excluded from power law fit): "
              f"{", ".join(format_compute(c) for c in sorted(incomplete_C_budgets))}")

    print(f"\n📈 Plotting {total_points} data points across {len(data_by_C)} compute budgets...")
    optimal_points = make_plot(data_by_C, args.output, incomplete_C_budgets)

    # Figure 3: Compute vs Optimal Training Tokens (power law fit)
    fit_result = None
    if optimal_points:
        tokens_output = args.output.replace(".png", "_optimal_tokens.png")
        fit_result = make_optimal_tokens_plot(optimal_points, tokens_output)

    # Comparison table: your fit vs Llama-3 paper
    if fit_result is not None:
        alpha, A = fit_result
        # Show all sweep C budgets plus a few reference scales
        table_budgets = sorted(set(list(GBS_BY_COMPUTE.keys()) + [1e21, 3.8e25]))
        print_optimal_allocation_table(alpha, A, table_budgets)


if __name__ == "__main__":
    main()
