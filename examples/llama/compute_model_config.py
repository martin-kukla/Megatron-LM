#!/usr/bin/env python3
"""
Given a FLOPs budget, compute valid (HIDDEN_SIZE, NUM_LAYERS) configurations
for a LLaMA3-style architecture across a sweep of training token budgets.

Uses the standard approximation: C ≈ 6 × N × D
  where N = non-embedding params, D = total tokens processed.

Architecture conventions (matching train_llama3_scaling_laws_b200_fp8.sh):
  - head_dim = 128, GQA ratio = 4:1
  - ffn_hidden = 3.5 × hidden_size (SwiGLU)
  - seq_length = 8192

Hyperparameter strategy (LLaMA 3 scaling law paper):
  - GBS: fixed per compute scale (250K–4M tokens/batch for C from 6e18 to 1e22).
    Specified via --gbs. If omitted, a scale-appropriate default is printed.
  - Peak LR: interpolated between 4e-4 (for 40M params) and 2e-4 (for 16B params),
    depending on the target model size N. Smaller model → higher LR.
    LR = 4e-4 * (N / 40e6)^(-log(2)/log(400))  [log-linear interpolation]
  - Cosine decay to 0.1 × peak, weight decay = 0.1 × LR at each step.
  - LR warmup: 2000 steps (fixed, as in LLaMA 3).

Sweep logic:
  C and D are the primary inputs. For each sweep point:
    N = C / (6D)   →  architecture
    GBS (fixed)    →  S = D / (GBS × seq_len)   [steps derived, not given]
    LR from N      →  printed as suggestion

Usage:
    # Sweep D around Chinchilla optimum (recommended):
    python3 examples/llama/compute_model_config.py --flops 1e19 --gbs 128
    # Single token budget:
    python3 examples/llama/compute_model_config.py --flops 1e19 --tokens 5e9 --gbs 128
    # Manual LR override:
    python3 examples/llama/compute_model_config.py --flops 1e19 --gbs 128 --base-lr 3e-4

References:
  [1] Chinchilla: Hoffmann et al. 2022 (D ≈ 20N for compute-optimal)
  [2] LLaMA 3: Dubey et al. 2024 (scaling law experiments, Sec. 3.1)
"""

import argparse
import math

# ─────────────────────────────────────────────────────────────────────────────
# LLaMA3 architecture constants
# ─────────────────────────────────────────────────────────────────────────────
HEAD_DIM = 128
GQA_RATIO = 4
FFN_MULTIPLIER = 3.5  # ffn_hidden_size = 3.5 × hidden_size
SEQ_LENGTH = 8192
VOCAB_SIZE = 128256
HIDDEN_SIZE_STEP = HEAD_DIM * GQA_RATIO  # 512, minimum granularity

# ─────────────────────────────────────────────────────────────────────────────
# LLaMA 3 peak LR range (from paper, Section 3.1)
# "The peak learning rate is set between 2×10⁻⁴ and 4×10⁻⁴ depending on the
#  size of the model."  Model sizes range from 40M to 16B.
# We use log-linear interpolation in N between the two anchor points.
# ─────────────────────────────────────────────────────────────────────────────
LR_N_SMALL  = 40e6    # 40M params  → peak LR = 4e-4
LR_N_LARGE  = 16e9    # 16B params  → peak LR = 2e-4
LR_AT_SMALL = 4e-4
LR_AT_LARGE = 2e-4

# ─────────────────────────────────────────────────────────────────────────────
# LLaMA 3 fixed-batch-size guide (from paper, Section 3.1)
# "fixed batch size for each compute scale, ranging between 250K and 4M [tokens]"
# Shown here for reference when choosing --gbs.
# ─────────────────────────────────────────────────────────────────────────────
LLAMA3_GBS_GUIDE = {
    6e18:  ("250K", 32),    # GBS 32 @ seq 8192 ≈ 262K tokens
    1e19:  ("300K", 40),
    3e19:  ("450K", 56),
    6e19:  ("600K", 72),
    1e20:  ("750K", 96),
    3e20:  ("1.1M", 136),
    6e20:  ("1.5M", 192),
    1e21:  ("2M",   256),
    3e21:  ("3M",   384),
    1e22:  ("4M",   512),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: LLaMA 3 peak LR from model size N
# ─────────────────────────────────────────────────────────────────────────────
def llama3_peak_lr(N):
    """
    Log-linear interpolation of LLaMA 3 peak LR based on model size N.

    Source: LLaMA 3 paper, Section 3.1:
      "Peak LR is set between 2×10⁻⁴ and 4×10⁻⁴ depending on model size."
      Range: 40M params → 4e-4, 16B params → 2e-4.

    Formula: log(LR) = log(LR_small) + slope × (log(N) - log(N_small))
      slope = (log(LR_large) - log(LR_small)) / (log(N_large) - log(N_small))

    Extrapolation (outside 40M–16B) is clamped to [LR_AT_LARGE, LR_AT_SMALL].
    """
    log_lr = math.log(LR_AT_SMALL) + (
        (math.log(LR_AT_LARGE) - math.log(LR_AT_SMALL))
        / (math.log(LR_N_LARGE) - math.log(LR_N_SMALL))
        * (math.log(N) - math.log(LR_N_SMALL))
    )
    lr = math.exp(log_lr)
    # Clamp: outside the anchor range, do not extrapolate wildly
    return max(LR_AT_LARGE, min(LR_AT_SMALL, lr))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: suggest GBS for a FLOPs budget
# ─────────────────────────────────────────────────────────────────────────────
def suggest_gbs(C, seq_length):
    """Return the LLaMA 3 suggested GBS for compute budget C."""
    keys = sorted(LLAMA3_GBS_GUIDE.keys())
    if C <= keys[0]:
        return LLAMA3_GBS_GUIDE[keys[0]][1]
    if C >= keys[-1]:
        return LLAMA3_GBS_GUIDE[keys[-1]][1]
    # Linear interpolation between nearest anchors in log space
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= C <= hi:
            t = (math.log(C) - math.log(lo)) / (math.log(hi) - math.log(lo))
            gbs_lo = LLAMA3_GBS_GUIDE[lo][1]
            gbs_hi = LLAMA3_GBS_GUIDE[hi][1]
            gbs_exact = gbs_lo + t * (gbs_hi - gbs_lo)
            return max(8, int(round(gbs_exact / 8)) * 8)
    return 128  # fallback


# ─────────────────────────────────────────────────────────────────────────────
# Architecture param counting
# ─────────────────────────────────────────────────────────────────────────────
def params_per_layer(d):
    """Non-embedding params per transformer layer for LLaMA3-style arch."""
    attn = d * d * (2 + 2 / GQA_RATIO)     # Q, K, V, O projections
    mlp  = 3 * d * (d * FFN_MULTIPLIER)    # SwiGLU gate + up + down
    norm = 2 * d                            # pre-attn + pre-MLP RMSNorm
    return attn + mlp + norm


def total_non_embedding_params(d, L):
    """Total non-embedding params (what enters the 6ND formula)."""
    return L * params_per_layer(d)


def total_params(d, L):
    """Total params including embeddings (untied input + output)."""
    non_emb = total_non_embedding_params(d, L)
    emb      = 2 * VOCAB_SIZE * d  # untied embeddings
    final_n  = d                   # final RMSNorm
    return non_emb + emb + final_n


def format_params(n):
    """Human-readable parameter count."""
    if n >= 1e9:   return f"{n / 1e9:.2f}B"
    if n >= 1e6:   return f"{n / 1e6:.1f}M"
    if n >= 1e3:   return f"{n / 1e3:.1f}K"
    return str(int(n))


def fmt_sci(x):
    """Compact scientific notation: 10000 → 1e4, 15000 → 1.5e4."""
    return f"{x:.1e}".replace("+0", "").replace("+", "").replace(".0e", "e")


def fmt_params_int(n):
    """Human-readable count without decimals: 708.9M → 708M, 5.3B → 5B."""
    if n >= 1e9:   return f"{int(n / 1e9)}B"
    if n >= 1e6:   return f"{int(n / 1e6)}M"
    if n >= 1e3:   return f"{int(n / 1e3)}K"
    return str(int(n))


# ─────────────────────────────────────────────────────────────────────────────
# Core: compute and print table for one (C, D) point
# ─────────────────────────────────────────────────────────────────────────────
def compute_and_print_table(C, D, gbs, args):
    """
    Causal flow:
      C, D  →  N = C / (6D)      [target model size]
      GBS   →  S = D / (GBS × L) [steps, derived not given]
      N     →  LR                 [LLaMA 3 log-linear interpolation]
    """
    # Step 1: model size
    N_target = C / (6 * D)

    # Step 2: steps (GBS is fixed per compute scale, passed in)
    S = max(1, int(round(D / (gbs * args.seq_length))))
    TPP = D / N_target

    # Print header
    print()
    print(f"  FLOPs budget (C):  {C:.2e}")
    print(f"  Total tokens (D):  {D:.2e}  (TPP = D/N = {TPP:.1f},  Chinchilla opt = 20)")
    print(f"  Target N (C/6D):   {N_target:.2e}  ({format_params(N_target)})")
    print(f"  GBS:               {gbs}  [fixed per compute scale]")
    print(f"  Tokens/step:       {gbs} × {args.seq_length:,} = {gbs * args.seq_length:,}")
    print(f"  Total steps (S):   {S:,}  [derived: D / (GBS × seq_len)]")
    print(f"  Reference W/D:     128.0  (LLaMA 3 8B = 4096 / 32)")
    print()

    # Architecture table
    header = (
        f"{'HIDDEN':>8}  {'LAYERS':>6}  {'W/D':>6}  {'N (non-emb)':>14}  {'N (total)':>14}  "
        f"{'Actual C':>12}  {'C/target':>8}  {'heads':>5}  {'GQA':>4}  {'FFN':>7}"
    )
    print(header)
    print("─" * len(header))

    results = []
    for d in range(args.min_hidden, args.max_hidden + 1, HIDDEN_SIZE_STEP):
        ppl     = params_per_layer(d)
        L       = round(N_target / ppl)
        if L < args.min_layers or L > args.max_layers:
            continue

        N_actual = total_non_embedding_params(d, L)
        N_total  = total_params(d, L)
        C_actual = 6 * N_actual * D
        C_ratio  = C_actual / C
        num_heads       = d // HEAD_DIM
        num_query_groups = max(1, num_heads // GQA_RATIO)
        ffn             = int(d * FFN_MULTIPLIER)
        aspect_ratio    = d / L if L > 0 else 0

        results.append(
            (d, L, aspect_ratio, N_actual, N_total, C_actual, C_ratio,
             num_heads, num_query_groups, ffn)
        )

    # Sort: within 10% FLOPs budget first, then by closest W/D to 128
    results.sort(key=lambda x: (abs(1.0 - x[6]) > 0.10, abs(128 - x[2]), abs(1.0 - x[6])))

    for d, L, ar, N_actual, N_total, C_actual, C_ratio, nh, gqa_g, ffn in results:
        marker = " ◄" if abs(C_ratio - 1.0) <= 0.10 else ""
        print(
            f"{d:>8}  {L:>6}  {ar:>6.1f}  {N_actual:>14,.0f}  {N_total:>14,.0f}  "
            f"{C_actual:>12.2e}  {C_ratio:>7.1%}{marker:>3}  {nh:>5}  {gqa_g:>4}  {ffn:>7}"
        )

    if not results:
        print("  No valid configurations found in the given range.")
        print(f"  Target N={format_params(N_target)} may be too small or too large.")
        print(f"  Try adjusting --min-layers/--max-layers or --min-hidden/--max-hidden.")
        return

    # Best match
    best       = results[0]
    hidden_size = best[0]
    num_layers  = best[1]
    best_N      = best[3]   # N_actual (non-embedding params)

    print(
        f"\n✓ Best match (W/D closest to 128): HIDDEN_SIZE={hidden_size}  NUM_LAYERS={num_layers}  "
        f"({format_params(best_N)} non-emb params, "
        f"actual C={best[5]:.2e}, {best[6]:.1%} of budget)"
    )

    # Peak LR: LLaMA 3 log-linear interpolation from model size
    if args.base_lr is not None:
        peak_lr   = args.base_lr
        lr_source = "user-specified"
    else:
        peak_lr   = llama3_peak_lr(best_N)
        lr_source = f"LLaMA3 log-linear: 4e-4 @ 40M → 2e-4 @ 16B (N={format_params(best_N)})"

    print(f"  Peak LR (η):       {peak_lr:.2e}  [{lr_source}]")
    print(f"  Min LR:            {peak_lr * 0.1:.2e}  [0.1 × peak, cosine decay]")
    print(f"  Weight decay:      0.1  [= 0.1 × LR at each step, LLaMA 3 convention]")
    print(f"  LR warmup:         2000 steps  [fixed, LLaMA 3 convention]")

    # Build command
    s_str = fmt_sci(S)
    c_str = fmt_sci(C)
    # Integer-only human-readable suffixes for directory names (e.g. 708M not 708.9M)
    d_str = fmt_params_int(D)
    n_str = fmt_params_int(best_N)

    run_tag  = f"C{c_str}_D{d_str}_N{n_str}"
    arch_tag = f"h{hidden_size}_l{num_layers}_s{s_str}"
    ckpt_dir = f"~/checkpoints/llama3_{run_tag}_{arch_tag}_fp8"
    tb_dir   = f"~/tensorboard_logs/llama3_{run_tag}_{arch_tag}_fp8"

    print(f"\n  # D={fmt_sci(D)} tokens  →  GBS={gbs}  →  S={S:,} steps")
    print(
        f"  PEAK_LR={peak_lr:.2e} GLOBAL_BATCH_SIZE={gbs} "
        f"HIDDEN_SIZE={hidden_size} NUM_LAYERS={num_layers} TOTAL_STEPS={S} \\"
    )
    print(f"    ./examples/llama/train_llama3_scaling_laws_b200_fp8.sh \\")
    print(f"        {ckpt_dir} \\")
    print(f"        {tb_dir} \\")
    print(f"        meta-llama/Meta-Llama-3-8B \\")
    print(f"        ~/pile_tokenized/pile_llama3_text_document")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compute (HIDDEN_SIZE, NUM_LAYERS) sweep for LLaMA3-style scaling laws",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--flops", type=float, required=True,
        help="Total FLOPs budget C (e.g., 1e19)",
    )
    parser.add_argument(
        "--tokens", type=float, default=None,
        help=(
            "Total training tokens D. "
            "If omitted, sweeps D over [0.25x, 0.5x, 1x, 2x, 4x] the Chinchilla-optimal D."
        ),
    )
    parser.add_argument(
        "--gbs", type=int, default=None,
        help=(
            "Global batch size (sequences/step), FIXED for the entire sweep. "
            "If omitted, the LLaMA 3 suggested value for your FLOPs budget is printed and used."
        ),
    )
    parser.add_argument(
        "--seq-length", type=int, default=SEQ_LENGTH,
        help=f"Sequence length in tokens (default: {SEQ_LENGTH})",
    )
    parser.add_argument(
        "--base-lr", type=float, default=None,
        help="Override peak LR instead of computing from LLaMA 3 model-size interpolation.",
    )
    parser.add_argument(
        "--min-layers", type=int, default=2,   help="Minimum number of layers (default: 2)"
    )
    parser.add_argument(
        "--max-layers", type=int, default=64,  help="Maximum number of layers (default: 64)"
    )
    parser.add_argument(
        "--min-hidden", type=int, default=512, help="Minimum hidden size (default: 512)"
    )
    parser.add_argument(
        "--max-hidden", type=int, default=8192, help="Maximum hidden size (default: 8192)"
    )
    args = parser.parse_args()

    # Chinchilla-optimal D: C = 6ND, D = 20N  →  D_opt = sqrt(C × 10/3)
    D_opt = math.sqrt(args.flops * 10.0 / 3.0)
    N_opt = D_opt / 20.0

    # Fixed GBS for this compute scale
    if args.gbs is not None:
        gbs = args.gbs
        gbs_source = "user-specified"
    else:
        gbs = suggest_gbs(args.flops, args.seq_length)
        gbs_source = f"LLaMA 3 suggested for C={args.flops:.1e}"

    if args.tokens is not None:
        compute_and_print_table(args.flops, args.tokens, gbs, args)
    else:
        print(f"\n{'='*70}")
        print(f" SWEEP MODE: Chinchilla-based scaling law grid")
        print(f" C = {args.flops:.2e} FLOPs")
        print(f" Chinchilla-optimal D = {D_opt:.2e} tokens  (D = sqrt(C × 10/3))")
        print(f" Chinchilla-optimal N = {format_params(N_opt)}  (N = D / 20)")
        print(f" GBS = {gbs}  [{gbs_source}]  ← fixed for ALL sweep points")
        print(f" LR: {'user-specified=' + str(args.base_lr) if args.base_lr else 'LLaMA 3 log-linear from N: 4e-4 @ 40M → 2e-4 @ 16B'}")
        print(f" NOTE: S (steps) is derived per point as D / (GBS × seq_len)")
        print()
        print(f" LLaMA 3 GBS reference table (tokens/batch → suggested GBS @ seq={args.seq_length}):")
        for C_ref, (tok_str, gbs_ref) in sorted(LLAMA3_GBS_GUIDE.items()):
            marker = " ◄ your scale" if C_ref == args.flops else ""
            print(f"   C={C_ref:.0e}  →  {tok_str} tokens/batch  →  GBS={gbs_ref}{marker}")
        print(f"{'='*70}")

        # Sweep D around the Chinchilla optimum; GBS is FIXED
        multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
        for mult in multipliers:
            D = D_opt * mult
            print(f"\n\n{'─'*70}")
            print(f" SWEEP: {mult:.2f}x Chinchilla-optimal tokens  →  D = {D:.2e}")
            print(f"{'─'*70}")
            compute_and_print_table(args.flops, D, gbs, args)


if __name__ == "__main__":
    main()
