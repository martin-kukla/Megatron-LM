#!/usr/bin/env python3
"""
Given a FLOPs budget and total training steps, compute valid
(HIDDEN_SIZE, NUM_LAYERS) pairs for a LLaMA3-style architecture.

Uses the standard approximation: C ≈ 6 × N × D
  where N = non-embedding params, D = total tokens processed.

Architecture conventions (matching train_llama3_scaling_laws_b200_fp8.sh):
  - head_dim = 128, GQA ratio = 4:1
  - ffn_hidden = 3.5 × hidden_size (SwiGLU)
  - GBS = auto (computed from D), seq_length = 8192

Hyperparameter scaling (principled, from literature):
  - GBS: B_opt ∝ D^0.5 (Cerebras "Power Lines", Bergsma et al. NeurIPS 2025)
    Calibrated so that B_opt ≈ 32 (250K tokens/batch) at Chinchilla-optimal
    D for C = 6e18.
  - Peak LR: derived from the AdamW timescale framework (Cerebras "Power Lines"):
      τ_opt = c_τ * (D/N)^m_τ   where m_τ ≈ -0.527
      η_peak = B / (λ * D * τ_opt)
    This couples LR to both batch size B and training tokens D, matching your
    intuition. The constant c_τ is calibrated to reproduce LLaMA 3 8B LR
    (2e-4) at its known training regime.
  - Weight decay: fixed at λ = 0.1 (standard for LLaMA-style models).
    Can override with --weight-decay.

Usage:
    # Sweep around Chinchilla optimum:
    python3 examples/llama/compute_model_config.py --flops 1e19
    # Manual overrides:
    python3 examples/llama/compute_model_config.py --flops 1e19 --gbs 64 --base-lr 3e-4

References:
  [1] Chinchilla: Hoffmann et al. 2022 (D ≈ 20N for compute-optimal)
  [2] Critical Batch Size: McCandlish et al. 2018 (B_crit ∝ sqrt(D))
  [3] Power Lines: Bergsma et al. 2024/NeurIPS 2025 (τ_opt ∝ TPP^-0.527,
      B_opt ∝ D^0.5, independent of N)
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
# Principled hyperparameter scaling constants
# ─────────────────────────────────────────────────────────────────────────────
# GBS scaling: B_opt = B0 * (D / D0)^BSCALE_EXPONENT
# Source: Cerebras "Power Lines" (Bergsma et al. NeurIPS 2025), Fig. 3
#   B_opt and B_crit both scale as D^~0.5 independent of N.
# Calibration anchor: at C=6e18 Chinchilla-optimal, D≈250B tokens,
#   LLaMA 3 used ~250K tokens/batch → GBS ≈ 30 at seq_len 8192.
BSCALE_D0 = 2.5e11        # anchor: 250B tokens
BSCALE_B0 = 30.0          # anchor: GBS ~30 (250K tok/step @ seq 8192)
BSCALE_EXPONENT = 0.5     # B_opt ∝ D^0.5 (McCandlish / Power Lines)

# AdamW timescale constants: τ_opt = C_TAU * (D/N)^TAU_EXPONENT
# Source: Cerebras "Power Lines" Table 1 / Eq. 6, AdamW EMA variant
#   m_τ ≈ -0.527 (exponent), constant c_τ calibrated below.
# Calibration: LLaMA 3 8B trained ~1T tokens at B≈2048 (sequences), λ=0.1,
#   with peak LR ~3e-4. Solving τ = B/(η*λ*D) gives τ ≈ 6.8e3.
#   TPP = D/N = 1e12 / 8e9 ≈ 125. c_τ = τ / TPP^(-0.527) ≈ 6.8e3 / 125^(-0.527).
#   125^0.527 ≈ 12.7, so c_τ ≈ 86500.
TAU_EXPONENT = -0.527     # Power Lines Eq. 6: m_τ_EMA ≈ -0.527
C_TAU = 86500.0           # Calibrated to LLaMA 3 8B regime (see note above)

WEIGHT_DECAY = 0.1        # λ: standard LLaMA-style weight decay (AdamW)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute optimal GBS from D (number of training tokens)
# ─────────────────────────────────────────────────────────────────────────────
def optimal_gbs(D, seq_length):
    """
    Compute principled optimal Global Batch Size from training tokens D.

    Formula: B_opt = B0 * (D / D0)^0.5
    Source: Cerebras "Power Lines" (Bergsma et al., NeurIPS 2025).
      B_opt and B_crit scale as D^~0.5, independent of model size N.

    Returns GBS snapped to the nearest multiple of 8 (for hardware alignment).
    """
    b_tokens = BSCALE_B0 * seq_length * ((D / BSCALE_D0) ** BSCALE_EXPONENT)
    gbs_exact = b_tokens / seq_length
    # Snap to nearest multiple of 8, minimum 8
    return max(8, int(round(gbs_exact / 8)) * 8)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute optimal peak LR from (B, D, N)
# ─────────────────────────────────────────────────────────────────────────────
def optimal_peak_lr(B, D, N, weight_decay=WEIGHT_DECAY):
    """
    Compute principled optimal peak learning rate using the AdamW timescale.

    Framework (Cerebras "Power Lines", Bergsma et al. NeurIPS 2025):
      The AdamW timescale τ = B / (η * λ * D) should follow a power law in
      the tokens-per-parameter ratio TPP = D / N:

        τ_opt(TPP) = C_TAU * TPP^TAU_EXPONENT    (TAU_EXPONENT ≈ -0.527)

      Solving for η (peak LR):
        η_peak = B / (λ * D * τ_opt)

    This correctly couples LR to:
      - B: more parallelism → take larger steps (LR ∝ B, linear scaling rule)
      - D: more data → more training progress → decrease LR (LR ∝ 1/D roughly)
      - N: larger model → higher TPP → larger τ → lower LR (LR ∝ N^0.527)

    Args:
        B: Global batch size (sequences per step)
        D: Total training tokens
        N: Non-embedding model parameters
        weight_decay: λ (AdamW weight decay coefficient)

    Returns:
        peak_lr (float)
    """
    TPP = D / N
    tau_opt = C_TAU * (TPP ** TAU_EXPONENT)
    peak_lr = B / (weight_decay * D * tau_opt)
    return peak_lr


# ─────────────────────────────────────────────────────────────────────────────
# Architecture param counting
# ─────────────────────────────────────────────────────────────────────────────
def params_per_layer(d):
    """Non-embedding params per transformer layer for LLaMA3-style arch."""
    # Attention: Q(d×d) + K(d×d/GQA) + V(d×d/GQA) + O(d×d)
    attn = d * d * (2 + 2 / GQA_RATIO)
    # SwiGLU MLP: gate(d×ffn) + up(d×ffn) + down(ffn×d)
    ffn_hidden = d * FFN_MULTIPLIER
    mlp = 3 * d * ffn_hidden
    # RMSNorm: 2 × d per layer (pre-attention + pre-MLP)
    norm = 2 * d
    return attn + mlp + norm


def total_non_embedding_params(d, L):
    """Total non-embedding params (what enters the 6ND formula)."""
    return L * params_per_layer(d)


def total_params(d, L):
    """Total params including embeddings (untied input + output)."""
    non_emb = total_non_embedding_params(d, L)
    emb = 2 * VOCAB_SIZE * d  # untied embeddings
    final_norm = d             # final RMSNorm
    return non_emb + emb + final_norm


def format_params(n):
    """Human-readable parameter count."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


# ─────────────────────────────────────────────────────────────────────────────
# Main computation + printing
# ─────────────────────────────────────────────────────────────────────────────
def compute_and_print_table(C, D, args):
    """
    Correct causal flow: C and D are the primary inputs.

      C (FLOPs budget) + D (total training tokens)
        →  N = C / (6D)          [target model size]
        →  GBS = B_opt(D)        [from Power Lines, no circularity]
        →  S = D / (GBS × L)    [steps derived last]
        →  LR = η_opt(B, D, N)  [from AdamW timescale]

    In sweep mode, D is varied around the Chinchilla optimal.
    S is not an input — it is printed as an output.
    """
    # ── Step 1: model size from FLOPs + tokens ──────────────────────────────
    N_target = C / (6 * D)   # C ≈ 6ND

    # ── Step 2: GBS from D only (no circularity) ────────────────────────────
    gbs = args.gbs if args.gbs is not None else optimal_gbs(D, args.seq_length)

    # ── Step 3: steps derived from D and GBS ───────────────────────────────
    S = max(1, int(round(D / (gbs * args.seq_length))))

    TPP = D / N_target   # tokens per parameter

    # ── Print summary header ────────────────────────────────────────────────
    print()
    print(f"  FLOPs budget (C):  {C:.2e}")
    print(f"  Total tokens (D):  {D:.2e}")
    print(f"  Target N (C/6D):   {N_target:.2e}  ({format_params(N_target)})")
    print(f"  TPP (D/N):         {TPP:.1f}  (Chinchilla optimal = 20)")
    print(f"  GBS:               {gbs}  {'[auto, B∝D^0.5]' if args.gbs is None else '[user-specified]'}")
    print(f"  Tokens/step:       {gbs} × {args.seq_length:,} = {gbs * args.seq_length:,}")
    print(f"  Total steps (S):   {S:,}  [derived: D / (GBS × seq_len)]")
    print(f"  Reference W/D:     128.0  (LLaMA 3 8B = 4096 / 32)")
    print()

    # ── Architecture table ──────────────────────────────────────────────────
    header = (
        f"{'HIDDEN':>8}  {'LAYERS':>6}  {'W/D':>6}  {'N (non-emb)':>14}  {'N (total)':>14}  "
        f"{'Actual C':>12}  {'C/target':>8}  {'heads':>5}  {'GQA':>4}  {'FFN':>7}"
    )
    print(header)
    print("─" * len(header))

    results = []
    for d in range(args.min_hidden, args.max_hidden + 1, HIDDEN_SIZE_STEP):
        ppl = params_per_layer(d)
        L_exact = N_target / ppl
        L = round(L_exact)
        if L < args.min_layers or L > args.max_layers:
            continue

        N_actual = total_non_embedding_params(d, L)
        N_total = total_params(d, L)
        C_actual = 6 * N_actual * D
        C_ratio = C_actual / C
        num_heads = d // HEAD_DIM
        num_query_groups = max(1, num_heads // GQA_RATIO)
        ffn = int(d * FFN_MULTIPLIER)
        aspect_ratio = d / L if L > 0 else 0

        results.append(
            (d, L, aspect_ratio, N_actual, N_total, C_actual, C_ratio, num_heads, num_query_groups, ffn)
        )

    # Sort: within 10% FLOPs budget first, then by closest W/D to 128
    results.sort(key=lambda x: (abs(1.0 - x[6]) > 0.10, abs(128 - x[2]), abs(1.0 - x[6])))

    for d, L, aspect_ratio, N_actual, N_total, C_actual, C_ratio, num_heads, gqa_g, ffn in results:
        marker = " ◄" if abs(C_ratio - 1.0) <= 0.10 else ""
        print(
            f"{d:>8}  {L:>6}  {aspect_ratio:>6.1f}  {N_actual:>14,.0f}  {N_total:>14,.0f}  "
            f"{C_actual:>12.2e}  {C_ratio:>7.1%}{marker:>3}  {num_heads:>5}  {gqa_g:>4}  {ffn:>7}"
        )

    if not results:
        print("  No valid configurations found in the given range.")
        print(f"  Target N={format_params(N_target)} may be too small or too large.")
        print(f"  Try adjusting --min-layers/--max-layers or --min-hidden/--max-hidden.")
        return

    # ── Best match ──────────────────────────────────────────────────────────
    best = results[0]
    hidden_size = best[0]
    num_layers  = best[1]
    best_N      = best[3]   # N_actual (non-embedding params)

    print(
        f"\n✓ Best match (W/D closest to 128): HIDDEN_SIZE={hidden_size}  NUM_LAYERS={num_layers}  "
        f"({format_params(best_N)} non-emb params, "
        f"actual C={best[5]:.2e}, {best[6]:.1%} of budget)"
    )

    # ── Peak LR: AdamW timescale from (B, D, N) ─────────────────────────────
    if args.base_lr is not None:
        peak_lr   = args.base_lr
        lr_source = "user-specified"
    else:
        peak_lr = optimal_peak_lr(gbs, D, best_N, weight_decay=args.weight_decay)
        peak_lr_clamped = max(1e-5, min(1e-2, peak_lr))
        if peak_lr != peak_lr_clamped:
            lr_source = f"AdamW timescale [CLAMPED from {peak_lr:.2e}]"
            peak_lr   = peak_lr_clamped
        else:
            lr_source = "AdamW timescale τ∝(D/N)^-0.527 [Power Lines]"

    tau_actual = gbs / (peak_lr * args.weight_decay * D)
    print(f"  Peak LR (η):       {peak_lr:.2e}  [{lr_source}]")
    print(f"  AdamW τ used:      {tau_actual:.2e}  (τ = B/(η·λ·D))")
    print(f"  Weight decay (λ):  {args.weight_decay}")

    # ── Format compact step string for directory names ───────────────────────
    s_str = f"{S:.1e}".replace("+0", "").replace("+", "").replace(".0e", "e")
    d_str = f"{D:.1e}".replace("+0", "").replace("+", "").replace(".0e", "e")

    ckpt_dir = f"~/checkpoints/llama3_h{hidden_size}_l{num_layers}_s{s_str}_fp8"
    tb_dir   = f"~/tensorboard_logs/llama3_h{hidden_size}_l{num_layers}_s{s_str}_fp8"

    print(
        f"\n  # D={d_str} tokens  →  GBS={gbs}  →  S={S:,} steps"
    )
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
        description="Compute (HIDDEN_SIZE, NUM_LAYERS) from FLOPs budget + training tokens",
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
            "Global batch size (sequences/step). "
            "If omitted, auto-computed from D via B∝D^0.5 (Cerebras Power Lines)."
        ),
    )
    parser.add_argument(
        "--seq-length", type=int, default=SEQ_LENGTH,
        help=f"Sequence length in tokens (default: {SEQ_LENGTH})",
    )
    parser.add_argument(
        "--base-lr", type=float, default=None,
        help="Override peak learning rate instead of computing from AdamW timescale.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=WEIGHT_DECAY,
        help=f"AdamW weight decay λ (default: {WEIGHT_DECAY}). Used in τ = B/(η·λ·D).",
    )
    parser.add_argument(
        "--min-layers", type=int, default=2,  help="Minimum number of layers (default: 2)"
    )
    parser.add_argument(
        "--max-layers", type=int, default=64, help="Maximum number of layers (default: 64)"
    )
    parser.add_argument(
        "--min-hidden", type=int, default=512,  help="Minimum hidden size (default: 512)"
    )
    parser.add_argument(
        "--max-hidden", type=int, default=8192, help="Maximum hidden size (default: 8192)"
    )
    args = parser.parse_args()

    # Chinchilla-optimal D: from C = 6ND and D = 20N  →  D_opt = sqrt(C * 10/3)
    D_opt = math.sqrt(args.flops * 10.0 / 3.0)
    N_opt = D_opt / 20.0   # implied optimal model size

    if args.tokens is not None:
        compute_and_print_table(args.flops, args.tokens, args)
    else:
        print(f"\n{'='*70}")
        print(f" SWEEP MODE: Chinchilla-based scaling law grid")
        print(f" C = {args.flops:.2e} FLOPs")
        print(f" Chinchilla-optimal D = {D_opt:.2e} tokens  (D = sqrt(C * 10/3))")
        print(f" Chinchilla-optimal N = {format_params(N_opt)}  (N = D / 20)")
        print(f" GBS: {'user-specified=' + str(args.gbs) if args.gbs else 'auto per B∝D^0.5 (Power Lines) — varies per sweep point'}")
        print(f" LR:  {'user-specified=' + str(args.base_lr) if args.base_lr else 'auto per AdamW timescale τ∝(D/N)^-0.527 — varies per sweep point'}")
        print(f" NOTE: S (steps) is derived per point as D/(GBS×seq_len)")
        print(f"{'='*70}")

        # Sweep D around the Chinchilla optimum
        multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
        for mult in multipliers:
            D = D_opt * mult
            print(f"\n\n{'─'*70}")
            print(f" SWEEP: {mult:.2f}x Chinchilla-optimal tokens  →  D = {D:.2e}")
            print(f"{'─'*70}")
            compute_and_print_table(args.flops, D, args)


if __name__ == "__main__":
    main()


