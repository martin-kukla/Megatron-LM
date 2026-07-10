#!/usr/bin/env python3
"""
Given a FLOPs budget and total training steps, compute valid
(HIDDEN_SIZE, NUM_LAYERS) pairs for a LLaMA3-style architecture.

Uses the standard approximation: C ≈ 6 × N × D
  where N = non-embedding params, D = total tokens processed.

Architecture conventions (matching train_llama3_scaling_laws_b200_fp8.sh):
  - head_dim = 128, GQA ratio = 4:1
  - ffn_hidden = 3.5 × hidden_size (SwiGLU)
  - GBS = 128, seq_length = 8192

Usage:
    python3 examples/llama/compute_model_config.py --flops 1e19 --total-steps 10000
    python3 examples/llama/compute_model_config.py --flops 6e18 --total-steps 5000 --gbs 128
    python3 examples/llama/compute_model_config.py --flops 1e19 # Sweeps around Chinchilla opt
"""

import argparse
import math

# LLaMA3 architecture constants
HEAD_DIM = 128
GQA_RATIO = 4
FFN_MULTIPLIER = 3.5  # ffn_hidden_size = 3.5 × hidden_size
SEQ_LENGTH = 8192
VOCAB_SIZE = 128256
HIDDEN_SIZE_STEP = HEAD_DIM * GQA_RATIO  # 512, minimum granularity

# LLaMA 3 empirical optimal Global Batch Size (GBS) lookup table for seq_length=8192.
# Derived from the LLaMA 3 scaling law experiments (C from 6e18 to 1e22, batch from 250K to 4M tokens)
# B_tokens = 250,000 * (C / 6e18)^0.3736
LLAMA3_GBS_LOOKUP = {
    6e18: 32,    # ~250K tokens
    1e19: 40,    # ~300K tokens
    3e19: 64,    # ~450K tokens
    6e19: 80,    # ~590K tokens
    1e20: 96,    # ~710K tokens
    3e20: 128,   # ~1.05M tokens
    6e20: 160,   # ~1.36M tokens
    1e21: 192,   # ~1.64M tokens
    3e21: 256,   # ~2.42M tokens
    1e22: 512,   # ~4.00M tokens
}

def get_gbs_for_flops(flops):
    """Return GBS based on LLaMA 3 empirical table or interpolation."""
    if flops in LLAMA3_GBS_LOOKUP:
        return LLAMA3_GBS_LOOKUP[flops]
    
    # Fallback to power-law interpolation if not exactly in table
    b_tokens = 250000 * ((flops / 6e18) ** 0.3736)
    gbs_approx = b_tokens / SEQ_LENGTH
    # Snap to nearest multiple of 16
    return max(16, int(round(gbs_approx / 16)) * 16)


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
    final_norm = d  # final RMSNorm
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


def compute_and_print_table(C, S, args):
    D = S * args.gbs * args.seq_length  # total tokens

    # Target non-embedding params from 6ND approximation
    N_target = C / (6 * D)

    print()
    print(f"  FLOPs budget (C):  {C:.2e}")
    print(f"  Total steps (S):   {S:,}")
    print(f"  Tokens/step:       {args.gbs} × {args.seq_length:,} = {args.gbs * args.seq_length:,}")
    print(f"  Total tokens (D):  {D:.2e}")
    print(f"  Target N (C/6D):   {N_target:.2e}  ({format_params(N_target)})")
    print(f"  Reference W/D:     128.0  (LLaMA 3 8B is 4096 / 32 = 128)")
    print()

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

    # Sort by within 10% FLOP budget first, then by closest to W/D = 128
    results.sort(key=lambda x: (abs(1.0 - x[6]) > 0.10, abs(128 - x[2]), abs(1.0 - x[6])))

    for d, L, aspect_ratio, N_actual, N_total, C_actual, C_ratio, num_heads, gqa, ffn in results:
        marker = " ◄" if abs(C_ratio - 1.0) <= 0.10 else ""
        print(
            f"{d:>8}  {L:>6}  {aspect_ratio:>6.1f}  {N_actual:>14,.0f}  {N_total:>14,.0f}  "
            f"{C_actual:>12.2e}  {C_ratio:>7.1%}{marker:>3}  {num_heads:>5}  {gqa:>4}  {ffn:>7}"
        )

    if not results:
        print("  No valid configurations found in the given range.")
        print(
            f"  Target N={format_params(N_target)} may be too small or too large."
        )
        print(f"  Try adjusting --min-layers/--max-layers or --min-hidden/--max-hidden.")
        return

    # Print the best match
    best = results[0]
    print(
        f"\n✓ Best match (W/D closest to 128): HIDDEN_SIZE={best[0]}  NUM_LAYERS={best[1]}  "
        f"({format_params(best[3])} non-emb params, "
        f"actual C={best[5]:.2e}, {best[6]:.1%} of budget)"
    )
    hidden_size = best[0]
    num_layers = best[1]
    # Format total steps S in a compact scientific notation (e.g. 10000 -> 1e4, 15000 -> 1.5e4)
    s_str = f"{S:.1e}".replace("+0", "").replace("+", "").replace(".0e", "e")
    
    ckpt_dir = f"~/checkpoints/llama3_h{hidden_size}_l{num_layers}_s{s_str}_fp8"
    tb_dir = f"~/tensorboard_logs/llama3_h{hidden_size}_l{num_layers}_s{s_str}_fp8"
    
    print(f"\n  GLOBAL_BATCH_SIZE={args.gbs} HIDDEN_SIZE={hidden_size} NUM_LAYERS={num_layers} TOTAL_STEPS={S} \\")
    print(f"    ./examples/llama/train_llama3_scaling_laws_b200_fp8.sh \\")
    print(f"        {ckpt_dir} \\")
    print(f"        {tb_dir} \\")
    print(f"        meta-llama/Meta-Llama-3-8B \\")
    print(f"        ~/pile_tokenized/pile_llama3_text_document")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute (HIDDEN_SIZE, NUM_LAYERS) from FLOPs budget + training steps"
    )
    parser.add_argument(
        "--flops", type=float, required=True, help="Total FLOPs budget (e.g., 1e19)"
    )
    parser.add_argument(
        "--total-steps", type=int, required=False, default=None, help="Total training steps (optional, if omitted will sweep around Chinchilla optimal)"
    )
    parser.add_argument(
        "--gbs", type=int, default=None,
        help="Global batch size. If omitted, uses LLaMA 3 empirical optimal mapping.",
    )
    parser.add_argument(
        "--seq-length", type=int, default=SEQ_LENGTH,
        help=f"Sequence length (default: {SEQ_LENGTH})",
    )
    parser.add_argument(
        "--min-layers", type=int, default=2, help="Minimum number of layers (default: 2)"
    )
    parser.add_argument(
        "--max-layers", type=int, default=64, help="Maximum number of layers (default: 64)"
    )
    parser.add_argument(
        "--min-hidden", type=int, default=512, help="Minimum hidden size (default: 512)"
    )
    parser.add_argument(
        "--max-hidden", type=int, default=8192, help="Maximum hidden size (default: 8192)"
    )
    args = parser.parse_args()

    if args.gbs is None:
        args.gbs = get_gbs_for_flops(args.flops)
        print(f"\n[INFO] Auto-selected GBS={args.gbs} based on FLOPs budget (C={args.flops:.2e})")

    if args.total_steps is not None:
        compute_and_print_table(args.flops, args.total_steps, args)
    else:
        # Calculate optimal steps based on Chinchilla D = 20 * N
        # C = 6 * N * D -> C = 6 * (D/20) * D = 0.3 * D^2 -> D = sqrt(C / 0.3) = sqrt(C * 3.3333333)
        D_opt = math.sqrt(args.flops * 3.333333333)
        S_opt_float = D_opt / (args.gbs * args.seq_length)
        S_opt = int(round(S_opt_float))
        
        print(f"\n=====================================================================")
        print(f" SWEEP MODE: Generating scaling law grid around Chinchilla optimum")
        print(f" C = {args.flops:.2e} FLOPs")
        print(f" Chinchilla Optimal D ≈ {D_opt:.2e} tokens")
        print(f" Chinchilla Optimal S ≈ {S_opt:,} steps")
        print(f"=====================================================================")
        
        multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
        for mult in multipliers:
            S = int(round(S_opt * mult))
            print(f"\n\n\n=== [ SWEEP: {mult}x Optimal Steps (S = {S:,}) ] =======================================")
            compute_and_print_table(args.flops, S, args)


if __name__ == "__main__":
    main()
