# stage09_debugging/test_attention.py

import torch

def check_attention_causality(attn_weights):
    """
    PURPOSE:
    Ensure that attention weights do NOT assign probability to future tokens.

    INPUT:
    attn_weights: [B, heads, T, T]

    METHOD:
    - Extract one head
    - Check upper triangle (future positions)
    - These values must be ~0
    """

    # Take first batch, first head
    attn = attn_weights[0, 0]  # shape: [T, T]

    T = attn.shape[0]

    # Extract upper triangle (future positions)
    upper = torch.triu(attn, diagonal=1)

    max_val = upper.max().item()

    print("\n[ATTENTION CAUSALITY CHECK]")
    print(f"max attention to future tokens: {max_val:.6e}")

    # Must be effectively zero
    assert max_val < 1e-5, "attention is leaking to future tokens"

    print("attention causality correct")


def attention_row_sums(attn_weights):
    """
    PURPOSE:
    Ensure attention distribution is valid (softmax correctness).

    METHOD:
    - Each row should sum to 1
    """

    attn = attn_weights[0, 0]  # [T, T]

    row_sums = attn.sum(dim=-1)

    print("\n[ATTENTION ROW SUM CHECK]")
    print(row_sums)

    # Check closeness to 1
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        "attention rows do not sum to 1"

    print("attention rows sum to 1")


def attention_entropy(attn_weights):
    """
    PURPOSE:
    Measure how sharp or diffuse attention is.

    METHOD:
    - Compute entropy across each row
    - Low entropy = very sharp attention
    - High entropy = diffuse attention (spread across many tokens).
    - Extremely low entropy early in training can indicate issues
    """

    attn = attn_weights[0, 0]  # [T, T]

    entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1)

    mean_entropy = entropy.mean().item()

    print("\n[ATTENTION ENTROPY]")
    print(f"mean entropy: {mean_entropy:.6f}")

    return mean_entropy


def attention_mask_alignment(attn_weights, tgt_mask):
    """
    PURPOSE:
    Ensure that zeroed mask positions correspond to zero attention.

    METHOD:
    - Wherever mask is False, attention should be ~0
    """

    attn = attn_weights[0, 0]       # [T, T]
    mask = tgt_mask[0]              # [T, T]

    # Positions where mask is False
    blocked_positions = (~mask)

    leaked_attention = attn[blocked_positions]

    if leaked_attention.numel() > 0:
        max_leak = leaked_attention.max().item()
    else:
        max_leak = 0.0

    print("\n[ATTENTION MASK ALIGNMENT]")
    print(f"max attention on masked positions: {max_leak:.6e}")

    assert max_leak < 1e-5, "attention assigned to masked positions"

    print("mask and attention aligned")