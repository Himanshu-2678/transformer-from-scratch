# stage10_analysis/attn_entropy.py

import torch

def compute_attention_entropy(attn, mask=None, eps=1e-9, normalize=True):
    """
    Compute entropy of attention distributions.

    Args:
        attn: Tensor [B, H, T, T]
        mask: Optional mask [B, 1, T, T] or [B, T, T]
              True = valid, False = masked
        eps: Numerical stability
        normalize: Whether to normalize entropy by log(valid_tokens)

    Returns:
        entropy_per_head: Tensor [H]
        entropy_full: Tensor [B, H, T]
    """

    # ---- Basic validation ----
    assert attn.dim() == 4, "Attention must be [B, H, T, T]"
    assert not torch.isnan(attn).any(), "NaNs in attention"

    B, H, T, _ = attn.shape

    # ---- Apply mask ----
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, 1, T, T]

        attn = attn * mask  # zero out invalid positions

        valid_counts = mask.sum(dim=-1)  # [B, 1, T]
    else:
        valid_counts = torch.full((B, 1, T), T, device=attn.device)

    # Avoiding division by zero
    valid_counts = valid_counts.clamp(min=1)

    # ---- Entropy computation ----
    log_attn = torch.log(attn + eps)
    entropy = -(attn * log_attn).sum(dim=-1)  # [B, H, T]

    # ---- Normalize ----
    if normalize:
        max_entropy = torch.log(valid_counts)  # [B, 1, T]
        entropy = entropy / (max_entropy + eps)

    # ---- Aggregate ----
    entropy_per_head = entropy.mean(dim=(0, 2))  # [H]

    return entropy_per_head, entropy


def summarize_entropy(attn_dict, mask=None):
    """
    Compute entropy across all layers.
    Args:
        attn_dict: dict[layer_name -> tensor[B, H, T, T]]
        mask: optional mask

    Returns:
        summary: dict[layer_name -> entropy_per_head]
    """

    summary = {}

    for layer, attn in attn_dict.items():
        entropy_per_head, _ = compute_attention_entropy(attn, mask)
        summary[layer] = entropy_per_head.cpu()

    return summary


def print_entropy_summary(summary):

    for layer, values in summary.items():
        values = values.numpy()

        print(f"\nLayer: {layer}")
        print(f"  Mean Entropy: {values.mean():.4f}")
        print(f"  Min Entropy : {values.min():.4f}")
        print(f"  Max Entropy : {values.max():.4f}")

        for i, v in enumerate(values):
            print(f"    Head {i}: {v:.4f}")