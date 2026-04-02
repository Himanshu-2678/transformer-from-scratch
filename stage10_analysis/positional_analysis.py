# stage10_analysis/positional_analysis.py

import torch

def compute_positional_profile(attn):
    """
    Compute attention distribution over relative distances.
    Args:
        attn: Tensor [B, H, T, T]

    Returns:
        profile: Tensor [H, T]  (distance distribution)
    """

    B, H, T, _ = attn.shape

    profile = torch.zeros(H, T, device=attn.device)

    for i in range(T):
        for j in range(T):
            dist = abs(i - j)
            profile[:, dist] += attn[:, :, i, j].mean(dim=0)

    # normalize
    profile = profile / (profile.sum(dim=-1, keepdim=True) + 1e-9)

    return profile


def detect_positional_peaks(profile):
    """
    Identify dominant attention distance per head.
    Args:
        profile: [H, T]

    Returns:
        list of (head_idx, dominant_distance, strength)
    """

    peaks = []

    for h in range(profile.shape[0]):
        values = profile[h]
        dist = torch.argmax(values).item()
        strength = values[dist].item()

        peaks.append((h, dist, strength))

    return peaks


def compute_diagonal_strength(attn):
    """
    Measure how much attention is focused on self (diagonal).
    Returns:
        Tensor [H]
    """

    diag = attn.diagonal(dim1=-2, dim2=-1)  # [B, H, T]
    return diag.mean(dim=(0, 2))