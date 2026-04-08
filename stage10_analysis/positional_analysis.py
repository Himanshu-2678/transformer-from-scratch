# stage10_analysis/positional_analysis.py

import torch
import math
from collections import defaultdict


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


def compute_relative_position_distribution(attn_weights):
    """
    Convert attention maps into relative position distributions.

    Args:
        attn_weights: Tensor (B, H, T, T)

    Returns:
        dict:
            head_idx -> {relative_position (int): probability (float)}
    """

    B, H, T, _ = attn_weights.shape

    # Accumulate counts
    rel_pos_counts = [defaultdict(float) for _ in range(H)]

    for b in range(B):
        for h in range(H):
            for i in range(T):          # query
                for j in range(T):      # key
                    r = j - i
                    rel_pos_counts[h][r] += attn_weights[b, h, i, j].item()

    # Normalize per head
    rel_pos_dist = []

    for h in range(H):
        total = sum(rel_pos_counts[h].values()) + 1e-8
        dist = {r: v / total for r, v in rel_pos_counts[h].items()}
        rel_pos_dist.append(dist)

    return rel_pos_dist


def summarize_head_behavior(rel_pos_dist):
    """
    Extract peak position, entropy, and variance for each head.

    Args:
        rel_pos_dist: output of compute_relative_position_distribution

    Returns:
        List of dicts (one per head)
    """

    summaries = []

    for dist in rel_pos_dist:
        positions = list(dist.keys())
        probs = list(dist.values())

        # Peak
        peak_r = max(dist, key = dist.get)

        # Entropy
        entropy = -sum(p * math.log(p + 1e-9)for p in probs)

        # Mean
        mean = sum(r * p for r, p in dist.items())

        # Variance
        var = sum(((r - mean) ** 2) * p for r, p in dist.items())

        summaries.append({
            "peak_r": peak_r,
            "entropy": entropy,
            "variance": var
        })

    return summaries


def classify_heads(
    summaries,
    token_entropy,
    entropy_threshold=1.7,
    var_threshold=6.0,
    std_threshold=0.25,
    low_entropy_threshold=0.5
):
    """
    Final classification including content-based heads
    """

    labels = []

    for s, te in zip(summaries, token_entropy):
        peak = s["peak_r"]
        entropy = s["entropy"]
        var = s["variance"]

        mean_e = te["mean_entropy"]
        std_e = te["std_entropy"]

        # ---- Content-based (KV heads) ----
        if mean_e < low_entropy_threshold and std_e > std_threshold:
            labels.append("content")

        # ---- Diffuse ----
        elif entropy > entropy_threshold:
            labels.append("diffuse")

        # ---- Identity ----
        elif abs(peak) == 0 and var < var_threshold and mean_e < 0.6:
            labels.append("identity")

        # ---- Shift ----
        elif abs(peak) > 0 and var < var_threshold:
            labels.append(f"shift_{peak}")

        else:
            labels.append("weak_positional")

    return labels



def compute_per_token_entropy(attn_weights):
    """
    Compute entropy per query token per head.

    Args:
        attn_weights: (B, H, T, T)

    Returns:
        dict:
            head_idx -> {
                "mean_entropy": float,
                "std_entropy": float
            }
    """
    B, H, T, _ = attn_weights.shape

    results = []

    for h in range(H):
        entropies = []

        for b in range(B):
            for i in range(T):
                probs = attn_weights[b, h, i]

                entropy = -torch.sum(
                    probs * torch.log(probs + 1e-9)
                ).item()

                entropies.append(entropy)

        entropies = torch.tensor(entropies)

        results.append({
            "mean_entropy": entropies.mean().item(),
            "std_entropy": entropies.std().item()
        })

    return results