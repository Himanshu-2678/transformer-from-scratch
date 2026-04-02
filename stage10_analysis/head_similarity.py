# stage10_analysis/head_similarity.py

import torch


def compute_head_similarity(attn, mask=None, eps=1e-9):
    """
    Compute cosine similarity between attention heads.
    Args:
        attn: Tensor [B, H, T, T]
        mask: Optional mask [B, 1, T, T] or [B, T, T]
              True = valid, False = masked
        eps: numerical stability

    Returns:
        sim_matrix: Tensor [H, H]
    """

    assert attn.dim() == 4, "Attention must be [B, H, T, T]"
    assert not torch.isnan(attn).any(), "NaNs in attention"

    B, H, T, _ = attn.shape

    # ---- Apply mask ----
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, 1, T, T]

        attn = attn * mask
        attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)

    # ---- Flatten attention maps ----
    # shape: [B, H, T*T]
    attn_flat = attn.reshape(B, H, -1)

    # ---- Normalize vectors ----
    attn_flat = attn_flat / (attn_flat.norm(dim=-1, keepdim=True) + eps)

    # ---- Vectorized cosine similarity ----
    # [B, H, H]
    sim = torch.matmul(attn_flat, attn_flat.transpose(1, 2))

    # Average over batch → [H, H]
    sim_matrix = sim.mean(dim=0)

    # ---- Sanity check: diagonal should be ~1 ----
    assert torch.allclose(
        sim_matrix.diag(),
        torch.ones(H, device=sim_matrix.device),
        atol=1e-4
    ), "Diagonal similarity deviates from 1"

    return sim_matrix


def get_head_similarity(attn_dict, mask=None):
    """
    Compute similarity matrices for all layers.
    Args:
        attn_dict: dict[layer_name -> tensor[B, H, T, T]]

    Returns:
        layerwise_similarity: dict[layer_name -> sim_matrix[H, H]]
    """

    layerwise_similarity = {}

    for layer, attn in attn_dict.items():
        sim = compute_head_similarity(attn, mask)
        layerwise_similarity[layer] = sim  # keep on device

    return layerwise_similarity

def detect_redundant_heads(sim_matrix, threshold=0.9):
    """
    Detect highly similar (redundant) heads.
    Args:
        sim_matrix: Tensor [H, H]
        threshold: similarity above which heads are considered redundant

    Returns:
        list of (i, j, similarity)
    """

    H = sim_matrix.shape[0]
    redundant_pairs = []

    for i in range(H):
        for j in range(i + 1, H):
            sim = sim_matrix[i, j].item()
            if sim >= threshold:
                redundant_pairs.append((i, j, sim))

    return redundant_pairs

def build_head_similarity_report(layerwise_similarity, threshold=0.9):
    """
    Build structured report from similarity matrices.
    Returns:
        report: dict[layer -> {
            "mean_similarity": float,
            "max_similarity": float,
            "redundant_heads": list
        }]
    """

    report = {}

    for layer, sim in layerwise_similarity.items():
        sim_cpu = sim.detach().cpu()

        report[layer] = {
            "mean_similarity": sim_cpu.mean().item(),
            "max_similarity": sim_cpu.max().item(),
            "redundant_heads": detect_redundant_heads(sim_cpu, threshold)}

    return report