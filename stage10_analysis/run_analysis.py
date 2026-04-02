# stage10_analysis/run_analysis.py

import torch

from stage10_analysis.attention_utils import collect_attention
from stage10_analysis.attn_entropy import summarize_entropy
from stage10_analysis.head_similarity import (
    get_head_similarity,
    build_head_similarity_report
)
from stage10_analysis.positional_analysis import (
    compute_positional_profile,
    detect_positional_peaks,
    compute_diagonal_strength
)
from stage10_analysis.intervention import measure_ablation_impact


def run_attention_analysis(
    model,
    dataloader,
    loss_fn,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_batches=10
):
    model.to(device)
    model.eval()

    all_attn = {}
    batch_count = 0

    # -----------------------------
    # Collect Attention
    # -----------------------------
    for batch in dataloader:
        if batch_count >= max_batches:
            break

        src, tgt_input, src_mask, tgt_mask, tgt_output = batch

        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        attn = collect_attention(model, src, tgt_input, src_mask, tgt_mask)

        for layer, tensor in attn.items():
            if layer not in all_attn:
                all_attn[layer] = []
            all_attn[layer].append(tensor.detach())

        batch_count += 1

    # Stack
    for layer in all_attn:
        all_attn[layer] = torch.cat(all_attn[layer], dim=0)

    # -----------------------------
    # 1. Entropy
    # -----------------------------
    entropy_summary = summarize_entropy(all_attn)

    # -----------------------------
    # 2. Similarity
    # -----------------------------
    similarity = get_head_similarity(all_attn)
    similarity_report = build_head_similarity_report(similarity)

    # -----------------------------
    # 3. Positional Analysis
    # -----------------------------
    positional_profiles = {}
    positional_peaks = {}
    diagonal_strength = {}

    for layer, attn in all_attn.items():
        profile = compute_positional_profile(attn)
        peaks = detect_positional_peaks(profile)
        diag = compute_diagonal_strength(attn)

        positional_profiles[layer] = profile.detach().cpu()
        positional_peaks[layer] = peaks
        diagonal_strength[layer] = diag.detach().cpu()

    # -----------------------------
    # 4. Ablation (single head)
    # -----------------------------
    ablation_results = {}

    # reuse last batch (fast but noisy)
    for layer in all_attn.keys():
        H = all_attn[layer].shape[1]

        layer_results = []

        for h in range(H):
            config = {
                "layer_0": [0,1,2,3],
                "layer_1": [0,1,2,3]
            }

            result = measure_ablation_impact(
                model,
                src,
                tgt_input,
                tgt_output,
                src_mask,
                tgt_mask,
                loss_fn,
                config
            )

            layer_results.append({
                "head": h,
                "delta_loss": result["delta"]
            })

        ablation_results[layer] = layer_results

    # -----------------------------
    # Final Output
    # -----------------------------
    results = {
        "entropy": entropy_summary,
        "similarity": similarity_report,
        "positional": {
            "profiles": positional_profiles,
            "peaks": positional_peaks,
            "diagonal_strength": diagonal_strength
        },
        "ablation": ablation_results
    }

    return results