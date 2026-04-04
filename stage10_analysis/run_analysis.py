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

        for layer, layer_attn in attn.items():
            if layer not in all_attn:
                all_attn[layer] = {
                    "self": [],
                    "cross": []
                }

            all_attn[layer]["self"].append(layer_attn["self"].detach())
            all_attn[layer]["cross"].append(layer_attn["cross"].detach())

        batch_count += 1

    # Stack batches
    for layer in all_attn:
        all_attn[layer]["self"] = torch.cat(all_attn[layer]["self"], dim=0)
        all_attn[layer]["cross"] = torch.cat(all_attn[layer]["cross"], dim=0)

    # -----------------------------
    # 1. Entropy
    # -----------------------------
    self_attn_only = {
        layer: all_attn[layer]["self"]
        for layer in all_attn
    }

    entropy_summary = summarize_entropy(self_attn_only)

    # -----------------------------
    # 2. Similarity
    # -----------------------------
    similarity = get_head_similarity(self_attn_only)
    similarity_report = build_head_similarity_report(similarity)

    # -----------------------------
    # 3. Positional Analysis
    # -----------------------------
    positional_profiles = {}
    positional_peaks = {}
    diagonal_strength = {}

    for layer in all_attn:
        attn = all_attn[layer]["self"]

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
        H = all_attn[layer]["self"].shape[1]

        layer_results = []

        for h in range(H):

            config = {
                layer: [h]   # ONLY ablate this head
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


if __name__ == "__main__":
    from models.transformer import Transformer
    from stage10_analysis.tasks.reverse import generate_reverse_task

    PAD = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 20

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        max_seq_len=50,
        dropout_p=0.0,
        pad_idx=PAD,
    ).to(device)

    model.load_state_dict(torch.load("checkpoint_kv.pt", map_location=device))

    # Dummy dataloader (simple loop)
    def simple_dataloader(num_batches=10):
        for _ in range(num_batches):
            src, tgt_input, tgt_output = generate_reverse_task(
                batch_size=32,
                seq_len=6,
                vocab_size=vocab_size,
                device=device
            )

            src_mask = (src != PAD)

            T_len = tgt_input.size(1)
            tgt_pad_mask = (tgt_input != PAD).unsqueeze(1)
            causal_mask = torch.tril(torch.ones(T_len, T_len, device=device)).bool().unsqueeze(0)
            tgt_mask = tgt_pad_mask & causal_mask

            yield src, tgt_input, src_mask, tgt_mask, tgt_output

    results = run_attention_analysis(
        model,
        simple_dataloader(),
        loss_fn=torch.nn.CrossEntropyLoss(ignore_index=PAD),
        device=device
    )

    print("\n=== RESULTS ===")
    print(results["entropy"])
    print(results["similarity"])
    print(results["positional"]["diagonal_strength"])