# stage10_analysis/intervention.py

import torch

def measure_ablation_impact(
    model,
    src,
    tgt_input,
    tgt_output,
    src_mask,
    tgt_mask,
    loss_fn,
    ablation_config
):
    """
    Measure effect of head ablation on loss.

    Returns:
        dict with baseline_loss, ablated_loss, delta
    """

    model.eval()

    device = next(model.parameters()).device

    src = src.to(device)
    tgt_input = tgt_input.to(device)
    tgt_output = tgt_output.to(device)
    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)

    with torch.no_grad():

        # ---- Baseline ----
        baseline_output = model(
            src,
            tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        if isinstance(baseline_output, tuple):
            baseline_output = baseline_output[0]

        baseline_loss = loss_fn(
            baseline_output.reshape(-1, baseline_output.size(-1)),
            tgt_output.reshape(-1)
        )

        # ---- Ablated ----
        ablated_output = model(
            src,
            tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            head_ablation_config=ablation_config
        )

        if isinstance(ablated_output, tuple):
            ablated_output = ablated_output[0]

        ablated_loss = loss_fn(
            ablated_output.reshape(-1, ablated_output.size(-1)),
            tgt_output.reshape(-1)
        )

    return {
        "baseline_loss": baseline_loss.item(),
        "ablated_loss": ablated_loss.item(),
        "delta": (ablated_loss - baseline_loss).item()
    }