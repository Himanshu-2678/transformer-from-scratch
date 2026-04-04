# stage10_analysis/attention_utils.py

import torch

def collect_attention(model, src, tgt_input, src_mask, tgt_mask):
    """
    Returns:
        dict[layer] = {
            "self": tensor (B, H, T, T),
            "cross": tensor (B, H, T, S)
        }
    """

    model.eval()

    with torch.no_grad():
        _, attn = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            return_attention=True
        )

    # Validate structure
    for layer, layer_attn in attn.items():
        assert "self" in layer_attn
        assert "cross" in layer_attn

        assert layer_attn["self"].dim() == 4, f"{layer} self wrong shape"
        assert layer_attn["cross"].dim() == 4, f"{layer} cross wrong shape"

    return attn