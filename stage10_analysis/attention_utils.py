# stage10_analysis/attention_utils.py

import torch

def collect_attention(model, src, tgt, src_mask, tgt_mask):
    model.eval()

    device = next(model.parameters()).device
    src = src.to(device)
    tgt = tgt.to(device)
    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)

    with torch.no_grad():
        output, attn = model(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            return_attention=True
        )

    for k, v in attn.items():
        assert v.dim() == 4, f"{k} wrong shape"
        assert not torch.isnan(v).any(), f"{k} has NaNs"

    return attn