# stage06_decoder/decoder.py

import torch
import torch.nn as nn

from stage03_multihead.multihead_attention import MultiHeadAttention


def make_causal_mask(T: int) -> torch.Tensor:
    return torch.tril(torch.ones(T, T)).unsqueeze(0)


#------------------------------------------------------------
# Decoder Layer 
# ------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_p=0.1, verbose = False):
        super().__init__()

        self.verbose = verbose

        # Attention block
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout_p=dropout_p, verbose=verbose)
        
        self.cross_attn = MultiHeadAttention(
            d_model, num_heads, dropout_p=dropout_p, verbose=verbose)

        # Feed Forward 
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model))
        
        # LayerNorms (Post-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dropout3 = nn.Dropout(dropout_p)

    def forward(
        self,
        x,
        encoder_output,
        tgt_mask=None,
        src_mask=None,
        return_attention=False,
        head_ablation=None
    ):
        # 1. Masked Self-Attention
        attn_out, self_attn_weights = self.self_attn(
            x, x, x,
            tgt_mask,
            head_ablation=head_ablation
        )
        x = self.norm1(x + self.dropout1(attn_out))

        # 2. Cross-Attention (no ablation here for now)
        cross_out, _ = self.cross_attn(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout2(cross_out))

        # 3. Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        if return_attention:
            return x, self_attn_weights

        return x
    

# Decoder Stack
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout_p=0.1, verbose=False):
        super().__init__()

        self.verbose = verbose

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_p, verbose)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x,
        encoder_output,
        tgt_mask=None,
        src_mask=None,
        return_attention=False,
        head_ablation_config=None
    ):
        attn_dict = {}

        for i, layer in enumerate(self.layers):

            layer_name = f"layer_{i}"

            # get ablation config for this layer
            ablation = None
            if head_ablation_config and layer_name in head_ablation_config:
                ablation = head_ablation_config[layer_name]

            if return_attention:
                x, attn = layer(
                    x,
                    encoder_output,
                    tgt_mask,
                    src_mask,
                    return_attention=True,
                    head_ablation=ablation
                )
                attn_dict[layer_name] = attn

            else:
                x = layer(
                    x,
                    encoder_output,
                    tgt_mask,
                    src_mask,
                    return_attention=False,
                    head_ablation=ablation
                )

        if return_attention:
            return x, attn_dict

        return x