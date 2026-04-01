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

    def forward(self, x, encoder_output, tgt_mask=None, src_mask = None):
        """
        x: (B, T, d_model)
        encoder_output: (B, S, d_model)
        """

        assert x.dim() == 3, "x must be (B, T, d_model)"
        assert encoder_output.dim() == 3, "encoder_output must be (B, S, d_model)"
        assert x.size(0) == encoder_output.size(0), "Batch size mismatch"

        if self.verbose:
            print(f"[DecoderLayer] x: {x.shape}")
            print(f"[DecoderLayer] encoder_output: {encoder_output.shape}")

        # 1. Masked Self-Attention
        attn_out, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 2. Cross-Attention
        cross_out, _ = self.cross_attn(
            x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_out))

        # 3. Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x, self_attn_weights
    

# Decoder Stack
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout_p=0.1, verbose=False):
        super().__init__()

        self.verbose = verbose

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_p, verbose)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        x: (B, T, d_model)
        encoder_output: (B, S, d_model)
        """

        assert x.dim() == 3
        assert encoder_output.dim() == 3
        assert x.size(0) == encoder_output.size(0), \
            f"Batch size mismatch: x {x.size(0)} vs memory {encoder_output.size(0)}"
        assert x.size(2) == encoder_output.size(2),\
            f"d_model mismatch: x {x.size(2)} vs memory {encoder_output.size(2)}"
        
        if self.verbose:
            print(f"[Decoder] input x: {x.shape}")

        all_weights = []

        for i, layer in enumerate(self.layers):
            if self.verbose:
                print(f"[Decoder] Layer {i}")

            x, attn_w = layer(x, encoder_output, tgt_mask, src_mask)
            all_weights.append(attn_w)

        # take last layer attention
        final_weights = all_weights[-1]

        return x, final_weights