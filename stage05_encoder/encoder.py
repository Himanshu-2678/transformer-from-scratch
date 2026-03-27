# stage05_encoder/encoder.py

from stage03_multihead.multihead_attention import MultiHeadAttention
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_p=0.0, verbose=False):
        super().__init__()

        self.verbose = verbose

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_p, verbose)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x, src_mask=None):

        assert x.dim() == 3, f"Expected 3D input, got {x.shape}"
        assert x.shape[2] == self.norm1.normalized_shape[0], \
            f"d_model mismatch: {x.shape[2]} vs {self.norm1.normalized_shape[0]}"

        if self.verbose:
            print(f"[EncoderLayer] Input: {x.shape}")

        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        if self.verbose:
            print(f"[EncoderLayer] Output: {x.shape}")

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout_p=0.0, verbose=False):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout_p, verbose)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None):

        assert x.dim() == 3, f"Expected 3D input, got {x.shape}"

        for layer in self.layers:
            x = layer(x, src_mask)

        return x