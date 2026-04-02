# models/transformer.py

import math
import torch
import torch.nn as nn

from stage04_positional_encoding.positional_encoding import PositionalEncoding
from stage05_encoder.encoder import Encoder
from stage06_decoder.decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout_p=0.1,
        pad_idx=0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout_p)

        # encoder / decoder
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout_p)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout_p)

        # output layer
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        return_attention=False,
        head_ablation_config=None
    ):
        """
        src: [B, S]
        tgt: [B, T]
        """

        assert src.dim() == 2
        assert tgt.dim() == 2
        assert src.size(0) == tgt.size(0)

        B, S = src.shape
        T = tgt.shape[1]
        device = src.device

        # ---- masks ----
        assert src_mask is not None
        assert tgt_mask is not None

        assert src_mask.shape == (B, S)
        assert tgt_mask.shape == (B, T, T)

        # ---- embeddings ----
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)

        # ---- encoder ----
        memory = self.encoder(src_emb, src_mask)

        # ---- decoder ----
        if return_attention:
            out, attn_weights = self.decoder(
                tgt_emb,
                memory,
                tgt_mask,
                src_mask,
                return_attention=True,
                head_ablation_config=head_ablation_config
            )
        else:
            out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask,
                src_mask,
                return_attention=False,
                head_ablation_config=head_ablation_config
            )

        # ---- output ----
        logits = self.output_projection(out)

        if return_attention:
            return logits, attn_weights

        return logits