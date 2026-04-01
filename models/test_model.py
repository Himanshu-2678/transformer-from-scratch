# models/test_model.py

import torch
from models.transformer import Transformer

model = Transformer(
    src_vocab_size=100,
    tgt_vocab_size=100,
    d_model=32,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=64,
    max_seq_len=50,
    dropout_p=0.0,
    pad_idx=0,
)

model.eval()

# ---- dummy input ----
src = torch.randint(1, 100, (2, 10))  # [B=2, S=10]
tgt = torch.randint(1, 100, (2, 7))   # [B=2, T=7]

# ---- forward pass ----
with torch.no_grad():
    out, _ = model(src, tgt)

print("Output shape:", out.shape)