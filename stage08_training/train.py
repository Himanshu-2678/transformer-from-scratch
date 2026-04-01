# stage08_training/train.py

import torch
import torch.nn as nn
from torch.optim import Adam

from models.transformer import Transformer


# =========================
# Constants
# =========================
PAD = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Hyperparameters
# =========================
vocab_size = 100
d_model = 32
num_steps = 200
warmup_steps = 50


# =========================
# Model
# =========================
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=d_model,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=64,
    max_seq_len=50,
    dropout_p=0.0,
    pad_idx=PAD,
).to(device)


# =========================
# Loss + Optimizer
# =========================
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

optimizer = Adam(
    model.parameters(),
    lr=1.0,  # will be scaled by scheduler
    betas=(0.9, 0.98),
    eps=1e-9
)


# =========================
# LR Scheduler
# =========================
def get_lr(step, d_model, warmup_steps):
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


# =========================
# Training Loop
# =========================
model.train()

for step in range(1, num_steps + 1):

    # ---- Dummy batch (fixed for overfitting test) ----
    src = torch.tensor([[5, 6, 7, 8, 9, 0]], device=device)
    tgt = torch.tensor([[1, 5, 6, 7, 8, 2]], device=device)

    # ---- Target shift ----
    tgt_input = tgt[:, :-1]     # [B, T-1]
    tgt_output = tgt[:, 1:]     # [B, T-1]

    # ---- Masks ----
    # Source mask: [B, S]
    src_mask = (src != PAD)

    # Target mask
    T_len = tgt_input.size(1)

    tgt_pad_mask = (tgt_input != PAD).unsqueeze(1)   # [B, 1, T]
    
    causal_mask = torch.tril(
        torch.ones(T_len, T_len, device=device)
    ).bool().unsqueeze(0)   # [1, T, T]

    tgt_mask = tgt_pad_mask & causal_mask   # [B, T, T]

    # ---- Forward ----
    logits = model(src, tgt_input, src_mask, tgt_mask)

    # ---- Loss ----
    B, T_out, V = logits.shape

    logits = logits.view(B * T_out, V)
    tgt_output = tgt_output.reshape(B * T_out)

    loss = criterion(logits, tgt_output)

    # ---- Backward ----
    loss.backward()

    # ---- Gradient clipping ----
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # ---- LR update ----
    lr = get_lr(step, d_model, warmup_steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # ---- Optimizer step ----
    optimizer.step()
    optimizer.zero_grad()

    # ---- Logging ----
    if step % 10 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.6f} | LR: {lr:.6f}")