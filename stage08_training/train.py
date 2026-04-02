# stage08_training/train.py

import torch
import torch.nn as nn
from torch.optim import Adam

from models.transformer import Transformer
from stage10_analysis.synthetic_tasks import generate_copy_task

# =========================
# Constants
# =========================
PAD = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Hyperparameters
# =========================
vocab_size = 20          # reduced for faster learning
d_model = 32
num_steps = 2500

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
    lr=1e-3,   # fixed LR (stable)
    betas=(0.9, 0.98),
    eps=1e-9
)

# =========================
# Training Loop
# =========================
model.train()

for step in range(1, num_steps + 1):

    # ---- Synthetic COPY task ----
    src, tgt = generate_copy_task(
        batch_size=64,
        seq_len=6,
        vocab_size=vocab_size
    )

    src = src.to(device)
    tgt = tgt.to(device)

    # ---- Target shift ----
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    # ---- Masks ----
    src_mask = (src != PAD)

    T_len = tgt_input.size(1)

    tgt_pad_mask = (tgt_input != PAD).unsqueeze(1)

    causal_mask = torch.tril(
        torch.ones(T_len, T_len, device=device)
    ).bool().unsqueeze(0)

    tgt_mask = tgt_pad_mask & causal_mask

    # ---- Forward ----
    logits = model(src, tgt_input, src_mask, tgt_mask)

    if isinstance(logits, tuple):
        logits = logits[0]

    # ---- Loss ----
    B, T_out, V = logits.shape

    logits = logits.reshape(B * T_out, V)
    tgt_output = tgt_output.reshape(B * T_out)

    loss = criterion(logits, tgt_output)

    # ---- Backward ----
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad()

    # ---- Logging ----
    if step % 50 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.6f}")

# =========================
# Save model
# =========================
torch.save(model.state_dict(), "checkpoint.pt")
print("Checkpoint saved.")