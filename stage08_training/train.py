# stage08_training/train.py

import torch
import torch.nn as nn
from torch.optim import Adam

from models.transformer import Transformer
from stage10_analysis.synthetic_tasks import generate_copy_task
from stage10_analysis.tasks.reverse import generate_reverse_task

# =========================
# Constants
# =========================
PAD = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Hyperparameters
# =========================
vocab_size = 20
d_model = 32
num_steps = 2000

# =========================
# EXPERIMENT CONFIG
# =========================
TASK = "kv"            # "copy" or "reverse" or "kv"
SEQ_LEN = 16
SAVE_PATH = "checkpoint_kv.pt"   # change per run - checkpoint_kv/reverse/copy.pt

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
    lr=1e-3,
    betas=(0.9, 0.98),
    eps=1e-9
)

# =========================
# Training Loop
# =========================
model.train()

for step in range(1, num_steps + 1):

    # ---- Data Generation ----
    if TASK == "copy":
        src, tgt = generate_copy_task(
            batch_size=64,
            seq_len=SEQ_LEN,
            vocab_size=vocab_size
        )

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

    elif TASK == "reverse":
        src, tgt_input, tgt_output = generate_reverse_task(
            batch_size=64,
            seq_len=SEQ_LEN,
            vocab_size=vocab_size,
            device=device
        )

    elif TASK == "kv":
        from stage10_analysis.tasks.kv import generate_kv_task

        src, tgt_input, tgt_output = generate_kv_task(
            batch_size=64,
            num_pairs=4,
            vocab_size=vocab_size,
            device=device
        )

    else:
        raise ValueError("Invalid TASK")

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
# Debug Sample
# =========================
model.eval()

if TASK == "copy":
    src, tgt = generate_copy_task(
        batch_size=1,
        seq_len=SEQ_LEN,
        vocab_size=vocab_size
    )

    src = src.to(device)
    tgt = tgt.to(device)

    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

elif TASK == "kv":
    src, tgt_input, tgt_output = generate_kv_task(
        batch_size=1,
        num_pairs=4,
        vocab_size=vocab_size,
        device=device
    )

else:
    src, tgt_input, tgt_output = generate_reverse_task(
        batch_size=1,
        seq_len=SEQ_LEN,
        vocab_size=vocab_size,
        device=device
    )

# ---- Masks ----
src_mask = (src != PAD)

T_len = tgt_input.size(1)

tgt_pad_mask = (tgt_input != PAD).unsqueeze(1)

causal_mask = torch.tril(
    torch.ones(T_len, T_len, device=device)
).bool().unsqueeze(0)

tgt_mask = tgt_pad_mask & causal_mask

# ---- Forward ----
with torch.no_grad():
    logits = model(src, tgt_input, src_mask, tgt_mask)

    if isinstance(logits, tuple):
        logits = logits[0]

    pred = logits.argmax(dim=-1)

print("\n=== TEST SAMPLE ===")
print("SRC:     ", src[0].tolist())
print("TARGET:  ", tgt_output[0].tolist())
print("PREDICT: ", pred[0].tolist())

# =========================
# Save model
# =========================
torch.save(model.state_dict(), SAVE_PATH)
print(f"Checkpoint saved to {SAVE_PATH}")