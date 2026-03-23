# stage01_foundations/run_stage1.py

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from stage01_foundations.tokenizer import CharacterTokenizer
from stage01_foundations.embeddings import TokenEmbedding
from stage01_foundations.softmax_crossentropy import CrossEntropyLoss
from stage01_foundations.training_loop import train_one_epoch, evaluate


# -----------------------------------------------------------------------------
# hyperparameters -- all in one place so we're not hunting through the file
# -----------------------------------------------------------------------------

D_MODEL       = 128
BATCH_SIZE    = 32
MAX_LEN       = 32        # sequence length
NUM_EPOCHS    = 10
LR            = 1e-3
DROPOUT       = 0.1
LABEL_SMOOTH  = 0.1
CLIP_NORM     = 1.0
VAL_SPLIT     = 0.1       # 10% of data goes to validation
NUM_SAMPLES   = 2000      # how many random sequences to generate
SEED          = 42
VERBOSE       = False

CHECKPOINT_DIR = "experiments/run_01/checkpoints"


# -----------------------------------------------------------------------------
# simple model - just enough to produce logits of the right shape.
# TokenEmbedding gives us [B, T, d_model], linear projects to [B, T, V].
# no attention, no encoder, no decoder. that all comes later.
# -----------------------------------------------------------------------------

class SimpleModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout_p: float):
        super().__init__()
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            dropout_p=dropout_p,
            verbose=False,
        )
        # projects from embedding space back to vocab size.
        # this is what produces the raw logits.
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x      : [B, T]
        # embedded: [B, T, d_model]
        # logits : [B, T, vocab_size]
        embedded = self.embedding(x)
        logits   = self.proj(embedded)
        return logits


# -----------------------------------------------------------------------------
# toy data - random token sequences where the target is the input shifted
# one position to the right. basically asking the model to predict the next
# token at every position. simple enough to not distract from the pipeline.
# -----------------------------------------------------------------------------

def make_toy_data(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> TensorDataset:
    torch.manual_seed(seed)

    # random token IDs - we keep values between 4 and vocab_size-1
    # so we don't accidentally generate PAD/SOS/EOS/UNK as content tokens
    src = torch.randint(4, vocab_size, (num_samples, seq_len))

    # target is src shifted left by one, last position filled with PAD
    # so at each position, the target is "what comes next"
    pad_col = torch.zeros(num_samples, 1, dtype=torch.long)
    tgt = torch.cat([src[:, 1:], pad_col], dim=1)

    return TensorDataset(src, tgt)


# -----------------------------------------------------------------------------
# checkpointing - saves everything we'd need to resume or analyze later
# -----------------------------------------------------------------------------

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    val_loss: float,
    checkpoint_dir: str,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss":           train_loss,
            "val_loss":             val_loss,
        },
        path,
    )
    print(f"  checkpoint saved -> {path}")


# --------------- main -------------
def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on: {device}")

    # --- tokenizer ---
    # we build a small corpus just so the tokenizer has something to fit on.
    # in a real setup this would be actual text data.
    corpus = ["hello world", "transformer from scratch", "attention is all you need"]
    tokenizer = CharacterTokenizer(lowercase=True)
    tokenizer.build_vocab(corpus)
    vocab_size = tokenizer.vocab_size
    print(f"vocab size: {vocab_size}")

    # --- toy dataset ---
    dataset = make_toy_data(
        num_samples=NUM_SAMPLES,
        seq_len=MAX_LEN,
        vocab_size=vocab_size,
        seed=SEED,
    )

    # split into train and val
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    # --- model ---
    model = SimpleModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        dropout_p=DROPOUT,
    ).to(device)

    # --- loss ---
    loss_fn = CrossEntropyLoss(
        vocab_size=vocab_size,
        pad_id=0,
        label_smoothing=LABEL_SMOOTH,
        verbose=VERBOSE,
    )

    # --- optimizer ---
    # Adam is what the paper uses. we'll tune betas and warmup in stage 8.
    # for now default Adam is fine.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- training loop ---
    print("\nstarting training...\n")
    best_val_loss = float("inf")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            clip_grad_norm=CLIP_NORM,
            verbose=VERBOSE,
        )

        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            verbose=VERBOSE,
        )

        elapsed = time.time() - t0

        print(
            f"epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"time: {elapsed:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_dir=CHECKPOINT_DIR,
            )
            print(f"  best model updated (val_loss={val_loss:.4f})")

    print("\nstage 1 complete.")


if __name__ == "__main__":
    main()

