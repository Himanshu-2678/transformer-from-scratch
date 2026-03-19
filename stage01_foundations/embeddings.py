# stage01_foundations/embeddings.py

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Converts integer token IDs into dense float vectors.

    Think of this as a lookup table of shape [vocab_size, d_model].
    Every token in our vocabulary gets its own row which is a learned vector
    that captures what that token means. During training, gradient descent
    nudges these vectors so that tokens used in similar contexts end up
    close together in this d_model-dimensional space.

    The scaling by sqrt(d_model) is from the original paper. Without it,
    embeddings tend to be much smaller in magnitude than the positional
    encodings we add in Stage 4, and the model loses positional information.
    Scaling brings them to the same order of magnitude.

    Args:
        vocab_size: Total number of tokens V. Get this from tokenizer.vocab_size.
        d_model:    Embedding dimension. Every single layer in the transformer
                    uses this same number. It's the backbone hyperparameter.
                    Paper uses 512. We'll use smaller values for learning.
        dropout_p:  Dropout probability applied after embedding + scale.
                    Paper uses 0.1. Set 0.0 to disable during debugging.
        verbose:    If True, print shapes during forward(). Turn off for training.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout_p: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__()

        # The actual lookup table. Shape: [vocab_size, d_model].
        # padding_idx=0 tells PyTorch: never update the row for token ID 0.
        # That's our PAD token. We don't want gradients flowing through padding
        # positions and corrupting the real token embeddings.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0,
        )

        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.verbose = verbose

        # Dropout applied after the embedding + scale step.
        # This is where it lives in the paper's architecture before the
        # first encoder/decoder layer sees the input.
        # During eval(), nn.Dropout automatically becomes a no-op.
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs, shape [B, T], dtype=torch.long

        Returns:
            Embeddings, shape [B, T, d_model], dtype=torch.float32
        """

        x = x.to(self.embedding.weight.device)

        assert x.dim() == 2, (
            f"Expected 2D input [B, T], got shape {x.shape}"
        )
        assert x.dtype == torch.long, (
            f"Expected torch.long (int64), got {x.dtype}. "
            f"Token IDs must be integers."
        )

        if self.verbose:
            print(f"  [Embedding] input  : {x.shape}  dtype={x.dtype}")

        # Lookup + scale. Shape goes [B, T] -> [B, T, d_model].
        out = self.embedding(x) * self.scale

        # Apply dropout. Shape stays [B, T, d_model].
        out = self.dropout(out)

        if self.verbose:
            print(f"  [Embedding] output : {out.shape}  dtype={out.dtype}")

        return out


# ── Shape walkthrough ─────────────────────────────────────────────────────────
#
# Input x:          [B, T]            e.g. [2, 5]
#                    |   |
#                    |   └── T: tokens per sequence
#                    └────── B: batch size
#
# After embedding:  [B, T, d_model]   e.g. [2, 5, 8]
#                    |   |   |
#                    |   |   └── d_model: embedding dimension
#                    |   └────── T: one vector per token (preserved)
#                    └────────── B: each sequence looked up independently
#
# After * scale:    [B, T, d_model]   same shape, values scaled by sqrt(d)
# After dropout:    [B, T, d_model]   same shape, some values zeroed randomly
# ─────────────────────────────────────────────────────────────────────────────


def demonstrate_embedding():
    torch.manual_seed(42)

    vocab_size = 100
    d_model    = 8
    batch_size = 2
    seq_len    = 5

    # verbose=True so we can see shape prints during the demo
    embed = TokenEmbedding(vocab_size, d_model, dropout_p=0.0, verbose=True)

    # Shape: [B, T] = [2, 5]
    # Position [0, 4] = token ID 0 -> this is a PAD token
    # Token ID 3 appears at [0,2] and [1,1] -> same embedding, different position
    token_ids = torch.tensor([
        [5, 12, 3, 7,  0],
        [8,  3, 14, 2, 11],
    ])

    print(f"token_ids shape : {token_ids.shape}")
    print(f"token_ids dtype : {token_ids.dtype}")
    print()

    # Put model in eval mode so dropout doesn't zero anything during demo
    embed.eval()
    with torch.no_grad():
        embeddings = embed(token_ids)

    print(f"\nFinal output shape : {embeddings.shape}")  # [2, 5, 8]

    # PAD token (ID=0) should be exactly zero: padding_idx=0 keeps it frozen
    print(f"\nPAD embedding (expect all zeros) :\n  {embeddings[0, 4, :]}")

    # Same token ID at two different positions -> identical embeddings
    # This is the core problem that positional encoding (Stage 4) solves.
    # Without PE, the model has no idea if "cat" appears first or last.
    pos1 = embeddings[0, 2, :]   # token ID 3, sequence 0
    pos2 = embeddings[1, 1, :]   # token ID 3, sequence 1
    print(f"\nToken ID=3 at position [0,2] : {pos1}")
    print(f"Token ID=3 at position [1,1] : {pos2}")
    print(f"Equal? = {torch.allclose(pos1, pos2)}")
    print("-> They should be. Same token = same embedding = no position info.")
    print("  This is exactly what Stage 4 (Positional Encoding) fixes.")


if __name__ == "__main__":
    demonstrate_embedding()
