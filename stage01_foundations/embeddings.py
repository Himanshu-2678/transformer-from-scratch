import torch 
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    Convert integer token IDs into dense float vector.
    this is usually a learned lookup table. During training, PyTorch adjust
    these vectors by gradient descent so that semantically similar vector end
    up very close to each other.

    Args: vocab_size = V = total numbers of tokens in vocabulary.
          d_model = Embedding dimension. This is the core hyperparameter that
                    flows through the entire transformer.
                    (d in the paper: 512)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        # nn. Embedding is PyTorch's built-in lookup table.
        # Shape - [vocab_size, d_model]
        # padding_idx = 0 := no update for token "0".
        # PAD Tokens should have gradients as 0 so that they can 
        # stay as zero vector and don't pollute other embeddings

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0
        )

        self.d_model = d_model

        # computing the scaling factor - sqrt(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Integer token IDs, shape [Batch, SeqLen] (B, T)

        Returns:
            Float embeddings, shape [Batch, SeqLen, d_model] (B, T, d)
        """

        assert x.dim() == 2, (
            f"TokenEmbedding expects 2D input [B, T], got {x.dim()}D tensor of shape {x.shape}"
        )
        assert x.dtype == torch.long, (
            f"TokenEmbedding expects torch.long (int64) input, got {x.dtype}. "
            f"Token IDs must be integers, not floats."
        )
        print(f" [Embedding] input  x: {x.shape}  dtype={x.dtype}")

        output = self.embedding(x) * self.scale

        print(f"  [Embedding] output embeddings: {output.shape}  dtype={output.dtype}")
        # Expected: [B, T, d_model]. If you see [B, T] you forgot the embedding layer.
        # If you see [B, d_model] you forgot the sequence dimension (squeezed by mistake).

        return output
    
def demonstrate_embedding():

    torch.manual_seed(42) 

    # Hyperparameters
    vocab_size = 100    # Small vocab for demo
    d_model = 8         # Tiny d_model so we can print it
    batch_size = 2      # B = 2 sequences in the batch
    seq_len = 5         # T = 5 tokens per sequence

    embed = TokenEmbedding(vocab_size, d_model)

    # Simulate a batch of token IDs
    # Shape: [B, T] = [2, 5]
    # Row 0: token IDs [5, 12, 3, 7, 0]  (0 = PAD)
    # Row 1: token IDs [8, 3, 14, 2, 11]
    token_ids = torch.tensor([
        [5, 12, 3, 7, 0],
        [8,  3, 14, 2, 11],
    ])
    print(f"Input shape:  {token_ids.shape}")   # [2, 5]

    # Forward pass
    embeddings = embed(token_ids)
    print(f"Output shape: {embeddings.shape}")  # [2, 5, 8]

    # Verify: PAD token (ID=0) should have near-zero embedding
    # (padding_idx=0 ensures its gradient is always zero)
    print(f"\nPAD embedding (should be ~zero): {embeddings[0, 4, :]}")

    # Verify: same token in different positions has SAME embedding
    # Token ID 3 appears at position [0,2] and [1,1]
    # Without positional encoding, they ARE identical this is why PE is needed.
    same_token_pos1 = embeddings[0, 2, :]  # token 3 in sequence 0
    same_token_pos2 = embeddings[1, 1, :]  # token 3 in sequence 1
    print(f"\nSame token (ID=3) at two positions:")
    print(f"  Pos [0,2]: {same_token_pos1.detach()}")
    print(f"  Pos [1,1]: {same_token_pos2.detach()}")
    print(f" Checking ifthey are equal? {torch.allclose(same_token_pos1, same_token_pos2)}")
    # This SHOULD print True and this is the problem PE solves!

    return embeddings



### Tensor Shape Walkthrough

"""Here is every shape, step by step:

Input token_ids:     [B, T]          e.g. [2, 5]
                      │   │
                      │   └── SeqLen (number of tokens per sequence)
                      └────── Batch size

After nn.Embedding:  [B, T, d_model] e.g. [2, 5, 8]
                      │   │   │
                      │   │   └── Embedding dimension (the "richness" of each token's repr)
                      │   └────── SeqLen (preserved one vector per token)
                      └────────── Batch size (preserved each sequence independently looked up)

After * scale:       [B, T, d_model]  (same shape, all values scaled)"""