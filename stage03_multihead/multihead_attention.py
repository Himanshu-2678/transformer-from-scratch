# stage03_multihead/multihead_attention.py

import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from stage02_attention.attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0, verbose: bool = False):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.verbose = verbose

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(
            dropout_p=dropout_p,
            verbose=verbose
        )

    def forward(self, Q, K, V, mask=None):

        assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3
        assert Q.shape[0] == K.shape[0] == V.shape[0]
        assert Q.shape[2] == K.shape[2] == V.shape[2] == self.d_model
        assert K.shape[1] == V.shape[1]

        B, T_q, _ = Q.shape
        T_k = K.shape[1]
        h = self.num_heads

        if self.verbose:
            print(f"[MHA] Input -> Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

        # projections
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # split heads
        Q = Q.view(B, T_q, h, self.d_k).transpose(1, 2)
        K = K.view(B, T_k, h, self.d_k).transpose(1, 2)
        V = V.view(B, T_k, h, self.d_k).transpose(1, 2)

        # flatten heads into batch
        Q = Q.reshape(B * h, T_q, self.d_k)
        K = K.reshape(B * h, T_k, self.d_k)
        V = V.reshape(B * h, T_k, self.d_k)

        # mask handling
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                attn_mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                attn_mask = mask.unsqueeze(1)
            else:
                attn_mask = mask

            assert attn_mask.shape[-1] == T_k

            attn_mask = attn_mask.expand(B, h, T_q, T_k).reshape(B * h, T_q, T_k)

        # attention
        out, weights = self.attention(Q, K, V, attn_mask)

        # restore heads
        out = out.reshape(B, h, T_q, self.d_k)
        out = out.transpose(1, 2)
        out = out.reshape(B, T_q, self.d_model)

        weights = weights.reshape(B, h, T_q, T_k)

        # final projection
        out = self.W_O(out)

        return out, weights
    

def demonstrate_multihead_attention():
    print("=" * 60)
    print("Multi-Head Attention - shape verification")
    print("=" * 60)

    torch.manual_seed(42)

    d_model = 16
    num_heads = 4   # each head gets d_k = 4

    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        verbose=True
    )

    # Case 1: Self-attention (Q = K = V)
    print("\nCase 1: Self-Attention")

    B, T = 2, 5
    x = torch.randn(B, T, d_model)

    out, w = mha(x, x, x)

    print(f"Expected out: [{B}, {T}, {d_model}]")
    print(f"Got out:      {list(out.shape)}")

    print(f"Expected weights: [{B}, {num_heads}, {T}, {T}]")
    print(f"Got weights:      {list(w.shape)}")

    assert out.shape == (B, T, d_model)
    assert w.shape == (B, num_heads, T, T)

    print("Self-attention passed")

    # Case 2: Cross-attention (different Q and K/V lengths)
    print("\nCase 2: Cross-Attention")

    B, T_q, T_k = 2, 3, 7
    q = torch.randn(B, T_q, d_model)
    kv = torch.randn(B, T_k, d_model)

    out, w = mha(q, kv, kv)

    print(f"Expected out: [{B}, {T_q}, {d_model}]")
    print(f"Got out:      {list(out.shape)}")

    print(f"Expected weights: [{B}, {num_heads}, {T_q}, {T_k}]")
    print(f"Got weights:      {list(w.shape)}")

    assert out.shape == (B, T_q, d_model)
    assert w.shape == (B, num_heads, T_q, T_k)

    print("Cross-attention passed")

    # Case 3: Masked self-attention (causal)
    print("\nCase 3: Masked Self-Attention")

    B, T = 2, 5
    x = torch.randn(B, T, d_model)

    # lower triangular mask: each token sees only past and itself
    mask = torch.tril(torch.ones(T, T))
    mask = mask.unsqueeze(0).expand(B, -1, -1)

    out, w = mha(x, x, x, mask=mask)

    print(f"Expected out: [{B}, {T}, {d_model}]")
    print(f"Got out:      {list(out.shape)}")

    assert out.shape == (B, T, d_model),      "Masked self-attn output shape wrong"
    assert w.shape   == (B, num_heads, T, T), "Masked self-attn weights shape wrong"

    # verify masking: no attention to future tokens
    forbidden = w.triu(diagonal=1)
    max_val = forbidden.abs().max().item()

    print(f"Max forbidden attention: {max_val:.6f}")

    assert max_val < 1e-6

    print("Causal mask passed")

    print("\nCase 4: Invalid Init")
    try:
        MultiHeadAttention(d_model=15, num_heads=4)
        print("ERROR - It should have failed")
    except AssertionError:
        print("Correctly rejected d_model=15, num_heads=4")

    print("\n" + "=" * 60)
    print("All checks passed")
    print("=" * 60)

# Driver Code
if __name__ == "__main__":
    demonstrate_multihead_attention()