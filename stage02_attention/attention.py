# stage02_attention/attention.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Core attention: consumes Q, K, V (no projections here)."""

    def __init__(self, dropout_p: float = 0.0, verbose: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.verbose = verbose

    def forward(
        self,
        Q: torch.Tensor,           # [B, T_q, d_k]
        K: torch.Tensor,           # [B, T_k, d_k]
        V: torch.Tensor,           # [B, T_k, d_v]
        mask: torch.Tensor | None = None,
        head_ablation=None,
        num_heads=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # --- shape checks ---
        assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3  # fix 1
        assert Q.shape[0] == K.shape[0] == V.shape[0]
        assert Q.shape[2] == K.shape[2]          # d_k match
        assert K.shape[1] == V.shape[1]          # T_k match

        B, T_q, d_k = Q.shape
        T_k = K.shape[1]

        if self.verbose:
            print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

        # --- attention scores ---
        scores = Q @ K.transpose(-2, -1)         # [B, T_q, T_k]
        scores = scores / math.sqrt(d_k)

        # --- mask ---
        if mask is not None:
            mask = mask.to(Q.device).bool()

            if mask.dim() == 2:
                # [B, T_k] → [B, T_q, T_k]
                mask = mask.unsqueeze(1).expand(-1, T_q, -1)

            elif mask.dim() == 3:
                # must already be [B, T_q, T_k]
                assert mask.shape[1] == T_q

            assert mask.shape[-1] == T_k

            # mask scores BEFORE softmax: use -inf instead of -1e9
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        # --- weights ---
        weights = F.softmax(scores, dim=-1)

        if mask is not None:
            # detect rows where all keys are masked
            valid_counts = mask.sum(dim=-1, keepdim=True)  # [B, T_q, 1]
            fully_masked = (valid_counts == 0)

            # expand to match weights
            fully_masked = fully_masked.expand_as(weights)

            # zero those rows
            weights = torch.where(
                fully_masked,
                torch.zeros_like(weights),
                weights
            )

        # ---- HEAD ABLATION ----
        if head_ablation is not None:
            assert num_heads is not None, "num_heads required for ablation"

            B_h = weights.shape[0]   # = B * H
            T_q, T_k = weights.shape[1], weights.shape[2]

            H = num_heads
            B = B_h // H

            # reshape to recover head dimension
            weights = weights.view(B, H, T_q, T_k)

            # zero selected heads
            weights[:, head_ablation, :, :] = 0.0

            # flatten back
            weights = weights.view(B * H, T_q, T_k)

        weights = self.dropout(weights)

        # --- output ---
        output = weights @ V                     # [B, T_q, d_v]

        if self.verbose:
            print(f"weights: {weights.shape}, output: {output.shape}")  # fix 4

        return output, weights


def demonstrate_attention():
    torch.manual_seed(42)
    B, T, d_k, d_v = 1, 3, 4, 4

    Q = torch.randn(B, T, d_k)
    K = torch.randn(B, T, d_k)
    V = torch.randn(B, T, d_v)

    attn = ScaledDotProductAttention(dropout_p=0.0, verbose=True)

    print("--- no mask ---")
    output, weights = attn(Q, K, V)
    print("weights:\n", weights[0].detach().round(decimals=3))
    print("row sums:", weights[0].sum(dim=-1).tolist())

    print("\n--- causal mask ---")
    mask = torch.tril(torch.ones(B, T, T))
    output, weights = attn(Q, K, V, mask=mask)
    print("weights:\n", weights[0].detach().round(decimals=3))
    print("row sums:", weights[0].sum(dim=-1).tolist())

    upper = weights[0].triu(diagonal=1)
    assert upper.abs().max().item() < 1e-6, "causal mask failed"
    print("\nCausal check passed.")

    # fix 3 — cross-attention shape: T_q != T_k
    print("\n--- cross-attention (T_q=2, T_k=3) ---")
    Q_cross = torch.randn(B, 2, d_k)
    K_cross = torch.randn(B, 3, d_k)
    V_cross = torch.randn(B, 3, d_v)
    output_cross, weights_cross = attn(Q_cross, K_cross, V_cross)
    print(f"output: {output_cross.shape}")    # expect [1, 2, 4]
    print(f"weights: {weights_cross.shape}")  # expect [1, 2, 3]
    print("row sums:", weights_cross[0].sum(dim=-1).tolist())
    print("\nCross-attention check passed.")


if __name__ == "__main__":
    demonstrate_attention()