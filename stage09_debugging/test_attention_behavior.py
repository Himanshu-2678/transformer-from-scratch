# stage09_debugging/test_attention_behavior.py

import torch

def test_strict_future_mask(weights, tgt_mask):
    """
    Ensure NO attention to future tokens (hard zero check).
    """
    B, h, T_q, T_k = weights.shape

    for t in range(T_q):
        future = weights[:, :, t, t+1:]

        if future.numel() > 0:
            assert torch.allclose(
                future,
                torch.zeros_like(future),
                atol=1e-6
            ), f"Future leakage at position {t}"

    print("strict future masking passed")


def test_row_sum(weights, tgt_mask):
    """
    Check attention probability normalization.
    """

    row_sum = weights.sum(dim=-1)  # [B, h, T_q]

    valid_rows = tgt_mask.sum(dim=-1) > 0  # [B, T_q]

    # expand for heads
    valid_rows = valid_rows.unsqueeze(1).expand_as(row_sum)

    # valid rows → sum ≈ 1
    assert torch.allclose(
        row_sum[valid_rows],
        torch.ones_like(row_sum[valid_rows]),
        atol=1e-5
    ), "Row sum not 1 for valid rows"

    # fully masked rows → sum == 0
    assert torch.all(
        row_sum[~valid_rows] == 0
    ), "Fully masked rows not zero"

    print("row sum check passed")


def test_entropy(weights):
    """
    Detect degenerate attention (too uniform).
    """

    eps = 1e-9
    entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)  # [B, h, T_q]

    mean_entropy = entropy.mean().item()

    print(f"[entropy] mean entropy: {mean_entropy:.4f}")

    # heuristic threshold
    assert mean_entropy < 5.0, \
        "Attention too uniform (likely not learning meaningful patterns)"

    print("entropy check passed")


def test_attention_behavior(model, src, tgt, src_mask, tgt_mask):
    """
    Run all attention behavior tests.
    """
    print(">>> ATTENTION TEST IS RUNNING <<<")
    
    model.eval()

    with torch.no_grad():
        # You must modify model to return attention weights
        # Example expected:
        # out, attn_weights = model(...)
        # where attn_weights = [B, h, T_q, T_k]

        out, attn_weights = model(src, tgt, src_mask, tgt_mask)

        assert attn_weights is not None, \
            "Model must return attention weights for this test"

        print("\n[ATTENTION BEHAVIOR TEST]")

        test_strict_future_mask(attn_weights, tgt_mask)
        test_row_sum(attn_weights, tgt_mask)
        test_entropy(attn_weights)

        print("all attention behavior tests passed")