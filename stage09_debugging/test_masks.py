# stage09_debugging/test_masks.py

import torch

def print_tgt_mask(tgt_mask):
    """
    PURPOSE:
    Print one sample of the target mask for visual inspection.

    EXPECTATION:
    - Should look like a lower-triangular matrix (causal mask).
    - Padding positions should be masked out (set to False).
    """
    print("[TGT MASK SAMPLE]")
    print(tgt_mask[0].int())


def assert_causal_mask(tgt_mask):
    """
    PURPOSE:
    Verify that the causal mask enforces autoregressive property.

    LOGIC:
    - for pos i, it should not attend to j if j > i.
    - used to ensure that no future tokens are visible during training.
    """

    B, T, _ = tgt_mask.shape
    for i in range(T):
        for j in range(T):
            if j > i:
                assert tgt_mask[0, i, j] == False, f"mask violation at position ({i},{j})"

    print("Causal structure correct")


def assert_padding_mask(tgt, tgt_mask, pad_id=0):
    """
    PURPOSE:
    Verify that padding tokens are fully blocked.

    LOGIC:
    - If tgt[j] is PAD, then column j in the mask should be all False.
    - That means no position i can attend to padding.
    """
    B, T = tgt.shape

    for j in range(T):
        if tgt[0, j].item() == pad_id:
            # Extract column j (all positions attending to token j)
            col = tgt_mask[0, :, j]

            # Sum should be 0 → all False
            assert col.sum().item() == 0, f"padding leak at position {j}"

    print("padding mask correct")


def extreme_mask_test(model, src, tgt, src_mask, tgt_mask):
    """
    PURPOSE:
    Ensure that masking actually affects model behavior.

    METHOD:
    - Compare output with normal mask vs fully blocked mask.
    - If masking is working, outputs should differ significantly.
    """

    model.eval()

    with torch.no_grad():
        # Normal forward pass
        normal_out, _ = model(src, tgt, src_mask, tgt_mask)

        # Fully blocked mask
        zero_tgt_mask = torch.zeros_like(tgt_mask).bool()

        # Masked forward pass
        masked_out, _ = model(src, tgt, src_mask, zero_tgt_mask)

        # Measure difference
        diff = (normal_out - masked_out).abs().mean().item()

        print(f"[extreme mask test] mean difference: {diff}")

        assert diff > 1e-3, \
            "Mask not affecting attention (no difference observed)"

    print("extreme mask test passed")