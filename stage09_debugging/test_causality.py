# stage09_debugging/test_causality.py

import torch


def causality_violation_test(model, src, tgt, src_mask, tgt_mask, vocab_size):
    """
    Verify that token at position t does NOT depend on tokens > t.

    METHOD:
    - Corrupt future tokens
    - Compare outputs at position t
    - Expect no change
    """

    model.eval()

    with torch.no_grad():
        B, T = tgt.shape

        # choose a stable middle position 
        t = T // 2

        # clone target
        tgt_corrupt = tgt.clone()

        # corrupt strictly future tokens
        # start from 4 to avoid PAD=0, SOS=1, EOS=2, UNK=3
        tgt_corrupt[:, t + 1:] = torch.randint(
            4, vocab_size, (B, T - (t + 1)), device=tgt.device
        )

        # forward passes
        out_clean, _ = model(src, tgt, src_mask, tgt_mask)
        out_corrupt, _ = model(src, tgt_corrupt, src_mask, tgt_mask)

        # compare only at position t
        diff_tensor = (out_clean[:, t] - out_corrupt[:, t]).abs()

        max_diff = diff_tensor.max().item()
        mean_diff = diff_tensor.mean().item()

        # diagnostic prints
        print("[CAUSALITY TEST]")
        print(f"sequence length: {T}")
        print(f"tested position: {t}")
        print(f"corrupted positions: {list(range(t + 1, T))}")
        print(f"max diff at position {t}: {max_diff:.6e}")
        print(f"mean diff at position {t}: {mean_diff:.6e}")

        # mask sanity check
        # catches common bug: mask built for shifted tgt but used on full tgt
        if tgt_mask is not None:
            print(f"tgt_mask shape: {tgt_mask.shape}")

            assert tgt_mask.shape[-1] == T, \
                "tgt_mask last dimension must equal sequence length"

            assert tgt_mask.shape[-2] == T, \
                "tgt_mask second last dimension must equal sequence length"

        # core assertion
        assert max_diff < 1e-5, (
            "causality broken: future tokens influence past\n"
            "check:"
            "1) mask applied before softmax\n"
            "2) mask broadcasting to heads\n"
            "3) mask semantics (True/False meaning)"
        )

    print("causality test passed")


def full_causality_sweep(model, src, tgt, src_mask, tgt_mask, vocab_size):
    """
    Tests causality across all positions instead of a single t.
    """

    model.eval()

    with torch.no_grad():
        B, T = tgt.shape

        for t in range(1, T - 1):
            tgt_corrupt = tgt.clone()

            tgt_corrupt[:, t + 1:] = torch.randint(
                4, vocab_size, (B, T - (t + 1)), device=tgt.device
            )

            out1, _ = model(src, tgt, src_mask, tgt_mask)
            out2, _ = model(src, tgt_corrupt, src_mask, tgt_mask)

            diff = (out1[:, t] - out2[:, t]).abs().max().item()

            print(f"[sweep] position {t}, diff {diff:.6e}")

            assert diff < 1e-5, f"leakage detected at position {t}"

    print("full causality sweep passed")