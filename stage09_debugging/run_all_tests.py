# stage09_debugging/run_all_tests.py

from stage09_debugging.test_causality import (
    causality_violation_test,
    full_causality_sweep
)

from stage09_debugging.test_masks import (
    print_tgt_mask,
    assert_causal_mask,
    assert_padding_mask,
    extreme_mask_test
)

from stage09_debugging.test_attention import (
    check_attention_causality,
    attention_row_sums,
    attention_entropy,
    attention_mask_alignment
)

from stage09_debugging.test_attention_behavior import test_attention_behavior

def run_all_tests(
    model,
    src,
    tgt,
    src_mask,
    tgt_mask,
    vocab_size,
    attn_weights=None
):
    print("\n===== DEBUGGING START =====\n")

    # STEP 1: MASK STRUCTURE
    print("[STEP 1] MASK STRUCTURE")
    print_tgt_mask(tgt_mask)
    assert_causal_mask(tgt_mask)
    assert_padding_mask(tgt, tgt_mask)

    # STEP 2: MASK ENFORCEMENT
    print("\n[STEP 2] MASK ENFORCEMENT")
    extreme_mask_test(model, src, tgt, src_mask, tgt_mask)

    # STEP 3: CAUSALITY
    print("\n[STEP 3] CAUSALITY")
    causality_violation_test(
        model, src, tgt, src_mask, tgt_mask, vocab_size
    )
    full_causality_sweep(
        model, src, tgt, src_mask, tgt_mask, vocab_size
    )

    # STEP 4: ATTENTION
    print("\n[STEP 4] ATTENTION")
    test_attention_behavior(model, src, tgt, src_mask, tgt_mask)

    print("\n===== ALL TESTS PASSED =====\n")


# -------------------------------
# DRIVER
# -------------------------------
if __name__ == "__main__":
    print("running stage 9 debug tests")

    import torch

    B, T, vocab_size, d_model = 2, 6, 50, 32
    PAD = 0

    # -------------------------------
    # DATA
    # -------------------------------
    src = torch.randint(4, vocab_size, (B, T))
    tgt = torch.randint(4, vocab_size, (B, T))

    # inject padding (both src and tgt)
    tgt[0, -2:] = PAD
    src[0, -1:] = PAD

    # -------------------------------
    # MASKS
    # -------------------------------

    # src mask: block PAD tokens
    src_mask = (src != PAD)  # [B, S]

    # causal mask
    causal = torch.tril(torch.ones(T, T)).bool()  # [T, T]

    # padding mask (keys)
    key_pad = (tgt != PAD).unsqueeze(1)  # [B, 1, T]

    # padding mask (queries)
    query_pad = (tgt != PAD).unsqueeze(2)  # [B, T, 1]

    # final tgt mask
    tgt_mask = causal.unsqueeze(0) & key_pad & query_pad  # [B, T, T]

    # enforce bool
    src_mask = src_mask.bool()
    tgt_mask = tgt_mask.bool()

    # -------------------------------
    # MODEL
    # -------------------------------
    from models.transformer import Transformer

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        max_seq_len=T,
        dropout_p=0.0,
        pad_idx=PAD
    )

    # -------------------------------
    # RUN TESTS
    # -------------------------------
    run_all_tests(
        model=model,
        src=src,
        tgt=tgt,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        vocab_size=vocab_size,
        attn_weights=None
    )