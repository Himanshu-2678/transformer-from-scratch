# stage06_decoder/runstage06.py

import torch
from stage06_decoder.decoder import DecoderLayer, Decoder, make_causal_mask


def test1():
    print("=== Test 1: DecoderLayer Basic Shape ===")
    x = torch.randn(2, 5, 16)
    memory = torch.randn(2, 7, 16)
    layer = DecoderLayer(d_model=16, num_heads=4, d_ff=64)
    out = layer(x, memory)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print(f"Input x: {x.shape}, memory: {memory.shape}")
    print(f"Output: {out.shape}")
    print("Passed")


def test2():
    print("\n=== Test 2: T_tgt and T_src Can Differ ===")
    x = torch.randn(2, 5, 16)
    memory = torch.randn(2, 11, 16)
    decoder = Decoder(num_layers=2, d_model=16, num_heads=4, d_ff=64)
    out = decoder(x, memory)
    assert out.shape == torch.Size([2, 5, 16]), f"Shape mismatch: {out.shape}"
    print(f"T_tgt=5, T_src=11 -> output: {out.shape}")
    print("Passed")


def test3():
    print("\n=== Test 3: Causal Mask Does Not Crash ===")
    x = torch.randn(2, 5, 16)
    memory = torch.randn(2, 7, 16)
    tgt_mask = make_causal_mask(5)
    decoder = Decoder(num_layers=2, d_model=16, num_heads=4, d_ff=64)
    out = decoder(x, memory, tgt_mask=tgt_mask)
    assert out.shape == torch.Size([2, 5, 16])
    print(f"Output: {out.shape}")
    print("Passed")


def test4():
    print("\n=== Test 4: src_mask Does Not Crash ===")
    x = torch.randn(2, 5, 16)
    memory = torch.randn(2, 7, 16)
    src_mask = torch.ones(2, 7)
    src_mask[0, 5:] = 0
    decoder = Decoder(num_layers=2, d_model=16, num_heads=4, d_ff=64)
    out = decoder(x, memory, src_mask=src_mask)
    assert out.shape == torch.Size([2, 5, 16])
    print(f"Output: {out.shape}")
    print("Passed")


def test5():
    print("\n=== Test 5: Both Masks Together ===")
    x = torch.randn(2, 5, 16)
    memory = torch.randn(2, 7, 16)
    tgt_mask = make_causal_mask(5)
    src_mask = torch.ones(2, 7)
    src_mask[0, 5:] = 0
    decoder = Decoder(num_layers=3, d_model=16, num_heads=4, d_ff=64)
    out = decoder(x, memory, tgt_mask=tgt_mask, src_mask=src_mask)
    assert out.shape == torch.Size([2, 5, 16])
    print(f"Output: {out.shape}")
    print("Passed")


def test6():
    print("\n=== Test 6: make_causal_mask Values ===")
    mask = make_causal_mask(5)
    print("Causal mask:\n", mask)
    assert mask.shape == torch.Size([1, 5, 5])

    # lower triangle must all be 1
    assert mask.tril().eq(mask).all(), "Lower triangle should be all ones"
    # upper triangle must all be 0
    upper = mask.triu(diagonal=1)
    assert upper.sum() == 0, "Upper triangle should be all zeros"
    print("Passed")


def test7():
    print("\n=== Test 7: Verbose Mode ===")
    x = torch.randn(2, 5, 16)
    memory = torch.randn(2, 7, 16)
    print("\nverbose=True:")
    decoder = Decoder(num_layers=2, d_model=16, num_heads=4, d_ff=64, verbose=True)
    _ = decoder(x, memory)
    print("\nverbose=False:")
    decoder = Decoder(num_layers=2, d_model=16, num_heads=4, d_ff=64, verbose=False)
    _ = decoder(x, memory)
    print("If nothing printed above for second run, verbose works")


# -----------------------------------------------------------
# New Tests (Behavioral + Gradients)
# -----------------------------------------------------------

def test8():
    print("\n=== Test 8: Causality Check ===")

    B, T, D = 1, 5, 16

    x = torch.randn(B, T, D)
    memory = torch.randn(B, 6, D)

    decoder = Decoder(num_layers=1, d_model=D, num_heads=4, d_ff=64)
    decoder.eval()  # disables dropout

    mask = make_causal_mask(T)

    out1 = decoder(x, memory, tgt_mask=mask)

    x_mod = x.clone()
    x_mod[:, -1, :] += 1000

    out2 = decoder(x_mod, memory, tgt_mask=mask)

    diff = (out1[:, :-1] - out2[:, :-1]).abs().mean().item()

    print("Difference (earlier tokens):", diff)
    assert diff < 1e-5, "Future token affected past → mask broken"
    print("Passed")


def test9():
    print("\n=== Test 9: Cross Attention Dependency ===")

    x = torch.randn(2, 5, 16)
    memory1 = torch.randn(2, 7, 16)
    memory2 = torch.randn(2, 7, 16) + 10  # shifted distribution

    decoder = Decoder(num_layers=2, d_model=16, num_heads=4, d_ff=64)

    out1 = decoder(x, memory1)
    out2 = decoder(x, memory2)

    diff = (out1 - out2).abs().sum()

    print("Output difference:", diff.item())

    assert diff > 1e-5, "Decoder is ignoring encoder output"
    print("Passed")


def test10():
    print("\n=== Test 10: Backward Pass ===")

    x = torch.randn(2, 5, 16, requires_grad=True)
    memory = torch.randn(2, 7, 16)

    decoder = Decoder(num_layers=2, d_model=16, num_heads=4, d_ff=64)

    out = decoder(x, memory)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradients did not flow back to input"
    print("Passed")


def test11():
    print("\n=== Test 11: src_mask Blocks Encoder Tokens ===")

    x = torch.randn(1, 5, 16)
    memory = torch.randn(1, 7, 16)

    decoder = Decoder(num_layers=1, d_model=16, num_heads=4, d_ff=64)
    decoder.eval()

    src_mask = torch.ones(1, 7)
    src_mask[:, 5:] = 0  # block positions 5 and 6

    # run 1: original memory with mask applied
    out1 = decoder(x, memory, src_mask=src_mask)

    # run 2: positions 5-6 modified heavily, same mask applied
    memory_mod = memory.clone()
    memory_mod[:, 5:, :] += 1000
    out2 = decoder(x, memory_mod, src_mask=src_mask)

    diff = (out1 - out2).abs().mean().item()
    print("Difference:", diff)
    assert diff < 1e-3, "Masked encoder positions are still influencing output"
    print("Passed")


def test12():
    print("\n=== Test 12: Unmasked encoder tokens DO influence ===")

    x = torch.randn(1, 5, 16)
    memory = torch.randn(1, 7, 16)

    decoder = Decoder(num_layers=1, d_model=16, num_heads=4, d_ff=64)
    decoder.eval()

    # NO MASK
    out1 = decoder(x, memory)

    memory_mod = memory.clone()
    memory_mod[:, 5:, :] += 1000

    out2 = decoder(x, memory_mod)

    diff = (out1 - out2).abs().mean().item()
    print("Difference:", diff)

    assert diff > 0.0, "Unmasked tokens are not influencing output"
    print("Passed")

if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()
    test9()
    test10()
    test11()
    test12()