# stage05_encoder/run_stage05.py

import torch

from stage05_encoder.encoder import EncoderLayer, Encoder


# ----------------------------
# Test 1 - Basic shape check
# ----------------------------
def test1():
    print("=== Test 1: EncoderLayer Shape Check ===")

    x = torch.randn(2, 5, 16)
    layer = EncoderLayer(d_model=16, num_heads=4, d_ff=64)

    out = layer(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Shape match:", out.shape == x.shape)

    print("Input mean/std:", x.mean().item(), x.std().item())
    print("Output mean/std:", out.mean().item(), out.std().item())


# ----------------------------
# Test 2 - Stacked encoder
# ----------------------------
def test2():
    print("\n=== Test 2: Encoder Stack Shape Check ===")

    x = torch.randn(2, 5, 16)
    encoder = Encoder(num_layers=3, d_model=16, num_heads=4, d_ff=64)

    out = encoder(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Shape match:", out.shape == x.shape)

    print("Output mean/std:", out.mean().item(), out.std().item())


# ----------------------------
# Test 3 - Mask does not crash
# ----------------------------
def test3():
    print("\n=== Test 3: Mask Pass Check ===")

    x = torch.randn(2, 5, 16)
    encoder = Encoder(num_layers=2, d_model=16, num_heads=4, d_ff=64)

    src_mask = torch.ones(2, 5)
    src_mask[0, 3:] = 0   # simulate padding
    src_mask[1, 4] = 0

    out = encoder(x, src_mask)

    print("Mask:\n", src_mask)
    print("Output shape:", out.shape)


# ----------------------------
# Test 4 - Mask effectiveness
# ----------------------------
def test4():
    print("\n=== Test 4: Mask Effectiveness ===")

    x = torch.randn(1, 5, 16)
    layer = EncoderLayer(d_model=16, num_heads=4, d_ff=64)

    src_mask = torch.ones(1, 5)
    src_mask[0, -1] = 0   # mask last token

    out_masked = layer(x, src_mask)
    out_unmasked = layer(x)

    print("Masked output (first token):", out_masked[0, 0, :5])
    print("Unmasked output (first token):", out_unmasked[0, 0, :5])

    diff = (out_masked - out_unmasked).abs().mean().item()

    assert diff > 0.0, "Mask had no effect"
    print("Mask is active, diff:", diff)

    print("Mean absolute difference:", diff)


# ----------------------------
# Test 5 - Verbose behavior
# ----------------------------
def test5():
    print("\n=== Test 5: Verbose Mode ===")

    x = torch.randn(2, 5, 16)

    print("\nRunning with verbose=True")
    layer_verbose = EncoderLayer(16, 4, 64, verbose=True)
    _ = layer_verbose(x)

    print("\nRunning with verbose=False")
    layer_silent = EncoderLayer(16, 4, 64, verbose=False)
    _ = layer_silent(x)

    print("If second run printed nothing → verbose works")


# ----------------------------
# Test 6 - Sensitivity check
# ----------------------------
def test6():
    print("\n=== Test 6: Sensitivity Check ===")

    encoder = Encoder(num_layers=2, d_model=16, num_heads=4, d_ff=64)

    x1 = torch.randn(1, 5, 16)
    x2 = x1.clone()
    x2[0, 0, 0] += 0.1

    out1 = encoder(x1)
    out2 = encoder(x2)

    diff = (out1 - out2).abs().mean().item()
    print("Mean difference after small input change:", diff)


# ----------------------------
# Running all the tests
# ----------------------------
if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()