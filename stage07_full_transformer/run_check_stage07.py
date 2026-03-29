# stage08_full_transformer/run_check_stage07.py

import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from stage07_full_transformer.transformer import Transformer


model = Transformer(
    src_vocab_size=100,
    tgt_vocab_size=100,
    d_model=32,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=64,
    max_seq_len=50,
    dropout_p=0.0,
    pad_idx=0,
)
model.eval()


src = torch.randint(1, 100, (2, 10))
tgt = torch.randint(1, 100, (2, 7))


# ---- Check 1: shape ----
logits = model(src, tgt)
assert logits.shape == (2, 7, 100)
print("Check 1 passed")


# ---- Check 2: padding mask correctness ----
src_with_pad = torch.randint(1, 100, (2, 10))
src_with_pad[0, 7:] = 0

fixed_mask = (src_with_pad != 0).float()

out1 = model(src_with_pad, tgt, src_mask=fixed_mask)

src_modified = src_with_pad.clone()
src_modified[0, 7:] = 99

out2 = model(src_modified, tgt, src_mask=fixed_mask)

diff = (out1[0] - out2[0]).abs().max().item()
assert diff < 1e-5, f"Padding mask failed: diff={diff}"
print("Check 2 passed")


# ---- Check 3: causality ----
tgt_test = torch.randint(1, 100, (1, 6))

out_original = model(src[:1], tgt_test)

tgt_modified = tgt_test.clone()
tgt_modified[0, 4] += 1

out_modified = model(src[:1], tgt_modified)

diff_before = (out_original[0, :4] - out_modified[0, :4]).abs().max().item()
diff_after = (out_original[0, 4:] - out_modified[0, 4:]).abs().max().item()

assert diff_before < 1e-5, f"Causality broken: diff={diff_before}"
assert diff_after > 0, "Change had no effect"

print("Check 3 passed")


# ---- Check 4: gradients ----
model.zero_grad()

logits = model(src, tgt)
loss = logits.sum()
loss.backward()

grad = model.output_projection.weight.grad

assert grad is not None
assert grad.abs().sum().item() > 0

print("Check 4 passed")


print("\nAll checks passed. Stage 7 complete.")