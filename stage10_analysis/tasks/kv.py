# stage10_analysis/tasks/kv.py

import torch

PAD = 0
START = 1

def generate_kv_task(batch_size, num_pairs=3, vocab_size=20, device="cpu"):
    """
    Dense KV task.

    Input:  [K1, V1, K2, V2, ...]
    Target: [V1, V1, V2, V2, ...]
    """

    keys = torch.stack([
        torch.randperm(vocab_size - 2)[:num_pairs] + 2
        for _ in range(batch_size)
    ]).to(device)

    values = torch.randint(2, vocab_size, (batch_size, num_pairs), device=device)

    # interleave
    src = torch.stack((keys, values), dim=2).reshape(batch_size, -1)

    # target
    tgt = torch.zeros_like(src)

    # fill targets
    for i in range(num_pairs):
        tgt[:, 2*i] = values[:, i]      # key → value
        tgt[:, 2*i+1] = values[:, i]    # value → value

    # decoder input (shifted)
    tgt_input = torch.zeros_like(src)
    tgt_input[:, 1:] = tgt[:, :-1]
    tgt_input[:, 0] = START

    tgt_output = tgt

    return src, tgt_input, tgt_output