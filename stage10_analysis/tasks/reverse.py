# stage10_analysis/tasks/reverse.py

import torch

PAD = 0
START = 1

def generate_reverse_task(batch_size, seq_len, vocab_size, device="cpu"):
    """
    Reverse sequence task with proper token handling.
    """

    # Avoid PAD (0) and START (1)
    src = torch.randint(
        low=2,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device
    )

    reversed_seq = torch.flip(src, dims=[1])

    tgt_input = torch.zeros_like(src)
    tgt_input[:, 1:] = reversed_seq[:, :-1]
    tgt_input[:, 0] = START   # critical fix

    tgt_output = reversed_seq

    return src, tgt_input, tgt_output