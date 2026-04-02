# stage08_training/data_loader.py

import torch
from stage10_analysis.synthetic_tasks import generate_copy_task

PAD = 0


def get_dataloader(device, num_batches=20):
    """
    Generates synthetic COPY task batches for analysis.
    """

    data = []

    for _ in range(num_batches):

        # ---- Generate synthetic data ----
        src, tgt = generate_copy_task(
            batch_size=16,
            seq_len=10,
            vocab_size=20
        )

        src = src.to(device)
        tgt = tgt.to(device)

        # ---- Shift targets ----
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # ---- Masks ----
        src_mask = (src != PAD)

        T_len = tgt_input.size(1)

        tgt_pad_mask = (tgt_input != PAD).unsqueeze(1)

        causal_mask = torch.tril(
            torch.ones(T_len, T_len, device=device)
        ).bool().unsqueeze(0)

        tgt_mask = tgt_pad_mask & causal_mask

        # ---- Append ----
        data.append((src, tgt_input, src_mask, tgt_mask, tgt_output))

    return data