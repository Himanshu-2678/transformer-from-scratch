# stage10_analysis/synthetic_tasks.py

import torch

def generate_copy_task(batch_size, seq_len, vocab_size):
    """
    Proper copy task for seq2seq Transformer.

    Token convention:
    0 = PAD
    1 = SOS
    2 = EOS
    3... = actual tokens
    """

    # generate random tokens
    src = torch.randint(3, vocab_size, (batch_size, seq_len))

    # force EOS at end
    src[:, -1] = 2

    # target = [SOS] + src
    sos = torch.full((batch_size, 1), 1)
    tgt = torch.cat([sos, src], dim=1)

    return src, tgt


def generate_reverse_task(batch_size, seq_len, vocab_size):
    src = torch.randint(3, vocab_size, (batch_size, seq_len))
    src[:, -1] = 2

    reversed_seq = torch.flip(src, dims=[1])

    sos = torch.full((batch_size, 1), 1)
    tgt = torch.cat([sos, reversed_seq], dim=1)

    return src, tgt


def generate_fixed_dependency_task(batch_size, seq_len, vocab_size, shift=3):

    src = torch.randint(3, vocab_size, (batch_size, seq_len))
    src[:, -1] = 2

    y = src.clone()

    for i in range(shift, seq_len):
        y[:, i] = src[:, i - shift]

    sos = torch.full((batch_size, 1), 1)
    tgt = torch.cat([sos, y], dim=1)

    return src, tgt


def evaluate_task(model, src, tgt, src_mask, tgt_mask, loss_fn):
    """
    Evaluate model performance on synthetic task.
    """

    model.eval()

    with torch.no_grad():
        output = model(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )

        loss = loss_fn(
            output.reshape(-1, output.size(-1)),
            tgt.reshape(-1)
        )

    return loss.item()