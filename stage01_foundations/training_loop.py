# stage01_foundations/training_loop.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0,
    verbose: bool = False,
) -> float:
    
    """
    It runs a full pass of training data and loss is computed. 
    Returns average loss across all valid batch in the epoch.

    Bathces where every token is PAD is skipped entirely. The loss
    will be Nan there which can corrupt the gradients later.
    """

    model.train()
    
    total_loss = 0.0
    valid_batches = 0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        # In PyTorch, gradients accumulate at the start so we have 
        # to remove them.
        optimizer.zero_grad()

        # forward pass
        logits = model(src)

        # backward pass
        loss = loss_fn(logits, tgt)

        if torch.isnan(loss):
            if verbose:
                print(f"  [Train] batch {batch_idx} skipped - loss is nan (all PAD?)")
            continue

        # PyTorch backpropagates and fills .grad for every parameter.
        loss.backward()

        # clip before the optimizer step, not after.
        # this prevents one bad batch from sending the weights flying.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

        if verbose:
            print(f"  [Train] batch {batch_idx} | loss: {loss.item():.4f}")

    # if somehow every batch was skipped, return nan so the caller knows
    if valid_batches == 0:
        return float("nan")

    return total_loss / valid_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    verbose: bool = False,
) -> float:
    
    """
    Runs full pass over the validation data.
    Returns: average loss across all valid batches.
    """

    model.eval()

    total_loss = 0.0
    valid_batches = 0

    # no_grad tells PyTorch not to build the computation graph here.
    # saves memory and speeds things up since we're not backpropagating.
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)

            logits = model(src)
            loss = loss_fn(logits, tgt)

            if torch.isnan(loss):
                if verbose:
                    print(f"  [Eval] batch {batch_idx} skipped - loss is nan (all PAD?)")
                continue

            total_loss += loss.item()
            valid_batches += 1

            if verbose:
                print(f"  [Eval] batch {batch_idx} | loss: {loss.item():.4f}")

    if valid_batches == 0:
        return float("nan")

    return total_loss / valid_batches