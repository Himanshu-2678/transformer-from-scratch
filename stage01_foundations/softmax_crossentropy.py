# stage01_foundations/softmax_crossentropy.py

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    """
    This module measures how far our model's predictions are from the correct answers.

    The model produces raw scores (logits) for every token position, and we compare
    those against the true token IDs. The result is a single scalar value: the loss.
    A higher value means the model is making worse predictions.

    One important detail: we always pass raw logits here. We do not apply softmax
    ourselves. PyTorch internally combines log_softmax with negative log likelihood
    in a numerically stable way. Applying softmax beforehand would distort the result.

    Padding tokens are excluded from the loss using ignore_index. The model still
    produces outputs for those positions, but they are not considered when computing
    the final average loss.

    If an entire batch consists only of PAD tokens, the result may become NaN.
    That situation is better handled in the training loop rather than inside this module.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        label_smoothing: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing
        self.verbose = verbose

        # CrossEntropyLoss internally applies log_softmax + NLLLoss.
        # label_smoothing allows us to soften the target distribution slightly,
        # which helps prevent the model from becoming too confident.
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, T, V] raw scores from the model
            targets: [B, T]    correct token IDs

        Returns:
            Scalar loss (mean over non-PAD tokens)
        """

        # Basic shape and type checks to catch issues early
        assert logits.dim() == 3, \
            f"logits expected [B, T, V], got {logits.shape}"
        assert targets.dim() == 2, \
            f"targets expected [B, T], got {targets.shape}"
        assert logits.shape[:2] == targets.shape, \
            f"Mismatch between logits {logits.shape} and targets {targets.shape}"
        assert targets.dtype == torch.long, \
            f"targets must be torch.long, got {targets.dtype}"

        if self.verbose:
            print(f"[Loss] logits  : {logits.shape}  dtype={logits.dtype}")
            print(f"[Loss] targets : {targets.shape}  dtype={targets.dtype}")
            print(f"[Loss] pad_id={self.pad_id}, label_smoothing={self.label_smoothing}")

        # Flatten [B, T, V] → [B*T, V] and [B, T] → [B*T]
        # Each token position is treated as an independent training example
        B, T, V = logits.shape
        logits  = logits.reshape(B * T, V)
        targets = targets.reshape(B * T)

        if self.verbose:
            print(f"[Loss] flattened logits: {logits.shape}, targets: {targets.shape}")

        # Compute loss (log_softmax + NLLLoss internally)
        loss = self.criterion(logits, targets)

        if self.verbose:
            print(f"[Loss] value: {loss.item():.4f}")

        return loss