# stage04_positional_encoding/positional_encoding.py

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional information to embeddings.
    Input and output shape: [B, T, d_model]
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 512,
        dropout_p: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__()

        assert d_model % 2 == 0, "d_model must be even"

        self.verbose = verbose
        self.dropout = nn.Dropout(p=dropout_p)

        # build positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # small i -> high frequency, large i -> low frequency
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        assert x.shape[2] == self.pe.shape[2]

        T = x.shape[1]
        assert T <= self.pe.shape[1]

        if self.verbose:
            print(f"[PE] Input: {x.shape}")
            print(f"[PE] PE slice: {self.pe[:, :T, :].shape}")

        x = x + self.pe[:, :T, :]

        if self.verbose:
            print(f"[PE] Output: {x.shape}")

        return self.dropout(x)


def demonstrate_positional_encoding():
    print("=" * 60)
    print("Positional Encoding verification")
    print("=" * 60)

    d_model = 16
    max_seq_len = 50

    pe_module = PositionalEncoding(
        d_model=d_model,
        max_seq_len=max_seq_len,
        verbose=True
    )

    # Case 1: shape check
    print("\nCase 1: Shape check")
    B, T = 2, 10
    x = torch.randn(B, T, d_model)
    out = pe_module(x)
    assert out.shape == (B, T, d_model)
    print("Shape correct")

    # Case 2: PE is non-zero
    print("\nCase 2: Non-zero check")
    assert pe_module.pe.abs().sum().item() > 0
    print("PE contains values")

    # Case 3: positions differ
    print("\nCase 3: Position uniqueness")
    pe = pe_module.pe[0]
    assert not torch.allclose(pe[0], pe[1])
    print("Positions are different")

    # Case 4: sequence too long
    print("\nCase 4: Sequence limit")
    try:
        pe_module(torch.randn(1, max_seq_len + 1, d_model))
        print("ERROR - should have failed")
    except AssertionError:
        print("Correctly rejected long sequence")

    # Case 5: PE values are always bounded between -1 and 1 (guaranteed by sin/cos)
    print("\nCase 5: Bounds check")
    max_val = pe_module.pe.abs().max().item()
    print(f"Max absolute PE value: {max_val:.6f}")
    assert max_val <= 1.0
    print("PE values correctly bounded in [-1, 1]")

    print("\n" + "=" * 60)
    print("All checks passed")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_positional_encoding()