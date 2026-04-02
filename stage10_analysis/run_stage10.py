# stage10_analysis/run_stage10.py

import torch
import torch.nn as nn

from stage10_analysis.run_analysis import run_attention_analysis
from stage08_training.data_loader import get_dataloader
from models.transformer import Transformer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model (MATCH TRAINING CONFIG) ----
    model = Transformer(
        src_vocab_size=20,
        tgt_vocab_size=20,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        max_seq_len=50,
        dropout_p=0.0,
        pad_idx=0,
    ).to(device)

    # ---- Load checkpoint ----
    model.load_state_dict(torch.load("checkpoint.pt", map_location=device))

    # ---- Data ----
    dataloader = get_dataloader(device)

    # ---- Loss ----
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # ---- Run analysis ----
    results = run_attention_analysis(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
        max_batches=10
    )

    # ---- Output ----
    print(results.keys())
    print(results["ablation"])

if __name__ == "__main__":
    main()