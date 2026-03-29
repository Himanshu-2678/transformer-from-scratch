# Transformer From Scratch

This repository is a deep dive into the Transformer architecture introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017)** using PyTorch.

## Transformer Architecture

![Transformer Architecture](assets/transformer_architecture.png)

*Figure: Encoder–decoder Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017).*

The goal of this project is to implement the complete encoder–decoder Transformer architecture from 
first principles using PyTorch, building each component step-by-step to develop a deep understanding 
of attention-based models.

Instead of using high-level libraries like HuggingFace, every component is implemented
step-by-step, including:

- tokenization and embeddings  
- scaled dot-product attention  
- multi-head attention  
- positional encoding  
- encoder blocks  
- decoder blocks with causal masking  
- encoder–decoder cross attention  
- full Transformer architecture  
- sequence-to-sequence training  

The repository is organized into stages so that each part of the Transformer can be
implemented, tested, and understood independently.

## Current Status

- Stage 01 – Foundations ✔  
- Stage 02 – Attention ✔  
- Stage 03 – Multi-head Attention ✔  
- Stage 04 – Positional Encoding ✔  
- Stage 05 – Encoder ✔  
- Stage 06 – Decoder ✔  
- Stage 07 – Full Transformer ✔  

The full Transformer architecture has been implemented and validated with:

- correct causal masking in the decoder  
- correct padding masking in encoder and cross-attention  
- verified encoder–decoder interaction  
- gradient flow through the entire model  

Training pipeline (Stage 08) will be implemented next.

By the end of this project, the full Transformer model will be trained on a
sequence-to-sequence task, providing hands-on understanding of how modern
attention-based architectures work internally, and serving as a foundation
for building small language models (SLMs).

This project is intended for educational purposes and for developing deep intuition
about Transformer models and attention mechanisms.

## Reference

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,  
Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.  

**Attention Is All You Need** (2017) - https://arxiv.org/abs/1706.03762

## License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this code with proper attribution.

See the [LICENSE](LICENSE) file for details.
