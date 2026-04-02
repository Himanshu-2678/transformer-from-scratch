# Transformer Analysis Findings

## Setup
- Architecture: Encoder-Decoder Transformer (Post-LN)
- d_model: 32
- Heads: 4
- Layers: 2 encoder, 2 decoder
- Task: Copy Task (synthetic)
- Training: Adam (lr=1e-3), no scheduler, no dropout
- Data: SOS/EOS aligned sequence generation

## Validation Status
- Masking: Correct (causal + padding)
- Attention computation: Verified (mask applied pre-softmax)
- Causality: No future leakage
- Numerical stability: No NaNs, fully masked rows handled
- Attention exposure: Layer-wise attention extracted

## Training Behavior
- Loss decreased from ~2.4 to ~0.02
- Model successfully learned copy task

## Attention Analysis

### Entropy
- Attention entropy reduced during training
- Indicates non-uniform, focused attention

### Head Similarity
- High similarity across heads within each layer
- No diversity observed

### Positional Behavior
- Strong diagonal attention patterns
- Model attends primarily to same-position tokens

### Ablation Results
- All heads produced identical delta_loss (~0.18)
- No head had higher or lower importance

## Interpretation
- Model learned symmetric attention behavior
- All heads converge to same function (diagonal copy)
- No head specialization observed
- Multi-head attention behaves as replicated single-head

## Conclusion
- Attention mechanism is functioning correctly
- Task complexity is insufficient to induce specialization
- Current setup validates correctness, not diversity

## Next Steps
- Train on reverse task (non-trivial positional dependency)
- Analyze head specialization under harder constraints
- Compare entropy, similarity, and ablation across tasks