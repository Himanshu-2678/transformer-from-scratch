# Transformer Analysis Findings

## 1. Experimental Setup

We analyze a small-scale encoder–decoder Transformer to study attention head behavior under varying task complexity.

- Architecture: Encoder–Decoder Transformer (Post LayerNorm)  
- d_model: 32  
- Heads: 4  
- Layers: 2 encoder, 2 decoder  
- Training: Adam (lr = 1e-3), no scheduler, no dropout  

Tasks evaluated:

- Copy task (baseline, positional identity)  
- Reverse task (positional transformation, seq_len = 16)  
- Key–Value (KV) task (content-based retrieval)  



## 2. Validation

Before analysis, correctness of the implementation was verified:

- Causal and padding masks are correctly applied  
- No future token leakage is observed  
- Attention weights are properly normalized  
- No numerical instability (NaNs) during training  
- Attention weights are extracted per layer and per head  

All tasks converge successfully, with near-zero training loss and correct outputs.



## 3. Methodology

We evaluate attention behavior using the following metrics:

- **Entropy**: Measures sharpness versus diffuseness of attention  
- **Head Similarity**: Measures redundancy across heads  
- **Positional Analysis**: Measures alignment with diagonal attention patterns (used for copy and reverse)  
- **Ablation Impact**: Measures causal importance of individual heads  

All metrics are computed over aggregated batches after training.



## 4. Results

### 4.1 Copy Task

The copy task serves as a baseline with minimal complexity.

Attention exhibits strong diagonal structure, where each position attends primarily to itself. Entropy is low in early layers, indicating sharp and focused attention, while deeper layers show slightly higher entropy due to residual mixing.

Head similarity is high, especially in deeper layers, indicating that multiple heads converge to nearly identical behavior. Ablation experiments show minimal variation in loss across heads, confirming that no individual head plays a distinct role.

Overall, the model learns a symmetric identity mapping with highly redundant attention heads.



### 4.2 Reverse Task

The reverse task introduces non-trivial positional dependency, requiring mapping from position *i* to position *T − i*.

Despite this increased complexity, attention patterns remain largely diagonal. No anti-diagonal or reverse-aligned structure is observed.

Entropy shows slightly higher variance in early layers, suggesting minor differentiation across heads. However, head similarity remains high and, in some cases, exceeds that of the copy task, indicating even stronger redundancy.

Ablation results do not reveal any head-specific importance. All heads contribute similarly, and no functional specialization is observed.



### 4.3 Key–Value (KV) Task

The KV task introduces content-based dependency, requiring the model to retrieve values associated with keys.

Unlike positional tasks, this setup forces attention to act as a retrieval mechanism.

In this setting, a partial shift in behavior is observed:

- Early layer heads exhibit increased entropy variance, indicating differentiated attention patterns  
- Head similarity in early layers decreases compared to positional tasks, with no strongly redundant head pairs  
- Some heads become highly focused, while others remain more diffuse  

However, deeper layers remain highly redundant, with head similarity remaining close to previous tasks.

This indicates that specialization begins to emerge, but is limited to early layers.



## 5. Key Findings

1. Increasing positional complexity (copy to reverse) does not induce attention head specialization  
2. The model solves reverse mapping without developing structured attention patterns  
3. Content-based tasks (KV) introduce measurable differentiation across heads  
4. Specialization emerges partially and is concentrated in early layers  
5. Deeper layers remain highly redundant across all tasks  



## 6. Interpretation

These results indicate that attention is not the primary mechanism used for solving positional transformations in this setting.

Instead, the model likely relies on:

- positional encodings  
- feedforward transformations  
- residual connections  

However, when the task requires content-based retrieval, attention becomes functionally relevant, leading to partial specialization.

The emergence of specialization is therefore task-dependent and layer-dependent, rather than an inherent property of the architecture.



## 7. Conclusion

Attention head specialization is not an inherent property of Transformers.

It does not emerge from positional complexity alone. Instead, specialization begins to appear when the task requires content-based reasoning, but remains limited in scope and does not uniformly affect all layers.

These findings suggest that meaningful head specialization requires both sufficient task complexity and appropriate architectural capacity.



## 8. Next Steps

- Perform functional classification of heads (identity, shift, global, content-based)  
- Analyze attention patterns in KV task at head level  
- Evaluate causal importance using structured ablation  
- Study scaling effects on specialization (number of heads, layers)