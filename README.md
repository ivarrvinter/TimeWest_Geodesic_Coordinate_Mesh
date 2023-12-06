# TimeWest Geodesic Coordinate Mesh

## Overview
This repository contains an implementation of a custom transformer model designed for classification tasks. The model incorporates multi-head self-attention mechanisms, feedforward layers, positional encoding, and padding masks.

## Model Architecture
The architecture consists of the following components:

### 1. **Self-Attention Mechanism:**

The core of the Transformer architecture lies in the self-attention mechanism, defined by the `SelfAttentionLayer`. It computes the attention $\( \text{A} \)$ by projecting queries $\( Q \)$, keys $\( K \)$, and values $\( V \)$ linearly:
```math

\text{Energy} = Q \times K^T \quad \text{(Einstein Summation)}
```
```math
\text{Attention} = \text{softmax}\left(\frac{\text{Energy}}{\sqrt{\text{Embedding Size}}}\right) \\
```
```math
\text{Output} = \text{Attention} \times V

```
The attention calculation involves tensor contractions, masked to handle irrelevant positions. This operation has a complexity of $\( O(N^2 \times \text{Seq Len}) \)$, making it crucial for parallelization strategies in large-scale applications.

### 2. **Self-Attention Module:**

The `SelfAttentionModule` integrates the self-attention mechanism within a multi-layer structure, employing layer normalization $\( LN \)$ and a feed-forward network $\( FFN \)$:
```math
\text{Normalized Attention} = LN(\text{Output of Attention} + \text{Query})
```
```math
\text{Processed} = FFN(\text{Normalized Attention}) + \text{Normalized Attention}
```
This involves linear transformations, ReLU activation, and layer-wise addition, contributing to the non-linearity and enhanced expressiveness of the model.

### 3. **Embedding Attention Transformer:**

The `EmbeddingAttentionTransformer` stage-manage the overall architecture, amalgamating token and positional embeddings $\( E \)$ and $\( P \)$ alongside stacked self-attention modules:
```math
\text{Embedding} = E(\text{Token IDs}) + P(\text{Positions})
```
```math
\text{Augmented Embedding} = \text{Stacked Self Attention}(\text{Embedding})
```

The actual module generates positional encodings to convey sequence order information to the model. The positional encoding is computed using sinusoidal functions:
```math
\text{PE}_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{\left(\frac{2i}{d_{\text{model}}}\right)}}\right)
```
```math
\text{PE}_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{\left(\frac{2i}{d_{\text{model}}}\right)}}\right)
```
This cascaded transformation encapsulates the core of the Transformer architecture, where embeddings undergo self-attention operations across multiple layers, enabling sophisticated relationships within sequences. The computational complexity is $\( O(L \times N^2) \)$, where $\( L \)$ represents the number of layers.

## References
- The Transformer Paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- Pytorch library: [Pytorch](https://github.com/pytorch/pytorch)
