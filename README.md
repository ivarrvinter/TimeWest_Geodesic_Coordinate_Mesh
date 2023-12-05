# TimeWest Geodesic Coordinate Mesh

## Overview
This repository contains an implementation of a custom transformer model designed for multi-domain text classification tasks. The model incorporates multi-head self-attention mechanisms, feedforward layers, positional encoding, and padding masks.

## Model Architecture
The architecture consists of the following components:

### Multi-Head Attention
The `SelfAttentionLayer` module splits the input embedding into multiple heads and computes attention scores, followed by combining and linear projection. The attention mechanism computes scaled dot-product attention as follows:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \times V
$$

### Transformer Block
The `SelfAttentionModule` module consists of a single block containing multi-head attention, layer normalization, and feedforward layers. The forward pass of a Transformer block involves:
1. Computing the attention output: $`\text{AttentionOutput} = \text{SelfAttentionLayer}(Q, K, V)`$
2. Applying layer normalization and adding a residual connection: $`\text{NormalizedOutput} = \text{LayerNorm}(\text{AttentionOutput} + \text{Input})`$
3. Feedforward network with ReLU activation: $`\text{FFNOutput} = \text{ReLU}(\text{Linear}(\text{NormalizedOutput}))`$
4. Layer normalization and residual connection: $`\text{FinalOutput} = \text{LayerNorm}(\text{FFNOutput} + \text{NormalizedOutput})`$

### Positional Encoding
The `EmbeddingAttentionTransformer` module generates positional encodings to convey sequence order information to the model. The positional encoding is computed using sinusoidal functions:
$`\text{PE}_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{\left(\frac{2i}{d_{\text{model}}}\right)}}\right)`$
$`\text{PE}_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{\left(\frac{2i}{d_{\text{model}}}\right)}}\right)`$


### Padding Mask
The `create_padding_mask` function generates a padding mask to avoid attending to padding tokens in variable-length sequences.

## References
- The Transformer Paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- Hugging Face Transformers library: [GitHub](https://github.com/huggingface/transformers)
