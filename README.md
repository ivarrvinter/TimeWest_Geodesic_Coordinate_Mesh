# TimeWest Geodesic Coordinate Mesh

## Overview
This repository contains an implementation of a custom Transformer model designed for multi-domain text classification tasks. The model incorporates a custom architecture that includes multi-head self-attention mechanisms, feedforward layers, positional encoding, and padding masks.

## Model Architecture
The architecture consists of the following components:

### Multi-Head Attention
The `MultiHeadAttention` module splits the input embedding into multiple heads and computes attention scores, followed by combining and linear projection.

#### Mathematics:
The attention mechanism computes scaled dot-product attention as follows:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \times V
$$

### Transformer Block
The `TransformerBlock` module consists of a single block containing multi-head attention, layer normalization, and feedforward layers.

#### Mathematics:
The forward pass of a Transformer block involves:
1. Computing the attention output: \( \text{AttentionOutput} = \text{MultiHeadAttention}(Q, K, V) \)
2. Applying layer normalization and adding a residual connection: \( \text{NormalizedOutput} = \text{LayerNorm}(\text{AttentionOutput} + \text{Input}) \)
3. Feedforward network with ReLU activation: \( \text{FFNOutput} = \text{ReLU}(\text{Linear}(\text{NormalizedOutput})) \)
4. Layer normalization and residual connection: \( \text{FinalOutput} = \text{LayerNorm}(\text{FFNOutput} + \text{NormalizedOutput}) \)

### Positional Encoding
The `CustomTransformer` module generates positional encodings to convey sequence order information to the model.

#### Mathematics:
The positional encoding is computed using sinusoidal functions:

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{(2i/d_{\text{model}})}}\right)
$$
$$
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{(2i/d_{\text{model}})}}\right)
$$

### Padding Mask
The `create_padding_mask` function generates a padding mask to avoid attending to padding tokens in variable-length sequences.

## Usage
To utilize the model:
1. Instantiate the `CustomTransformer` class with desired parameters.
2. Pass inputs and padding masks through the model using the `forward` method.

## References
- The Transformer Paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- Hugging Face Transformers library: [GitHub](https://github.com/huggingface/transformers)
