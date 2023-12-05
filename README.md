# TimeWest Geodesic Coordinate Mesh for Multi-Domain Classification

## Overview
This repository contains an implementation of a custom Transformer model designed for text classification tasks. The model incorporates a custom architecture that includes multi-head self-attention mechanisms, feedforward layers, positional encoding, and padding masks.

## Model Architecture
The architecture consists of the following components:

### Multi-Head Attention
The `MultiHeadAttention` module splits the input embedding into multiple heads and computes attention scores, followed by combining and linear projection.

#### Mathematics:
The attention mechanism computes scaled dot-product attention as follows:
Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) * V


### Transformer Block
The `TransformerBlock` module consists of a single block containing multi-head attention, layer normalization, and feedforward layers.

#### Mathematics:
The forward pass of a Transformer block involves:
1. Computing the attention output: `AttentionOutput = MultiHeadAttention(Q, K, V)`
2. Applying layer normalization and adding a residual connection: `NormalizedOutput = LayerNorm(AttentionOutput + Input)`
3. Feedforward network with ReLU activation: `FFNOutput = ReLU(Linear(NormalizedOutput))`
4. Layer normalization and residual connection: `FinalOutput = LayerNorm(FFNOutput + NormalizedOutput)`

### Positional Encoding
The `CustomTransformer` module generates positional encodings to convey sequence order information to the model.

#### Mathematics:
The positional encoding is computed using sinusoidal functions:
PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))


### Padding Mask
The `create_padding_mask` function generates a padding mask to avoid attending to padding tokens in variable-length sequences.

## Usage
To utilize the model:
1. Instantiate the `CustomTransformer` class with desired parameters.
2. Pass inputs and padding masks through the model using the `forward` method.

## References
- The Transformer Paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- Hugging Face Transformers library: [GitHub](https://github.com/huggingface/transformers)
