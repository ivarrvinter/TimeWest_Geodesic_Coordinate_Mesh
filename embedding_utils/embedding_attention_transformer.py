import torch.nn as nn
from embedding_utils.self_attention_module import SelfAttentionModule

class EmbeddingAttentionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads):
        super(EmbeddingAttentionTransformer, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList(
            [
                SelfAttentionModule(embed_size, heads) for _ in range(num_layers)
            ]
        )
        
    def forward(self, x, mask):
        out = self.embedding(x)
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out
