import torch.nn as nn
from embedding_utils.self_attention_layer import SelfAttentionLayer

class SelfAttentionModule(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttentionModule, self).__init__()
        self.attention = SelfAttentionLayer(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out
