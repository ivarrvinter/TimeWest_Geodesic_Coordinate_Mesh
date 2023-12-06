import torch
import torch.nn as nn
from embedding_utils.self_attention_module import SelfAttentionModule
from utils import truncate_positions

class EmbeddingAttentionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, max_seq_len):
        super(EmbeddingAttentionTransformer, self).__init__()
        self.max_seq_len = max_seq_len
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_size)
        self.layers = nn.ModuleList(
            [
                SelfAttentionModule(embed_size, heads) for _ in range(num_layers)
            ]
        )

        self.special_tokens = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2
        }
        
    def truncate_positions(positions, max_seq_len):
        if positions.size(0) > max_seq_len:
            positions = positions[:max_seq_len]
        return positions
    
    def forward(self, tokens_to_ids, max_seq_len, vocab_size):
        input_ids_tensor = tokens_to_ids.clone().detach()
        seq_length = input_ids_tensor.size(1)
        positions = torch.arange(seq_length, device=input_ids_tensor.device)
        positions = truncate_positions(positions, max_seq_len)

        assert positions.max() < max_seq_len, "Position values exceed max_seq_len"
        assert input_ids_tensor.max() < vocab_size, "Token IDs exceed vocabulary size"
    
        pos_embed = self.pos_embedding(positions)
        x = self.embedding(input_ids_tensor)
        x = x + pos_embed
        
        for layer in self.layers:
            x = layer(x, x, x, None)
        
        return x
