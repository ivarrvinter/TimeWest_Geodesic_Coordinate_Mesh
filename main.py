import torch
from data_preparation.preprocessing import Preprocessor
from mapper.term_vocab_mapper import TermVocabMapper
from embedding_utils.embedding_attention_transformer import EmbeddingAttentionTransformer

def main():
    seed = 42
    torch.manual_seed(seed)

    text = '''A neural network is a neural circuit of biological neurons, sometimes also called a biological neural network,
            or a network of artificial neurons or nodes in the case of an artificial neural network.[1]'''

    vocab_mapper = TermVocabMapper()
    preprocess = Preprocessor(vocab_mapper)
    text = preprocess.apply_substitution(text, r'[\d\.\,\[\]]')
    text = preprocess.convert_to_lower(text)
    tokens = preprocess.tokenize_text(text)
    lemmas = preprocess.lemmatize_text(tokens)

    vocab_mapper.map_terms_to_vocab(lemmas)
    tokens_to_ids = preprocess.convert_tokens_to_ids(lemmas)
    ids_to_tokens = preprocess.convert_ids_to_tokens(tokens_to_ids)
    # Revise
    vocab_size = len(word_to_index)
    embed_size = 512
    num_layers = 4
    heads = 8

    indices = [word_to_index[token] for token in tokens]
    values = torch.tensor(indices).unsqueeze(0)
    keys = torch.tensor(indices).unsqueeze(0)
    query = torch.tensor(indices).unsqueeze(0)

    seq_length = len(tokens)
    mask = torch.triu(torch.ones(seq_length, seq_length)) == 1
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    model = EmbeddingAttentionTransformer(vocab_size, embed_size, num_layers, heads)
    output = model(values, mask)
    print('Output embedding is: ', output)

if __name__ == '__main__':
    main()
