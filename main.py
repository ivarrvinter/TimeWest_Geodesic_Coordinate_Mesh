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

    vocab_size = 100
    embed_size = 128
    num_layers = 6
    heads = 8
    max_seq_len = len(tokens_to_ids)

    input_ids_tensor = torch.tensor(tokens_to_ids)
    input_ids_tensor = input_ids_tensor.unsqueeze(0)

    model = EmbeddingAttentionTransformer(vocab_size, embed_size, num_layers, heads, max_seq_len)
    with torch.no_grad():
        output = model(input_ids_tensor, max_seq_len, vocab_size)

    print("Output shape:", output.shape)
    print("Model output:", output)

if __name__ == '__main__':
    main()
