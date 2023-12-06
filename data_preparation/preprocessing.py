import re
from nltk.tokenize import word_tokenize

class Preprocessor:
    def __init__(self, vocab_mapper):
        self.vocab_mapper = vocab_mapper
        self.special_tokens = vocab_mapper.special_tokens

    def apply_substitution(self, text: str, rule):
        return re.sub(rule, '', text) if text else None
    
    def convert_to_lower(self, text: str):
        return text.lower() if text else None
    
    def tokenize_text(self, text: str):
        tokens = word_tokenize(text) if text else []
        tokens = [self.special_tokens.get(token, token) for token in tokens]
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return self.vocab_mapper.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids):
        return self.vocab_mapper.convert_ids_to_tokens(ids)
    
