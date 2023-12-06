class TermVocabMapper:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.unk_token = '[UNK]'
        self.add_special_tokens()

    def add_special_tokens(self):
        self.word_to_index[self.unk_token] = 0
        self.index_to_word[0] = self.unk_token

    def map_terms_to_vocab(self, tokens):
        index = len(self.word_to_index)

        for token in tokens:
            if token not in self.word_to_index:
                self.word_to_index[token] = index
                self.index_to_word[index] = token
                index += 1

    def convert_tokens_to_ids(self, tokens):
        return [self.word_to_index.get(token, 0) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.index_to_word.get(i, self.unk_token) for i in ids]
    
