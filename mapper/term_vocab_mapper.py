class TermVocabMapper():
    @staticmethod
    def map_terms_to_vocab(text):
        word_to_index = {}
        index_to_word = {}
        
        index = 0

        for word in text:
            if word not in word_to_index:
                word_to_index[word] = index
                index_to_word[index] = word
                index += 1
        
        return word_to_index, index_to_word
