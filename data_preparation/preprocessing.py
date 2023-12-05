import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class Preprocessor:
    def __init__(self, text: str = None):
        self.text = text
        self.tokens = None

    def apply_substitution(self, rule):
        return re.sub(rule, '', self.text) if self.text else None
    
    def convert_to_lower(self):
        return self.text.lower() if self.text else None
    
    def tokenize_text(self):
        return word_tokenize(self.text) if self.text else []
    
    def lemmatize_text(self, tokens):
        tokens = self.tokens or self.tokenize_text()
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
