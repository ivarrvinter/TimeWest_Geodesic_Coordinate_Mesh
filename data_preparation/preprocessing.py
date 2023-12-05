import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class Preprocessor:
    @staticmethod
    def apply_substitution(text: str, rule):
        return re.sub(rule, '', text) if text else None
    
    @staticmethod
    def convert_to_lower(text: str):
        return text.lower() if text else None
    
    @staticmethod
    def tokenize_text(text: str):
        return word_tokenize(text) if text else []
    
    @staticmethod
    def lemmatize_text(tokens):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
