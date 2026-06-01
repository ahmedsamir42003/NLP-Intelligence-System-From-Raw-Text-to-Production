import re
import string
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import logging

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)


class TextPreprocessor:
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_html: bool = True,
                 remove_accents: bool = True,
                 remove_special_chars: bool = False,
                 remove_stopwords: bool = True,
                 apply_stemming: bool = True,
                 apply_lemmatization: bool = True,
                 min_token_length: int = 2):

        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.remove_accents = remove_accents
        self.remove_special_chars = remove_special_chars
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.apply_lemmatization = apply_lemmatization
        self.min_token_length = min_token_length
        
        self.stemmer = PorterStemmer() if apply_stemming else None
        self.lemmatizer = WordNetLemmatizer() if apply_lemmatization else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def clean_text(self, text: str) -> str:

        if not isinstance(text, str):
            return ""
        
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        if self.remove_accents:
            text = self._remove_accents(text)
        
        if self.lowercase:
            text = text.lower()
        
        return text
    
    @staticmethod
    def _remove_accents(text: str) -> str:
   
        import unicodedata
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
    
    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)
    
    def process_tokens(self, tokens: List[str]) -> List[str]:
     
        processed = []
        
        for token in tokens:
   
            if self.remove_stopwords and token in self.stop_words:
                continue
            
  
            if len(token) < self.min_token_length:
                continue
            
            if token in string.punctuation:
                continue
            
            if self.apply_lemmatization:
                token = self.lemmatizer.lemmatize(token, pos='v')
                token = self.lemmatizer.lemmatize(token, pos='n')
            
            if self.apply_stemming:
                token = self.stemmer.stem(token)
            
            if token:  
                processed.append(token)
        
        return processed
    
    def preprocess(self, text: str) -> List[str]:


        cleaned = self.clean_text(text)
        
        tokens = self.tokenize(cleaned)
        
        processed = self.process_tokens(tokens)
        
        return processed
    
    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.preprocess(text) for text in texts]
    
    def preprocess_to_string(self, text: str) -> str:
        tokens = self.preprocess(text)
        return ' '.join(tokens)
    
    def get_stats(self, texts: List[str]) -> dict:
        all_tokens = []
        for text in texts:
            tokens = self.preprocess(text)
            all_tokens.extend(tokens)
        
        unique_tokens = set(all_tokens)
        
        return {
            'num_texts': len(texts),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(unique_tokens),
            'avg_tokens_per_text': len(all_tokens) / max(len(texts), 1),
            'vocab': list(unique_tokens)
        }


class DomainAdaptivePreprocessor(TextPreprocessor):
    
    def __init__(self, domain: str = 'general', **kwargs):

        self.domain = domain
        
        domain_configs = {
            'reviews': {
                'remove_urls': True,
                'remove_emails': True,
                'remove_stopwords': False,
                'apply_stemming': False,
                'apply_lemmatization': True,
                'min_token_length': 2
            },
            'social_media': {
                'remove_urls': False,  
                'remove_emails': True,
                'remove_stopwords': True,
                'apply_stemming': True,
                'apply_lemmatization': False,
                'min_token_length': 2
            },
            'news': {
                'remove_urls': True,
                'remove_emails': True,
                'remove_stopwords': True,
                'apply_stemming': False,
                'apply_lemmatization': True,
                'min_token_length': 3
            },
            'general': {
                'remove_urls': True,
                'remove_emails': True,
                'remove_stopwords': True,
                'apply_stemming': True,
                'apply_lemmatization': True,
                'min_token_length': 2
            }
        }
        
        domain_config = domain_configs.get(domain, domain_configs['general'])
        domain_config.update(kwargs)
        
        super().__init__(**domain_config)
        logger.info(f"Initialized DomainAdaptivePreprocessor for domain: {domain}")
