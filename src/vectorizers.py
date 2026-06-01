from typing import List, Dict, Union, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from rank_bm25 import BM25Okapi
import gensim.models as gensim_models
from gensim.utils import simple_preprocess
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class BoWVectorizer:
    
    def __init__(self, max_features: int = 50000, ngram_range: Tuple[int, int] = (1, 2)):

        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=False 
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"BoW vectorizer fitted with {len(self.vectorizer.get_feature_names_out())} features")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True


class TFIDFVectorizer:
    
    def __init__(self, max_features: int = 50000, ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 5, max_df: float = 0.8):
 
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=False 
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"TF-IDF vectorizer fitted with {len(self.vectorizer.get_feature_names_out())} features")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True


class BM25Vectorizer:
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):

        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus = None
        self.tokenized_corpus = None
    
    def fit(self, texts: List[str]):

        self.corpus = texts
        self.tokenized_corpus = [text.split() for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        logger.info(f"BM25 fitted with {len(self.corpus)} documents")
    
    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[int, float]]:
 
        if self.bm25 is None:
            raise ValueError("BM25 not fitted. Call fit() first.")
        
        query_tokens = query_text.split()
        scores = self.bm25.get_scores(query_tokens)
        
   
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        
        return results
    
    def batch_query(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[int, float]]]:
        return [self.query(q, top_k) for q in queries]
    
    def get_corpus(self) -> List[str]:
        return self.corpus
    
    def save(self, path: str):
        import pickle
        data = {
            'corpus': self.corpus,
            'tokenized_corpus': self.tokenized_corpus,
            'bm25': self.bm25,
            'k1': self.k1,
            'b': self.b
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.corpus = data['corpus']
        self.tokenized_corpus = data['tokenized_corpus']
        self.bm25 = data['bm25']
        self.k1 = data['k1']
        self.b = data['b']


class Word2VecEmbedding:
    
    def __init__(self, vector_size: int = 300, window: int = 5, min_count: int = 5,
                 workers: int = 4, sg: int = 1, epochs: int = 5):
 
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.model = None
    
    def fit(self, sentences: List[List[str]]):
 
        self.model = gensim_models.Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs
        )
        logger.info(f"Word2Vec model trained with vocab size: {len(self.model.wv)}")
    
    def get_vector(self, word: str) -> np.ndarray:
 
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        if word in self.model.wv:
            return self.model.wv[word]
        return np.zeros(self.vector_size)
    
    def get_sentence_vector(self, tokens: List[str]) -> np.ndarray:
 
        vectors = [self.get_vector(token) for token in tokens]
        valid_vectors = [v for v in vectors if np.any(v)]
        
        if not valid_vectors:
            return np.zeros(self.vector_size)
        
        return np.mean(valid_vectors, axis=0)
    
    def get_most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
    
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        if word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []
    
    def save(self, path: str):
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path: str):
        self.model = gensim_models.Word2Vec.load(path)
        self.vector_size = self.model.vector_size


class GloVeEmbedding:
    
    def __init__(self, model_path: str = None, vector_size: int = 300):

        self.vector_size = vector_size
        self.model_path = model_path
        self.embeddings = {}
        
        if model_path:
            self.load(model_path)
    
    def load(self, path: str):
  
        logger.info(f"Loading GloVe embeddings from {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                vector = np.array([float(v) for v in parts[1:]], dtype=np.float32)
                self.embeddings[word] = vector
        logger.info(f"Loaded {len(self.embeddings)} word embeddings")
    
    def get_vector(self, word: str) -> np.ndarray:
        if word in self.embeddings:
            return self.embeddings[word]
        return np.zeros(self.vector_size)
    
    def get_sentence_vector(self, tokens: List[str]) -> np.ndarray:
        vectors = [self.get_vector(token) for token in tokens]
        valid_vectors = [v for v in vectors if np.any(v)]
        
        if not valid_vectors:
            return np.zeros(self.vector_size)
        
        return np.mean(valid_vectors, axis=0)


class EmbeddingVectorizer:

    
    def __init__(self, embedding_model: Union[Word2VecEmbedding, GloVeEmbedding]):

        self.embedding_model = embedding_model
    
    def transform(self, texts: List[List[str]]) -> np.ndarray:

        vectors = []
        for tokens in texts:
            vector = self.embedding_model.get_sentence_vector(tokens)
            vectors.append(vector)
        
        return np.array(vectors)
