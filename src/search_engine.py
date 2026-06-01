from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from src.vectorizers import BM25Vectorizer
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:

    document_id: int
    text: str
    score: float
    sentiment: float = None
    metadata: Dict = None


class SentimentAnalyzer:
    
    def __init__(self):
        from textblob import TextBlob
        self.textblob = TextBlob
    
    def analyze(self, text: str) -> float:

        blob = self.textblob(text)
        return blob.sentiment.polarity


class BM25SearchEngine:
    
    def __init__(self, documents: List[str], 
                 use_sentiment_filter: bool = True,
                 k1: float = 1.5, 
                 b: float = 0.75):
   
        self.documents = documents
        self.use_sentiment_filter = use_sentiment_filter
        
        self.bm25 = BM25Vectorizer(k1=k1, b=b)
        self.bm25.fit(documents)
        
        if use_sentiment_filter:
            self.sentiment_analyzer = SentimentAnalyzer()
            self.sentiments = [self.sentiment_analyzer.analyze(doc) for doc in documents]
            logger.info(f"Computed sentiments for {len(documents)} documents")
        else:
            self.sentiments = None
    
    def search(self, query: str, top_k: int = 10, 
               min_sentiment: Optional[float] = None,
               max_sentiment: Optional[float] = None) -> List[SearchResult]:
    
        results = self.bm25.query(query, top_k=top_k * 2)  
        
        search_results = []
        for doc_id, score in results:
            if doc_id < len(self.documents):
                sentiment = self.sentiments[doc_id] if self.sentiments else None
                
                if sentiment is not None:
                    if min_sentiment is not None and sentiment < min_sentiment:
                        continue
                    if max_sentiment is not None and sentiment > max_sentiment:
                        continue
                
                result = SearchResult(
                    document_id=doc_id,
                    text=self.documents[doc_id],
                    score=score,
                    sentiment=sentiment
                )
                search_results.append(result)
                
                if len(search_results) >= top_k:
                    break
        
        return search_results
    
    def search_batch(self, queries: List[str], top_k: int = 10,
                     min_sentiment: Optional[float] = None,
                     max_sentiment: Optional[float] = None) -> List[List[SearchResult]]:
 
        return [self.search(q, top_k, min_sentiment, max_sentiment) for q in queries]
    
    def get_document(self, doc_id: int) -> Tuple[str, Optional[float]]:
 
        if 0 <= doc_id < len(self.documents):
            text = self.documents[doc_id]
            sentiment = self.sentiments[doc_id] if self.sentiments else None
            return text, sentiment
        return None, None
    
    def get_stats(self) -> Dict:

        stats = {
            'total_documents': len(self.documents),
            'avg_doc_length': np.mean([len(doc.split()) for doc in self.documents])
        }
        
        if self.sentiments:
            stats['avg_sentiment'] = np.mean(self.sentiments)
            stats['pos_documents'] = sum(1 for s in self.sentiments if s > 0.1)
            stats['neg_documents'] = sum(1 for s in self.sentiments if s < -0.1)
            stats['neutral_documents'] = sum(1 for s in self.sentiments if -0.1 <= s <= 0.1)
        
        return stats


class HybridSearchEngine:
    
    def __init__(self, documents: List[str],
                 embedding_vectorizer=None,
                 use_sentiment_filter: bool = True,
                 bm25_weight: float = 0.6,
                 semantic_weight: float = 0.4):

        self.documents = documents
        self.embedding_vectorizer = embedding_vectorizer
        self.use_sentiment_filter = use_sentiment_filter
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        
        self.bm25 = BM25Vectorizer()
        self.bm25.fit(documents)
        
        if embedding_vectorizer:
            self.doc_embeddings = self._compute_embeddings(documents)
        else:
            self.doc_embeddings = None
        
        if use_sentiment_filter:
            self.sentiment_analyzer = SentimentAnalyzer()
            self.sentiments = [self.sentiment_analyzer.analyze(doc) for doc in documents]
        else:
            self.sentiments = None
    
    def _compute_embeddings(self, documents: List[str]) -> np.ndarray:

        tokenized = [doc.split() for doc in documents]
        return self.embedding_vectorizer.transform(tokenized)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 10,
               min_sentiment: Optional[float] = None) -> List[SearchResult]:

        bm25_results = self.bm25.query(query, top_k=top_k * 2)
        
        query_embedding = None
        if self.doc_embeddings is not None:
            query_tokens = query.split()
            query_embedding = self.embedding_vectorizer.embedding_model.get_sentence_vector(query_tokens)
 
        combined_scores = {}
        
        for doc_id, bm25_score in bm25_results:
            score = self.bm25_weight * (bm25_score / max(100, max([s for _, s in bm25_results])))
            
            if query_embedding is not None and self.doc_embeddings is not None:
                semantic_sim = self._cosine_similarity(
                    query_embedding,
                    self.doc_embeddings[doc_id]
                )
                score += self.semantic_weight * semantic_sim
            
            combined_scores[doc_id] = score
        
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if doc_id < len(self.documents):
                sentiment = self.sentiments[doc_id] if self.sentiments else None
                
                if min_sentiment is not None and sentiment and sentiment < min_sentiment:
                    continue
                
                result = SearchResult(
                    document_id=doc_id,
                    text=self.documents[doc_id],
                    score=float(score),
                    sentiment=sentiment
                )
                results.append(result)
        
        return results[:top_k]
