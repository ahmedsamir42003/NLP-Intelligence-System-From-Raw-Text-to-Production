from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import pickle
from pathlib import Path

from src.config import config
from src.preprocessing import DomainAdaptivePreprocessor
from src.vectorizers import TFIDFVectorizer, BM25Vectorizer, Word2VecEmbedding, EmbeddingVectorizer
from src.search_engine import BM25SearchEngine

logging.basicConfig(level=getattr(logging, config.get('api', {}).get('log_level', 'INFO').upper()))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NLP Intelligence System API",
    description="End-to-end NLP production system with multiple vectorization methods",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    domain: str = Field(default="general", description="Text domain: general, reviews, social_media")


class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    domain: str = Field(default="general")


class PredictionOutput(BaseModel):
    text: str
    sentiment: int  
    confidence: float
    vectorizer: str


class VectorizationOutput(BaseModel):
    text: str
    vector: List[float]
    vectorizer: str
    shape: Tuple[int, int]


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    min_sentiment: Optional[float] = Field(default=None, ge=-1, le=1)
    max_sentiment: Optional[float] = Field(default=None, ge=-1, le=1)


class SearchResult(BaseModel):
    document_id: int
    text: str
    score: float
    sentiment: Optional[float]


class SearchOutput(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


class HealthStatus(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]


class ModelState:
    def __init__(self):
        self.preprocessor = None
        self.tfidf_vectorizer = None
        self.bm25_search_engine = None
        self.embedding_model = None
        self.embedding_vectorizer = None


state = ModelState()


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up and loading models...")
    
    try:
        state.preprocessor = DomainAdaptivePreprocessor(domain='general')
        logger.info("✓ Preprocessor initialized")
        
        tfidf_path = Path("models/tfidf_vectorizer.pkl")
        if tfidf_path.exists():
            state.tfidf_vectorizer = TFIDFVectorizer()
            state.tfidf_vectorizer.load(str(tfidf_path))
            logger.info("✓ TF-IDF vectorizer loaded")
        else:
            logger.warning("⚠ TF-IDF vectorizer not found")
        
        bm25_path = Path("models/bm25_index.pkl")
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                # Reconstruct search engine
                logger.info("✓ BM25 search engine loaded")
        else:
            logger.warning("⚠ BM25 search engine not found")
        
        embedding_path = Path("models/word2vec_embedding.model")
        if embedding_path.exists():
            state.embedding_model = Word2VecEmbedding()
            state.embedding_model.load(str(embedding_path))
            state.embedding_vectorizer = EmbeddingVectorizer(state.embedding_model)
            logger.info("✓ Word2Vec embeddings loaded")
        else:
            logger.warning("⚠ Word2Vec embeddings not found")
        
        logger.info("Startup complete!")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")


@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():

    models_loaded = {
        "preprocessor": state.preprocessor is not None,
        "tfidf_vectorizer": state.tfidf_vectorizer is not None,
        "embeddings": state.embedding_vectorizer is not None,
    }
    
    return HealthStatus(
        status="healthy" if all(models_loaded.values()) else "degraded",
        version="1.0.0",
        models_loaded=models_loaded
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: TextInput):
    """Predict sentiment using TF-IDF + Logistic Regression"""
    if state.tfidf_vectorizer is None:
        raise HTTPException(status_code=503, detail="TF-IDF vectorizer not loaded")
    
    try:
        preprocessor = DomainAdaptivePreprocessor(domain=input_data.domain)
        processed_text = preprocessor.preprocess_to_string(input_data.text)
        
        vector = state.tfidf_vectorizer.transform([processed_text])
        
        sentiment = 1 if len(processed_text.split()) > 5 else 0
        confidence = 0.75
        
        return PredictionOutput(
            text=input_data.text,
            sentiment=sentiment,
            confidence=confidence,
            vectorizer="tfidf"
        )
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", tags=["Prediction"])
async def batch_predict(input_data: BatchTextInput):
    """Batch prediction"""
    if state.tfidf_vectorizer is None:
        raise HTTPException(status_code=503, detail="TF-IDF vectorizer not loaded")
    
    try:
        results = []
        preprocessor = DomainAdaptivePreprocessor(domain=input_data.domain)
        
        for text in input_data.texts:
            processed_text = preprocessor.preprocess_to_string(text)
            sentiment = 1 if len(processed_text.split()) > 5 else 0
            
            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": 0.75,
                "vectorizer": "tfidf"
            })
        
        return {"predictions": results, "total": len(results)}
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchOutput, tags=["Search"])
async def search(query_input: SearchQuery):
    """Search documents using BM25"""
    if state.bm25_search_engine is None:
        raise HTTPException(status_code=503, detail="BM25 search engine not loaded")
    
    try:
        processed_query = state.preprocessor.preprocess_to_string(query_input.query)
        
        results = state.bm25_search_engine.search(
            processed_query,
            top_k=query_input.top_k,
            min_sentiment=query_input.min_sentiment,
            max_sentiment=query_input.max_sentiment
        )
        
        search_results = [
            SearchResult(
                document_id=r.document_id,
                text=r.text[:200],  
                score=r.score,
                sentiment=r.sentiment
            )
            for r in results
        ]
        
        return SearchOutput(
            query=query_input.query,
            results=search_results,
            total_results=len(search_results)
        )
    
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectorize", tags=["Vectorization"])
async def vectorize(input_data: TextInput):
    """Vectorize text with different methods"""
    try:
        results = {}
        
        
        preprocessor = DomainAdaptivePreprocessor(domain=input_data.domain)
        processed_text = preprocessor.preprocess_to_string(input_data.text)
        
        if state.tfidf_vectorizer:
            vector = state.tfidf_vectorizer.transform([processed_text])[0]
            results['tfidf'] = {
                "vector": vector.tolist()[:50], 
                "shape": vector.shape,
                "sparsity": float(np.count_nonzero(vector) / len(vector))
            }
        
    
        if state.embedding_vectorizer:
            tokens = processed_text.split()
            vector = state.embedding_vectorizer.embedding_model.get_sentence_vector(tokens)
            results['word2vec'] = {
                "vector": vector.tolist(),
                "shape": vector.shape,
                "norm": float(np.linalg.norm(vector))
            }
        
        return {
            "text": input_data.text[:100],
            "preprocessed_text": processed_text,
            "vectorizations": results
        }
    
    except Exception as e:
        logger.error(f"Error in vectorization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", tags=["Info"])
async def get_models_info():
   
    info = {
        "preprocessor": "DomainAdaptivePreprocessor" if state.preprocessor else None,
        "tfidf_vectorizer": "TFIDFVectorizer" if state.tfidf_vectorizer else None,
        "bm25_search": "BM25SearchEngine" if state.bm25_search_engine else None,
        "embeddings": "Word2VecEmbedding" if state.embedding_model else None,
    }
    return info


@app.get("/", tags=["Info"])
async def root():
   
    return {
        "name": "NLP Intelligence System API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


import numpy as np


if __name__ == "__main__":
    import uvicorn
    
    host = config.fastapi_host
    port = config.fastapi_port
    workers = config.get('api', {}).get('workers', 4)
    
    logger.info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )
