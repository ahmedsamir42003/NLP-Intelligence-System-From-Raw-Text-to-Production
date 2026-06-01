import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import mlflow
import mlflow.sklearn
import pandas as pd

from src.preprocessing import TextPreprocessor, DomainAdaptivePreprocessor
from src.vectorizers import BoWVectorizer, TFIDFVectorizer, BM25Vectorizer, Word2VecEmbedding, EmbeddingVectorizer
from src.config import config

logger = logging.getLogger(__name__)


class NLPExperiment:
    
    def __init__(self, experiment_name: str, tracking_uri: str = None):
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.run = None
    
    def start_run(self, run_name: str, params: Dict[str, Any]):

        self.run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        logger.info(f"Started MLflow run: {run_name}")
    
    def log_metrics(self, metrics: Dict[str, float]):

        mlflow.log_metrics(metrics)
    
    def log_artifact(self, path: str):

        mlflow.log_artifact(path)
    
    def end_run(self):

        if self.run:
            mlflow.end_run()
    
    def get_run_id(self) -> str:

        if self.run:
            return self.run.info.run_id
        return None


class VectorizationExperiment(NLPExperiment):
    
    def __init__(self, texts: np.ndarray, labels: np.ndarray,
                 domains: list = None,
                 test_size: float = 0.2,
                 random_seed: int = 42):

        super().__init__("vectorization_comparison", config.mlflow_uri)
        
        self.texts = texts
        self.labels = labels
        self.domains = domains or ['general', 'reviews', 'social_media']
        self.test_size = test_size
        self.random_seed = random_seed
        
        self.X_train_texts, self.X_test_texts, self.y_train, self.y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_seed
        )
        
        self.results = {}
    
    def preprocess_texts(self, texts: list, domain: str = 'general') -> list:

        preprocessor = DomainAdaptivePreprocessor(domain=domain)
        preprocessed = []
        for text in texts:
            processed = preprocessor.preprocess_to_string(text)
            preprocessed.append(processed)
        return preprocessed
    
    def train_bow(self):

        logger.info("Training BoW classifier...")
        self.start_run(
            "bow_logistic_regression",
            {
                'vectorizer': 'bow',
                'classifier': 'logistic_regression',
                'domain': 'general'
            }
        )
        
        try:

            X_train_prep = self.preprocess_texts(self.X_train_texts)
            X_test_prep = self.preprocess_texts(self.X_test_texts)
            
            vectorizer = BoWVectorizer(max_features=5000)
            X_train_vec = vectorizer.fit_transform(X_train_prep)
            X_test_vec = vectorizer.transform(X_test_prep)
            
            clf = LogisticRegression(max_iter=1000, random_state=self.random_seed)
            clf.fit(X_train_vec, self.y_train)
            
            y_pred = clf.predict(X_test_vec)
            metrics = self._compute_metrics(self.y_test, y_pred)
            
            self.log_metrics(metrics)
            self.results['bow'] = metrics
            
            logger.info(f"BoW Results: {metrics}")
        
        except Exception as e:
            logger.error(f"Error in BoW training: {e}")
        
        finally:
            self.end_run()
    
    def train_tfidf(self):

        logger.info("Training TF-IDF classifier...")
        self.start_run(
            "tfidf_logistic_regression",
            {
                'vectorizer': 'tfidf',
                'classifier': 'logistic_regression',
                'domain': 'general'
            }
        )
        
        try:

            X_train_prep = self.preprocess_texts(self.X_train_texts)
            X_test_prep = self.preprocess_texts(self.X_test_texts)
            
            vectorizer = TFIDFVectorizer(max_features=5000)
            X_train_vec = vectorizer.fit_transform(X_train_prep)
            X_test_vec = vectorizer.transform(X_test_prep)
            
            clf = LogisticRegression(max_iter=1000, random_state=self.random_seed)
            clf.fit(X_train_vec, self.y_train)
            
            y_pred = clf.predict(X_test_vec)
            metrics = self._compute_metrics(self.y_test, y_pred)
            
            self.log_metrics(metrics)
            self.results['tfidf'] = metrics
            
            logger.info(f"TF-IDF Results: {metrics}")
        
        except Exception as e:
            logger.error(f"Error in TF-IDF training: {e}")
        
        finally:
            self.end_run()
    
    def train_embeddings(self):

        logger.info("Training embedding-based classifier...")
        self.start_run(
            "word2vec_logistic_regression",
            {
                'vectorizer': 'word2vec',
                'classifier': 'logistic_regression',
                'domain': 'general'
            }
        )
        
        try:
            X_train_prep = self.preprocess_texts(self.X_train_texts)
            X_test_prep = self.preprocess_texts(self.X_test_texts)
            
            X_train_tokens = [text.split() for text in X_train_prep]
            X_test_tokens = [text.split() for text in X_test_prep]
            
            embedding_model = Word2VecEmbedding(vector_size=300, epochs=5)
            embedding_model.fit(X_train_tokens)
            
            vectorizer = EmbeddingVectorizer(embedding_model)
            X_train_vec = vectorizer.transform(X_train_tokens)
            X_test_vec = vectorizer.transform(X_test_tokens)
            
            clf = LogisticRegression(max_iter=1000, random_state=self.random_seed)
            clf.fit(X_train_vec, self.y_train)
            
            y_pred = clf.predict(X_test_vec)
            metrics = self._compute_metrics(self.y_test, y_pred)
            
            self.log_metrics(metrics)
            self.results['embeddings'] = metrics
            
            logger.info(f"Embeddings Results: {metrics}")
        
        except Exception as e:
            logger.error(f"Error in embedding training: {e}")
        
        finally:
            self.end_run()
    
    def run_all(self):

        logger.info("Running vectorization comparison experiments...")
        self.train_bow()
        self.train_tfidf()
        self.train_embeddings()
        
        return self.results
    
    @staticmethod
    def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['roc_auc'] = 0.0
        
        return metrics


class DomainComparisonExperiment(NLPExperiment):
    
    def __init__(self, amazon_df: pd.DataFrame, sentiment140_df: pd.DataFrame,
                 test_size: float = 0.2, random_seed: int = 42):
        super().__init__("domain_comparison", config.mlflow_uri)
        
        self.amazon_df = amazon_df
        self.sentiment140_df = sentiment140_df
        self.test_size = test_size
        self.random_seed = random_seed
        self.results = {}
    
    def run_comparison(self):

        logger.info("Running domain comparison experiments...")
        
        datasets = {
            'amazon': (self.amazon_df['text'].values, self.amazon_df['sentiment'].values),
            'sentiment140': (self.sentiment140_df['text'].values, self.sentiment140_df['sentiment'].values)
        }
        
        for dataset_name, (texts, labels) in datasets.items():
            logger.info(f"Processing {dataset_name}...")
            
            exp = VectorizationExperiment(texts, labels, test_size=self.test_size)
            results = exp.run_all()
            
            self.results[dataset_name] = results
        
        return self.results


if __name__ == "__main__":
    import os
    from pathlib import Path
    

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("NLP Experiments Pipeline")
    logger.info("=" * 80)
    
    mlflow_uri = config.mlflow_uri or "http://localhost:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info(f"MLflow tracking URI: {mlflow_uri}")
    
    data_dir = Path("data/processed")
    amazon_path = data_dir / "amazon_reviews_processed.csv"
    sentiment140_path = data_dir / "sentiment140_processed.csv"
    
    if not amazon_path.exists() or not sentiment140_path.exists():
        logger.error("Dataset files not found!")
        logger.error(f"Expected: {amazon_path}")
        logger.error(f"Expected: {sentiment140_path}")
        logger.info("Run: python scripts/initialize_project.py")
        exit(1)
    
    logger.info(f"Loading Amazon Reviews from {amazon_path}...")
    amazon_df = pd.read_csv(amazon_path)
    logger.info(f"  - Loaded {len(amazon_df)} reviews")
    
    logger.info(f"Loading Sentiment140 from {sentiment140_path}...")
    sentiment140_df = pd.read_csv(sentiment140_path)
    logger.info(f"  - Loaded {len(sentiment140_df)} tweets")
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Vectorization Comparison (Amazon Reviews)")
    logger.info("=" * 80)
    
    vec_exp = VectorizationExperiment(
        amazon_df['text'].values,
        amazon_df['sentiment'].values,
        test_size=config.test_split,
        random_seed=config.random_seed
    )
    vec_results = vec_exp.run_all()
    
    logger.info("\nVectorization Experiment Results:")
    for method, metrics in vec_results.items():
        logger.info(f"  {method.upper()}: {metrics}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Domain Comparison (Amazon vs Sentiment140)")
    logger.info("=" * 80)
    
    domain_exp = DomainComparisonExperiment(
        amazon_df,
        sentiment140_df,
        test_size=config.test_split,
        random_seed=config.random_seed
    )
    domain_results = domain_exp.run_comparison()
    
    logger.info("\nDomain Comparison Results:")
    for dataset_name, results in domain_results.items():
        logger.info(f"\n{dataset_name.upper()}:")
        for method, metrics in results.items():
            logger.info(f"  {method}: {metrics}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Experiments Complete!")
    logger.info(f"View results in MLflow: {mlflow_uri}")
    logger.info("=" * 80)
