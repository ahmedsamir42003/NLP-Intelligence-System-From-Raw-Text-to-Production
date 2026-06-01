import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._load_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_env_vars(self):

        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.mlflow_exp = os.getenv('MLFLOW_EXPERIMENT_NAME', 'nlp_text_vectorization')
        self.dvc_remote = os.getenv('DVC_REMOTE_URL')
        self.fastapi_host = os.getenv('FASTAPI_HOST', '0.0.0.0')
        self.fastapi_port = int(os.getenv('FASTAPI_PORT', 8000))
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'word2vec')
        self.embedding_dim = int(os.getenv('EMBEDDING_DIM', 300))
        self.random_seed = int(os.getenv('RANDOM_SEED', 42))
    
    def get(self, key: str, default=None) -> Any:
        
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def get_datasets(self) -> Dict[str, Any]:
        return self.config.get('datasets', {})
    
    def get_preprocessing(self) -> Dict[str, Any]:
        return self.config.get('preprocessing', {})
    
    def get_vectorization(self) -> Dict[str, Any]:
        return self.config.get('vectorization', {})
    
    def get_search_config(self) -> Dict[str, Any]:
        return self.config.get('search_engine', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        return self.config.get('api', {})
    
    def get_experiments_config(self) -> Dict[str, Any]:
        return self.config.get('experiments', {})
    
    @property
    def test_split(self) -> float:
        return self.get_experiments_config().get('test_split', 0.2)


config = Config()
