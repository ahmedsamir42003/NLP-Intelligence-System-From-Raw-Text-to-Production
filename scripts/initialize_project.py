import logging
import os
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from scripts.download_data import (
    download_amazon_reviews, download_sentiment140,
    prepare_amazon_reviews, prepare_sentiment140
)


def initialize_project(dataset_sample_size: int = None):

    logger.info("="*60)
    logger.info("NLP Intelligence System - Project Initialization")
    logger.info("="*60)
    
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Created directory structure")
    
    if not Path(".env").exists() and Path(".env.example").exists():
        import shutil
        shutil.copy(".env.example", ".env")
        logger.info("✓ Created .env from .env.example")
        logger.info("  Please update .env with your GitLab/DVC/MLflow credentials")
    
    
    logger.info("\nInitializing DVC...")
    if not Path(".dvc").exists():
        os.system("dvc init")
        logger.info("✓ DVC initialized")
    
    
    logger.info("\nDownloading datasets...")
    
    if not Path("data/raw/amazon_reviews.txt.gz").exists():
        logger.info("Downloading Amazon Fine Food Reviews...")
        if download_amazon_reviews():
            logger.info("✓ Amazon reviews downloaded")
        else:
            logger.warning("⚠ Failed to download Amazon reviews")
    
    if not Path("data/raw/sentiment140.csv.gz").exists():
        logger.info("Downloading Sentiment140...")
        if download_sentiment140():
            logger.info("✓ Sentiment140 downloaded")
        else:
            logger.warning("⚠ Failed to download Sentiment140")
    
    logger.info("\nPreparing datasets...")
    
    amazon_raw = "data/raw/amazon_reviews.txt.gz"
    if Path(amazon_raw).exists():
        logger.info("Processing Amazon reviews...")
        amazon_df = prepare_amazon_reviews(
            amazon_raw,
            "data/processed/amazon_reviews.csv",
            max_samples=dataset_sample_size
        )
        if amazon_df is not None:
            logger.info(f"✓ Prepared {len(amazon_df)} Amazon reviews")
    
    sentiment140_raw = "data/raw/sentiment140.csv.gz"
    if Path(sentiment140_raw).exists():
        logger.info("Processing Sentiment140...")
        sentiment140_df = prepare_sentiment140(
            sentiment140_raw,
            "data/processed/sentiment140.csv",
            max_samples=dataset_sample_size
        )
        if sentiment140_df is not None:
            logger.info(f"✓ Prepared {len(sentiment140_df)} Sentiment140 samples")
    
    logger.info("\n" + "="*60)
    logger.info("Project initialization complete!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Update .env with your DVC and MLflow settings")
    logger.info("2. Run: python -m experiments.train")
    logger.info("3. View experiments: mlflow ui")
    logger.info("4. Deploy: docker-compose up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize NLP Intelligence System")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of samples to use (default: all)")
    args = parser.parse_args()
    
    initialize_project(dataset_sample_size=args.sample_size)
