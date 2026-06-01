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


    for directory in ["data/raw", "data/processed", "models", "logs", "metrics"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
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

    amazon_raw = Path("data/raw/amazon_reviews.txt.gz")
    if not amazon_raw.exists():
        logger.info("Downloading Amazon Fine Food Reviews...")
        if download_amazon_reviews(str(amazon_raw)):
            logger.info("✓ Amazon reviews downloaded")
        else:
            logger.warning("⚠ Failed to download Amazon reviews")

    sentiment140_raw = Path("data/raw/sentiment140.csv.gz")
    if not sentiment140_raw.exists():
        logger.info("Downloading Sentiment140...")
        if download_sentiment140(str(sentiment140_raw)):
            logger.info("✓ Sentiment140 downloaded")
        else:
            logger.warning("⚠ Failed to download Sentiment140")

    logger.info("\nPreparing datasets...")

    if amazon_raw.exists():
        logger.info("Processing Amazon reviews...")
        amazon_processed = Path("data/processed/amazon_reviews.csv")
        amazon_df = prepare_amazon_reviews(
            str(amazon_raw),
            str(amazon_processed),
            max_samples=dataset_sample_size
        )
        if amazon_df is not None and len(amazon_df) > 0:
            logger.info(f"✓ Prepared {len(amazon_df)} Amazon reviews")
            logger.info(f"  Saved to {amazon_processed}")
            logger.info(f"  Full version saved to {amazon_processed.parent / (amazon_processed.stem + '_full.csv')}")
        else:
            logger.warning("⚠ No Amazon reviews were prepared — check the raw file")
    else:
        logger.warning("⚠ Amazon raw file not found, skipping")

    if sentiment140_raw.exists():
        logger.info("Processing Sentiment140...")
        sentiment140_processed = Path("data/processed/sentiment140.csv")
        sentiment140_df = prepare_sentiment140(
            str(sentiment140_raw),
            str(sentiment140_processed),
            max_samples=dataset_sample_size
        )
        if sentiment140_df is not None and len(sentiment140_df) > 0:
            logger.info(f"✓ Prepared {len(sentiment140_df)} Sentiment140 samples")
            logger.info(f"  Saved to {sentiment140_processed}")
        else:
            logger.warning("⚠ No Sentiment140 samples were prepared — check the raw file")
    else:
        logger.warning("⚠ Sentiment140 raw file not found, skipping")

    # Verify outputs
    logger.info("\nVerifying processed data...")
    for f in Path("data/processed").glob("*.csv"):
        size = f.stat().st_size / 1024
        logger.info(f"  {f.name}: {size:.1f} KB")

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