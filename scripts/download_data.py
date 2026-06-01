import os
import gzip
import csv
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> bool:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"File already exists: {output_path}")
        return True

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Downloaded: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        # Remove partial file if download failed
        if output_path.exists():
            output_path.unlink()
        return False


def download_amazon_reviews(output_path: str = "data/raw/amazon_reviews.txt.gz") -> bool:
    url = "https://snap.stanford.edu/data/finefoods.txt.gz"
    logger.info("Downloading Amazon Fine Food Reviews...")
    return download_file(url, output_path)


def download_sentiment140(output_path: str = "data/raw/sentiment140.csv.gz") -> bool:
    url = "http://help.sentiment140.appspot.com/files/training.1600000.processed.noemoticon.csv.gz"
    logger.info("Downloading Sentiment140...")
    return download_file(url, output_path)


def parse_amazon_reviews(file_path: str, max_samples: int = None) -> pd.DataFrame:
    """
    Parse Amazon Fine Foods reviews from block-format .txt.gz file.
    Each review is a block of key:value lines separated by blank lines.
    """
    logger.info(f"Parsing Amazon reviews from {file_path}")

    data = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            current = {}
            for line in f:
                line = line.strip()

                if not line:

                    if current.get('text') and current.get('rating'):
                        data.append(current)
                        if max_samples and len(data) >= max_samples:
                            break
                    current = {}
                    continue

                if line.startswith('product/productId:'):
                    current['product_id'] = line.split(':', 1)[1].strip()
                elif line.startswith('product/title:'):
                    current['title'] = line.split(':', 1)[1].strip()
                elif line.startswith('product/price:'):
                    current['price'] = line.split(':', 1)[1].strip()
                elif line.startswith('review/text:'):
                    current['text'] = line.split(':', 1)[1].strip()
                elif line.startswith('review/score:'):
                    try:
                        current['rating'] = int(float(line.split(':', 1)[1].strip()))
                    except ValueError:
                        pass
            if current.get('text') and current.get('rating'):
                data.append(current)

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")

    df = pd.DataFrame(data)
    logger.info(f"Parsed {len(df)} reviews")
    return df


def parse_sentiment140(file_path: str, max_samples: int = None) -> pd.DataFrame:

    logger.info(f"Parsing Sentiment140 from {file_path}")

    data = []
    try:
        open_func = gzip.open if file_path.endswith('.gz') else open
        with open_func(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONE, quotechar='')

            for idx, row in enumerate(reader):
                if max_samples and idx >= max_samples:
                    break

                try:
                    if len(row) >= 6:
                        target = int(row[0])
                        text = row[5]

                        sentiment = 0 if target == 0 else 1

                        data.append({
                            'text': text,
                            'sentiment': sentiment,
                            'original_target': target
                        })
                except Exception as e:
                    logger.debug(f"Error parsing row {idx}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")

    df = pd.DataFrame(data)
    logger.info(f"Parsed {len(df)} sentiment samples")
    return df


def prepare_amazon_reviews(raw_path: str, processed_path: str, max_samples: int = None):

    logger.info("Preparing Amazon reviews...")

    df = parse_amazon_reviews(raw_path, max_samples=max_samples)

    if len(df) == 0:
        logger.warning("No data parsed from Amazon reviews")
        return None

    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    df['sentiment'] = (df['rating'] >= 4).astype(int)

    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    full_path = processed_path.parent / f"{processed_path.stem}_full.csv"
    df.to_csv(full_path, index=False)
    logger.info(f"Saved full version to {full_path}")

    df[['text', 'sentiment']].to_csv(processed_path, index=False)
    logger.info(f"Saved to {processed_path}")

    return df


def prepare_sentiment140(raw_path: str, processed_path: str, max_samples: int = None):

    logger.info("Preparing Sentiment140...")

    df = parse_sentiment140(raw_path, max_samples=max_samples)

    if len(df) == 0:
        logger.warning("No data parsed from Sentiment140")
        return None

    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)

    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    full_path = processed_path.parent / f"{processed_path.stem}_full.csv"
    df.to_csv(full_path, index=False)
    logger.info(f"Saved full version to {full_path}")

    df[['text', 'sentiment']].to_csv(processed_path, index=False)
    logger.info(f"Saved to {processed_path}")

    return df


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    logger.info("Starting data download...")

    print("Data download completed. Use DVC for production setup:")
    print("  dvc add data/raw/")
    print("  dvc push")