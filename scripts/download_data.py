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

    logger.info(f"Parsing Amazon reviews from {file_path}")
    
    data = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                if max_samples and line_num >= max_samples:
                    break
                
                try:

                    parts = line.strip().split('\n')
                    product_id = None
                    title = None
                    category = None
                    price = None
                    text = None
                    rating = None
                    
                    for part in parts:
                        if part.startswith('product/productId:'):
                            product_id = part.split(':')[1].strip()
                        elif part.startswith('product/title:'):
                            title = part.split(':', 1)[1].strip()
                        elif part.startswith('product/price:'):
                            price = part.split(':')[1].strip()
                        elif part.startswith('review/text:'):
                            text = part.split(':', 1)[1].strip()
                        elif part.startswith('review/score:'):
                            rating = int(float(part.split(':')[1].strip()))
                    
                    if text and rating:
                        data.append({
                            'text': text,
                            'rating': rating,
                            'product_id': product_id,
                            'title': title,
                            'price': price
                        })
                except Exception as e:
                    logger.debug(f"Error parsing line {line_num}: {e}")
                    continue
    
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
                        
                        if target == 0:
                            sentiment = 0
                        elif target == 4:
                            sentiment = 1
                        else:
                            sentiment = 1
                        
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
    
    df.to_csv(f"{processed_path.stem}_full.csv", index=False)
    
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
    
    df.to_csv(f"{processed_path.stem}_full.csv", index=False)
    
    df[['text', 'sentiment']].to_csv(processed_path, index=False)
    
    logger.info(f"Saved to {processed_path}")
    return df


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Starting data download...")
        
    print("Data download completed. Use DVC for production setup:")
    print("  dvc add data/raw/")
    print("  dvc push")
