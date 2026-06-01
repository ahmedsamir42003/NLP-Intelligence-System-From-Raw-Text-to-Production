"""
Example usage of the NLP Intelligence System
"""
import pandas as pd
from src.preprocessing import DomainAdaptivePreprocessor
from src.vectorizers import TFIDFVectorizer, BM25Vectorizer, Word2VecEmbedding
from src.search_engine import BM25SearchEngine

# Example 1: Text Preprocessing
print("="*60)
print("Example 1: Text Preprocessing")
print("="*60)

# Create preprocessor for reviews domain
review_prep = DomainAdaptivePreprocessor(domain='reviews')

sample_text = """
This product is AMAZING!!! Best purchase ever. Fast shipping,
great quality. Check out http://example.com for more info.
Definitely recommend to anyone looking for quality products!
"""

tokens = review_prep.preprocess(sample_text)
print(f"\nOriginal: {sample_text[:50]}...")
print(f"Tokens: {tokens}")

# Example 2: TF-IDF Vectorization
print("\n" + "="*60)
print("Example 2: TF-IDF Vectorization")
print("="*60)

texts = [
    "great product amazing quality",
    "terrible quality waste money",
    "excellent value highly recommend",
    "good product reasonable price"
]

tfidf = TFIDFVectorizer(max_features=100)
vectors = tfidf.fit_transform(texts)
print(f"\nVectorized {len(texts)} texts")
print(f"Vector shape: {vectors.shape}")
print(f"Feature count: {len(tfidf.get_feature_names())}")

# Example 3: BM25 Search
print("\n" + "="*60)
print("Example 3: BM25 Search Engine")
print("="*60)

documents = [
    "fantastic product exceeded expectations",
    "poor quality breaks easily",
    "outstanding value great deal",
    "average nothing special",
    "best purchase amazing quality"
]

search_engine = BM25SearchEngine(documents, use_sentiment_filter=True)
results = search_engine.search("great quality product", top_k=3)

print(f"\nSearching: 'great quality product'")
for result in results:
    print(f"  Doc {result.document_id}: {result.text}")
    print(f"    Score: {result.score:.3f}, Sentiment: {result.sentiment:.2f}")

# Example 4: Word Embeddings
print("\n" + "="*60)
print("Example 4: Word2Vec Embeddings")
print("="*60)

# Prepare training data
tokenized_docs = [doc.split() for doc in documents]

# Train model
w2v = Word2VecEmbedding(vector_size=100, window=3, epochs=10)
w2v.fit(tokenized_docs)

print(f"\nTrained Word2Vec model")
print(f"Vocabulary size: {len(w2v.model.wv)}")

# Get similar words
similar = w2v.get_most_similar("quality", topn=3)
print(f"Words similar to 'quality': {similar}")

print("\n" + "="*60)
print("Examples complete! Check API docs for more details.")
print("="*60)
