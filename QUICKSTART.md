# Quick Start Guide

Get up and running with the NLP Intelligence System in 5 minutes.

## 🚀 Fastest Path to Success

### Step 1: Setup (2 minutes)

**Windows:**
```bash
setup.bat
```

**macOS/Linux:**
```bash
bash setup.sh
```

This will:
- Create Python virtual environment
- Install all dependencies
- Download NLTK data
- Create configuration files

### Step 2: Initialize Data (1 minute)

```bash
# For testing with small sample (recommended first)
python scripts/initialize_project.py --sample-size 10000

# For full dataset (takes longer)
python scripts/initialize_project.py
```

### Step 3: Run Examples (30 seconds)

```bash
python examples.py
```

You should see:
- Text preprocessing examples
- TF-IDF vectorization
- BM25 search results
- Word2Vec training

### Step 4: Start API (30 seconds)

In one terminal:
```bash
python -m api.main
```

### Step 5: Test API

In another terminal:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/docs
```

**API ready at**: `http://localhost:8000`
**API docs**: `http://localhost:8000/docs`

---

## 🧪 Run Your First Experiment

### Option A: Quick Test

```bash
python -c "
from src.preprocessing import DomainAdaptivePreprocessor
proc = DomainAdaptivePreprocessor(domain='reviews')
text = 'This product is amazing! Highly recommend it!'
print('Tokens:', proc.preprocess(text))
"
```

### Option B: Full MLflow Experiment

```bash
# Terminal 1: Start MLflow UI
mlflow ui

# Terminal 2: Run experiments
python -m experiments.train

# Terminal 3: View results at http://localhost:5000
```

---

## 📊 Load Your Own Data

```python
import pandas as pd
from src.preprocessing import DomainAdaptivePreprocessor
from src.vectorizers import TFIDFVectorizer

# Load your data
df = pd.read_csv('your_data.csv')

# Preprocess
preprocessor = DomainAdaptivePreprocessor(domain='reviews')
texts = [preprocessor.preprocess_to_string(t) for t in df['text']]

# Vectorize
vectorizer = TFIDFVectorizer(max_features=5000)
vectors = vectorizer.fit_transform(texts)

print(f"Vectorized {len(texts)} texts")
print(f"Feature matrix shape: {vectors.shape}")
```

---

## 🐳 Deploy with Docker

```bash
# Build and start
docker-compose -f docker/docker-compose.yml up --build

# Wait for services to start (2-3 minutes)

# Test
curl http://localhost:8000/health

# View logs
docker-compose -f docker/docker-compose.yml logs -f nlp-api
```

Services:
- **API**: http://localhost:8000
- **MLflow**: http://localhost:5000
- **API Docs**: http://localhost:8000/docs

---

## ⚙️ Configuration

### .env Setup

```bash
# Copy template
cp .env.example .env

# Edit .env with:
# For GitLab DVC (recommended)
DVC_REMOTE_URL=s3://your-bucket/dvc-storage

# For local MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# For production (in docker-compose)
MLFLOW_TRACKING_URI=http://mlflow:5000
```

---

## 🔍 Troubleshooting

### "ModuleNotFoundError"
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### "NLTK data not found"
```bash
python -m nltk.downloader punkt stopwords wordnet
```

### "Port 8000 already in use"
```bash
# Kill process or use different port
python -m api.main --port 8001
```

### "Docker issues"
```bash
# Clean build
docker-compose -f docker/docker-compose.yml build --no-cache
```

---

## 📚 Key Files

- **README.md** - Full documentation
- **examples.py** - Working code examples
- **config.yaml** - System configuration
- **.env** - Environment variables (create from .env.example)
- **requirements.txt** - Python dependencies

---

## 🎯 Next Steps

1. ✅ Setup complete? 
2. ⏭️ Read [README.md](README.md) for detailed documentation
3. 🧪 Run experiments and view results in MLflow
4. 🔍 Explore different vectorization methods
5. 🚀 Deploy to production

---

## 💡 Pro Tips

### Use Domain-Adaptive Preprocessing
```python
# Automatically configured for your domain
processor = DomainAdaptivePreprocessor(domain='reviews')
processor = DomainAdaptivePreprocessor(domain='social_media')
processor = DomainAdaptivePreprocessor(domain='news')
```

### Compare Vectorizers Side-by-Side
```python
from src.vectorizers import BoWVectorizer, TFIDFVectorizer, BM25Vectorizer

# All use same interface
bow = BoWVectorizer()
tfidf = TFIDFVectorizer()
bm25 = BM25Vectorizer()

bow.fit(texts)
tfidf.fit(texts)
bm25.fit(texts)
```

### Track Experiments Automatically
```python
# MLflow logs everything
from experiments.train import VectorizationExperiment

exp = VectorizationExperiment(texts, labels)
results = exp.run_all()

# View at: mlflow ui
```

### Search with Sentiment Filter
```python
from src.search_engine import BM25SearchEngine

engine = BM25SearchEngine(documents)
results = engine.search(
    "product quality",
    min_sentiment=0.5  # Only positive reviews
)
```

---

**Ready to start?** Run `setup.bat` (Windows) or `bash setup.sh` (macOS/Linux)!

Questions? See README.md for comprehensive documentation.
