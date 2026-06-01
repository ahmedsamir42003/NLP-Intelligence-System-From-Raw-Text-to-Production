# NLP Intelligence System: From Raw Text to Production

A comprehensive, production-grade NLP system demonstrating end-to-end pipeline implementation with multiple vectorization methods, experiment tracking, and containerized deployment.

**Learning outcomes**: Build reusable NLP pipelines, compare vectorization methods empirically, understand distributional semantics, version data with DVC, track experiments with MLflow, and deploy production APIs.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [System Components](#system-components)
- [Datasets](#datasets)
- [NLP Pipeline](#nlp-pipeline)
- [Experiments & Results](#experiments--results)
- [API Usage](#api-usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Development](#development)
- [MLflow & DVC Setup](#mlflow--dvc-setup)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This system bridges Lecture 1 NLP theory with production engineering by implementing a complete pipeline on two contrasting datasets:

### Datasets
- **Amazon Fine Food Reviews**: 568K+ e-commerce reviews (domain-specific, structured feedback)
- **Sentiment140**: 1.6M+ tweets (social media, informal, noisy)

### Key Insight
The same preprocessing, vectorization, and model pipeline produces **different** optimal results on each dataset. This forces real engineering decisions about:
- When to remove stopwords (reviews vs. tweets)
- Whether to stem or lemmatize (trade-off with semantic preservation)
- Which vectorization method suits which domain (BoW vs. TF-IDF vs. BM25 vs. embeddings)

### Learning Objectives
✅ Build reusable, configurable NLP preprocessing class  
✅ Compare BoW, TF-IDF, BM25, and word embeddings empirically  
✅ Visualize word embeddings and understand distributional semantics  
✅ Version data with DVC and track experiments with MLflow  
✅ Implement BM25-based search engine with sentiment filtering  
✅ Deploy production-grade FastAPI service in Docker  

---

## 🏗️ Architecture

```
Raw Data (Amazon, Sentiment140)
         ↓
    [DVC Versioning]
         ↓
  Text Preprocessing (Domain-Adaptive)
         ↓
  Tokenization, Stemming, Lemmatization
         ↓
  ┌─────────┬──────────┬────────┬─────────────┐
  ↓         ↓          ↓        ↓             ↓
 BoW      TF-IDF      BM25   Word2Vec     GloVe
  ↓         ↓          ↓        ↓             ↓
 LR        LR        Search    LR           LR
  ↓         ↓          ↓        ↓             ↓
┌──────────────────────────────────────────────┐
│      [MLflow Experiment Tracking]            │
│  - Metrics, Parameters, Artifacts            │
│  - Compare across datasets/methods           │
└──────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────┐
│    [FastAPI Service]                         │
│  /predict  /search  /vectorize  /health     │
└──────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────┐
│    [Docker Deployment]                       │
│  with MLflow tracking, PostgreSQL (optional) │
└──────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. **Prerequisites**
```bash
# Check requirements
python --version  # 3.9+
git --version
docker --version  # for deployment
```

### 2. **Clone & Setup Environment**
```bash
# Navigate to project
cd NLP-Intelligence-System-From-Raw-Text-to-Production

# Create virtual environment
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet
```

### 3. **Configure Environment**
```bash
# Copy example .env
cp .env.example .env

# Edit .env with your settings
# For local development, defaults should work
# For GitLab DVC: Set DVC_REMOTE_URL, credentials
# For MLflow: Can use local SQLite or hosted server
```

### 4. **Initialize Data**
```bash
# Download and prepare datasets (first run)
python scripts/initialize_project.py

# Or with smaller sample for testing:
python scripts/initialize_project.py --sample-size 10000
```

### 5. **Run Experiments**
```bash
# Launch MLflow UI (optional, for visualization)
mlflow ui
# Open http://localhost:5000

# Run all vectorization experiments
python -m experiments.train

# Results automatically logged to MLflow
```

### 6. **Start API Server**
```bash
# Development mode
python -m api.main

# Server available at http://localhost:8000
# OpenAPI docs: http://localhost:8000/docs
```

### 7. **Deploy with Docker**
```bash
# Start all services (API + MLflow)
docker-compose -f docker/docker-compose.yml up

# API: http://localhost:8000
# MLflow: http://localhost:5000
```

---

## 📊 Datasets

### Amazon Fine Food Reviews
- **Source**: Stanford snap dataset
- **Size**: 568,454 reviews (~240 MB)
- **Fields**: product ID, title, price, text, rating (1-5)
- **Characteristics**: Longer, well-formed sentences; domain-specific vocabulary

### Sentiment140
- **Source**: Twitter sentiment analysis dataset
- **Size**: 1,600,000 tweets (~340 MB)
- **Fields**: sentiment (0/4), ID, date, flag, user, text
- **Characteristics**: Short, informal language; noisy, less structured

---

## 📁 Project Structure

```
NLP-Intelligence-System/
├── src/                    # Core NLP modules
│   ├── preprocessing.py    # Text preprocessing
│   ├── vectorizers.py      # BoW, TF-IDF, BM25, embeddings
│   └── search_engine.py    # Search with sentiment filtering
├── experiments/            # MLflow experiment tracking
├── api/                    # FastAPI service
├── docker/                 # Docker & Docker Compose
├── scripts/                # Data download & setup
├── data/                   # Datasets (DVC tracked)
├── models/                 # Trained models
└── README.md              # This file
```

---

## 🚀 Quick Reference

### Commands
```bash
# Setup
python scripts/initialize_project.py
pip install -r requirements.txt

# Run
python -m experiments.train      # Train models
python -m api.main              # Start API server
mlflow ui                        # View experiments

# Deploy
docker-compose -f docker/docker-compose.yml up

# Test
curl http://localhost:8000/health
```

### API Endpoints
- `GET /health` - Health check
- `POST /predict` - Sentiment prediction
- `POST /search` - Document search with BM25
- `POST /vectorize` - Text vectorization
- `GET /docs` - OpenAPI documentation

---

## 💡 Key Concepts

- **Text Preprocessing**: Domain-adaptive cleaning, tokenization, lemmatization
- **Vectorization**: BoW, TF-IDF, BM25, Word2Vec embeddings
- **Experiment Tracking**: MLflow for reproducible ML
- **Data Versioning**: DVC for dataset management
- **Production API**: FastAPI with containerized deployment

---

## 📚 References

- NLTK for preprocessing
- scikit-learn for vectorization
- Gensim for embeddings
- FastAPI for web service
- MLflow for experiment tracking
- Docker for deployment

---

**Build something amazing!** 🚀