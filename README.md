# NLP Intelligence System: From Raw Text to Production


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
└──────────────────────────────────────────────┘
```

*dagshub_repo* : `https://dagshub.com/ahmedsamir42003/NLP-Intelligence-System-From-Raw-Text-to-Production-`
