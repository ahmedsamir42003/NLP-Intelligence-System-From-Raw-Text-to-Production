# Deployment & Production Setup

Complete guide to deploying the NLP Intelligence System with GitLab, DVC, and MLflow.

---

## 📋 Table of Contents

- [Local Development Setup](#local-development-setup)
- [DVC + GitLab Configuration](#dvc--gitlab-configuration)
- [MLflow Setup](#mlflow-setup)
- [Docker Deployment](#docker-deployment)
- [Production Checklist](#production-checklist)

---

## Local Development Setup

### 1. Initial Setup

```bash
# Clone repository
git clone <your-repo>
cd NLP-Intelligence-System-From-Raw-Text-to-Production

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet
```

### 2. Configure .env

```bash
cp .env.example .env

# Edit .env with your settings
```

### 3. Initialize DVC

```bash
dvc init

# Verify DVC is initialized
ls -la .dvc/
```

---

## DVC + GitLab Configuration

### Option 1: DVC with GitLab Remote Storage (Recommended)

**Prerequisites**:
- GitLab account
- GitLab project created
- GitLab runner or CI/CD enabled

**Setup Steps**:

#### 1.1 Create GitLab Project

```bash
# Log in to GitLab, create new project
# Example: https://gitlab.com/your-username/nlp-system
```

#### 1.2 Configure DVC Remote for GitLab

```bash
# Option A: Using GitLab Storage (if available)
dvc remote add -d gitlab-storage \
  gs+ssh://your-gitlab-instance/project-id/dvc-storage

# Option B: Using S3 (more common)
dvc remote add -d myremote s3://your-bucket/nlp-dvc

# Option C: Using Azure Blob Storage
dvc remote add -d myremote \
  azure://your-container/path

# Set default remote
dvc remote default myremote
```

#### 1.3 Update .env

```env
# .env
DVC_REMOTE_URL=s3://your-bucket/nlp-dvc
DVC_REMOTE_NAME=myremote
DVC_S3_ACCESS_KEY_ID=your_access_key
DVC_S3_SECRET_ACCESS_KEY=your_secret_key
```

#### 1.4 Configure AWS S3 Credentials (if using S3)

```bash
# Option A: Using environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# Option B: Using AWS credentials file
# ~/.aws/credentials
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key

# Option C: Using IAM role (recommended for production)
# Attach S3 access policy to your EC2/ECS task role
```

#### 1.5 Add Data Files to DVC

```bash
# Track data files
dvc add data/raw/
dvc add data/processed/

# Commit DVC files
git add data/.gitignore
git add data/raw.dvc
git add data/processed.dvc
git commit -m "Add DVC tracking for data"

# Push to remote storage
dvc push

# On another machine, pull data
dvc pull
```

### Option 2: DVC with Local Storage

```bash
# For testing/development only
dvc remote add -d local-storage /tmp/dvc-storage
dvc push
```

### DVC Pipeline

**dvc.yaml** is already configured. To run:

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro prepare_amazon

# View pipeline
dvc dag
```

---

## MLflow Setup

### Option 1: Local MLflow (Development)

```bash
# Start MLflow server
mlflow ui

# Access at http://localhost:5000

# Data stored in: mlruns/ directory
```

### Option 2: MLflow with PostgreSQL (Production)

#### 2.1 Create PostgreSQL Database

```bash
# Using Docker
docker run -d \
  --name mlflow-postgres \
  -e POSTGRES_USER=mlflow \
  -e POSTGRES_PASSWORD=secure_password \
  -e POSTGRES_DB=mlflow \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:14-alpine

# Verify connection
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "SELECT version();"
```

#### 2.2 Create Artifact Storage

```bash
# Using S3
aws s3 mb s3://mlflow-artifacts-bucket

# Or using local directory
mkdir -p /opt/mlflow/artifacts
chmod 777 /opt/mlflow/artifacts
```

#### 2.3 Start MLflow Server

```bash
# With PostgreSQL backend and S3 artifacts
mlflow server \
  --backend-store-uri postgresql://mlflow:secure_password@localhost:5432/mlflow \
  --default-artifact-root s3://mlflow-artifacts-bucket \
  --host 0.0.0.0 \
  --port 5000

# Access at http://localhost:5000
```

#### 2.4 Update .env

```env
# .env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=nlp_text_vectorization
```

### Option 3: MLflow in Docker Compose

**docker-compose.yml** already includes MLflow. Run:

```bash
docker-compose -f docker/docker-compose.yml up

# MLflow available at http://localhost:5000
```

### MLflow Configuration in Code

```python
import mlflow
from src.config import config

# Set tracking URI
mlflow.set_tracking_uri(config.mlflow_uri)

# Set experiment
mlflow.set_experiment(config.mlflow_exp)

# Log experiment
with mlflow.start_run(run_name="vectorization_comparison"):
    mlflow.log_params({
        'vectorizer': 'tfidf',
        'max_features': 5000
    })
    
    mlflow.log_metrics({
        'accuracy': 0.85,
        'f1': 0.83
    })
    
    mlflow.sklearn.log_model(model, "model")
```

---

## Docker Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and start all services
cd docker
docker-compose up --build

# Services running:
# - NLP API: http://localhost:8000
# - MLflow: http://localhost:5000

# Stop services
docker-compose down

# View logs
docker-compose logs -f nlp-api
docker-compose logs -f mlflow
```

### Option 2: Kubernetes Deployment

**Prerequisites**:
- Kubernetes cluster (minikube, EKS, GKE, etc.)
- kubectl configured
- Docker image pushed to registry

#### Setup Steps

```bash
# 1. Build and push Docker image
docker build -f docker/Dockerfile -t your-registry/nlp-intelligence:latest .
docker push your-registry/nlp-intelligence:latest

# 2. Create namespace
kubectl create namespace nlp-production

# 3. Create secrets for environment variables
kubectl create secret generic nlp-secrets \
  --from-literal=mlflow-uri=http://mlflow:5000 \
  --from-literal=dvc-remote-url=s3://bucket/path \
  -n nlp-production

# 4. Deploy services (example manifests below)
kubectl apply -f k8s/mlflow-deployment.yaml -n nlp-production
kubectl apply -f k8s/nlp-api-deployment.yaml -n nlp-production
kubectl apply -f k8s/service.yaml -n nlp-production

# 5. Check deployment
kubectl get pods -n nlp-production
kubectl logs -f deployment/nlp-api -n nlp-production
```

**Example k8s/nlp-api-deployment.yaml**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nlp-api
  namespace: nlp-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nlp-api
  template:
    metadata:
      labels:
        app: nlp-api
    spec:
      containers:
      - name: nlp-api
        image: your-registry/nlp-intelligence:latest
        ports:
        - containerPort: 8000
        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            secretKeyRef:
              name: nlp-secrets
              key: mlflow-uri
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: nlp-api-service
  namespace: nlp-production
spec:
  type: LoadBalancer
  selector:
    app: nlp-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

### Option 3: Cloud Platform Deployment

#### AWS ECS

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name nlp-production

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster nlp-production \
  --service-name nlp-api \
  --task-definition nlp-api \
  --desired-count 3 \
  --load-balancers targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=nlp-api,containerPort=8000
```

#### Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/nlp-intelligence

# Deploy to Cloud Run
gcloud run deploy nlp-api \
  --image gcr.io/YOUR_PROJECT/nlp-intelligence \
  --platform managed \
  --region us-central1 \
  --set-env-vars MLFLOW_TRACKING_URI=https://mlflow.example.com \
  --memory 2Gi \
  --cpu 2
```

---

## GitLab CI/CD Pipeline

**Create .gitlab-ci.yml**:

```yaml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  DOCKER_DRIVER: overlay2

test:
  stage: test
  image: python:3.10
  before_script:
    - pip install -r requirements.txt
    - python -m nltk.downloader punkt stopwords wordnet
  script:
    - pytest tests/ -v --cov=src
    - black --check src/ api/ experiments/
    - flake8 src/ api/ experiments/

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -f docker/Dockerfile -t $DOCKER_IMAGE .
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker push $DOCKER_IMAGE

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/nlp-api nlp-api=$DOCKER_IMAGE -n nlp-production
    - kubectl rollout status deployment/nlp-api -n nlp-production
  only:
    - main
```

---

## Production Checklist

- [ ] DVC configured with remote storage (S3/GitLab)
- [ ] MLflow configured with PostgreSQL backend
- [ ] Environment variables (.env) properly set
- [ ] Docker image built and tested
- [ ] Docker Compose services start successfully
- [ ] API endpoints respond correctly
- [ ] Health checks passing
- [ ] Logging configured and monitored
- [ ] Metrics collected (Prometheus/CloudWatch)
- [ ] Backup strategy for data and models
- [ ] Disaster recovery plan documented
- [ ] SSL/TLS certificates configured
- [ ] Rate limiting implemented
- [ ] Authentication/authorization setup
- [ ] Load testing completed
- [ ] Security scanning passed (OWASP, CVE checks)
- [ ] Documentation updated
- [ ] Team trained on deployment process

---

## Troubleshooting

### DVC Issues

```bash
# Check DVC status
dvc status

# List remotes
dvc remote list

# Test remote connection
dvc remote list -v

# Pull/push with verbose output
dvc pull -v
dvc push -v
```

### MLflow Issues

```bash
# Check tracking URI
echo $MLFLOW_TRACKING_URI

# View experiments
mlflow experiments search

# List runs
mlflow runs list -e experiment_id

# Clean up old runs
mlflow gc --backend-store-uri <URI>
```

### Docker Issues

```bash
# View logs
docker-compose logs -f

# Rebuild image
docker-compose build --no-cache

# Remove old images
docker image prune

# Check resource usage
docker stats
```

---

## Monitoring & Maintenance

### MLflow Model Registry

```python
# Register model
mlflow.sklearn.log_model(model, "model", registered_model_name="nlp-classifier")

# Transition stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="nlp-classifier",
    version=1,
    stage="Production"
)
```

### Prometheus Metrics

Add to your code:

```python
from prometheus_client import Counter, Histogram
import time

request_count = Counter('nlp_api_requests_total', 'Total API requests')
request_duration = Histogram('nlp_api_request_duration_seconds', 'Request duration')

@app.post("/predict")
@request_duration.time()
def predict(text):
    request_count.inc()
    # Your code
```

---

## Security Best Practices

1. **Credentials**:
   - Use environment variables, never hardcode
   - Rotate keys regularly
   - Use IAM roles for cloud services

2. **Data**:
   - Encrypt at rest and in transit
   - Restrict access to sensitive data
   - Audit data access

3. **Code**:
   - Regular security scanning (OWASP ZAP, Snyk)
   - Keep dependencies updated
   - Code reviews before production

4. **Infrastructure**:
   - Use VPC/network isolation
   - Enable logging and monitoring
   - Regular backups
   - Disaster recovery testing

---

**For questions, see README.md or contact the team.**
