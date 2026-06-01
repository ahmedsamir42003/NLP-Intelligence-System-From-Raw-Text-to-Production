#!/bin/bash
# Quick setup script for NLP Intelligence System
# Run: bash setup.sh

echo "NLP Intelligence System - Setup Script"
echo "======================================"

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -m nltk.downloader punkt stopwords wordnet

# Create .env from template
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ Created .env - please review and update with your settings"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models logs metrics

# Initialize git if not already done
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
fi

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your DVC and MLflow settings"
echo "2. Run: python scripts/initialize_project.py"
echo "3. Run experiments: python -m experiments.train"
echo "4. View MLflow: mlflow ui"
echo "5. Start API: python -m api.main"
echo ""
echo "Documentation: See README.md"
