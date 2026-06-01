@echo off
REM Quick setup script for NLP Intelligence System (Windows)
REM Run: setup.bat

echo NLP Intelligence System - Setup Script
echo ======================================

REM Check Python version
echo Checking Python version...
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

REM Download NLTK data
echo Downloading NLTK data...
python -m nltk.downloader punkt stopwords wordnet

REM Create .env from template
if not exist .env (
    echo Creating .env file...
    copy .env.example .env
    echo Created .env - please review and update with your settings
)

REM Create necessary directories
echo Creating directories...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist models mkdir models
if not exist logs mkdir logs
if not exist metrics mkdir metrics

REM Initialize git if not already done
if not exist .git (
    echo Initializing Git repository...
    git init
)

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Edit .env with your DVC and MLflow settings
echo 2. Run: python scripts\initialize_project.py
echo 3. Run experiments: python -m experiments.train
echo 4. View MLflow: mlflow ui
echo 5. Start API: python -m api.main
echo.
echo Documentation: See README.md
pause
