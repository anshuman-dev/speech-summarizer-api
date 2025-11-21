#!/bin/bash
# Quick setup script for Speech Summarizer API

echo "======================================"
echo "Speech Summarizer API - Quick Setup"
echo "======================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "(This may take 2-3 minutes for first time setup)"
pip install -r requirements.txt

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "To start the API server:"
echo "  1. source venv/bin/activate"
echo "  2. python app.py"
echo ""
echo "Then open: http://localhost:5000"
echo ""
echo "To test the API:"
echo "  python test_setup.py"
echo ""
