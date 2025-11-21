#!/usr/bin/env python3
"""
Test script to verify the Speech Summarizer API setup
Run this to check if everything is configured correctly
"""

import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print("  ✓ Python version is compatible (3.8+)")
        return True
    else:
        print(f"  ✗ Python version {version.major}.{version.minor} is too old. Need 3.8+")
        return False

def check_file_structure():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'summarizer.py',
        'requirements.txt',
        'README.md',
        'render.yaml',
        'Dockerfile',
        'Procfile',
        'runtime.txt',
        '.env.example',
        '.gitignore',
        'templates/index.html',
        'static/style.css',
        'static/script.js'
    ]

    print("\nChecking file structure...")
    all_exist = True

    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            all_exist = False

    return all_exist

def check_dependencies():
    """Check which dependencies are installed"""
    dependencies = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('sentence_transformers', 'Sentence-Transformers'),
        ('nltk', 'NLTK'),
        ('sklearn', 'scikit-learn'),
        ('numpy', 'NumPy'),
    ]

    print("\nChecking dependencies...")
    missing = []

    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - NOT INSTALLED")
            missing.append(display_name)

    return len(missing) == 0, missing

def main():
    print("=" * 60)
    print("Speech Summarizer API - Setup Verification")
    print("=" * 60)

    # Check Python version
    python_ok = check_python_version()

    # Check file structure
    files_ok = check_file_structure()

    # Check dependencies
    deps_ok, missing_deps = check_dependencies()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if python_ok and files_ok:
        print("✓ Python version: OK")
        print("✓ File structure: OK")
    else:
        if not python_ok:
            print("✗ Python version: FAILED")
        if not files_ok:
            print("✗ File structure: MISSING FILES")

    if deps_ok:
        print("✓ Dependencies: ALL INSTALLED")
        print("\n🎉 Setup is complete! You can run the API with:")
        print("   python app.py")
    else:
        print(f"✗ Dependencies: {len(missing_deps)} MISSING")
        print("\n📦 To install dependencies, run:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print("   pip install -r requirements.txt")
        print("   python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"")

    print("\n📚 For more information, see README.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
