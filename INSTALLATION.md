# Installation Guide - Trading Journal Analytics

## Prerequisites

- **Python 3.8 or higher** (3.10+ recommended)
- **pip** (Python package manager)
- **Git** (optional, for version control)

---

## Quick Start

### 1. Check Python Version
```bash
python3 --version
# Should show Python 3.8 or higher
```

### 2. Create Virtual Environment (Recommended)
```bash
# Navigate to project directory
cd "/Users/ashwathramsundar/Desktop/Database Project/Final"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

**Option A: Install Minimal Requirements** (Recommended for quick start)
```bash
pip3 install -r requirements-minimal.txt
```

**Option B: Install All Requirements** (Full functionality)
```bash
pip3 install -r requirements.txt
```

### 4. Verify Installation
```bash
# Check if key packages are installed
python3 -c "import flask, pandas, numpy, sklearn, xgboost; print('All core packages installed!')"
```

### 5. Run the Application
```bash
# Start the Flask web server
python3 app.py
```

Then open your browser to: **http://127.0.0.1:5000**

---

## Detailed Installation Steps

### For macOS Users

#### Using Homebrew (Recommended)
```bash
# Install Python 3 if not already installed
brew install python@3.11

# Verify installation
python3 --version

# Install dependencies
pip3 install -r requirements.txt
```

#### For M1/M2 Mac Users (Apple Silicon)
If you're using Apple Silicon, you may need special TensorFlow builds:

```bash
# Install TensorFlow for Apple Silicon
pip3 install tensorflow-macos
pip3 install tensorflow-metal

# Then install other requirements
pip3 install -r requirements-minimal.txt
```

### For Windows Users

#### Using Python from python.org
```bash
# Make sure pip is up to date
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### For Linux Users

#### Ubuntu/Debian
```bash
# Install Python 3 and pip
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Install dependencies
pip3 install -r requirements.txt
```

---

## Troubleshooting Common Issues

### Issue 1: TensorFlow Installation Fails

**Solution for Mac M1/M2:**
```bash
pip3 uninstall tensorflow
pip3 install tensorflow-macos
pip3 install tensorflow-metal
```

**Solution for Windows/Linux:**
```bash
pip3 install tensorflow --upgrade
```

### Issue 2: XGBoost Installation Fails

**Solution:**
```bash
# Install XGBoost separately
pip3 install xgboost --no-cache-dir
```

### Issue 3: "No module named 'flask'"

**Solution:**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install Flask
pip3 install Flask
```

### Issue 4: Database Not Found

**Solution:**
```bash
# Verify database path in config.py
# Make sure Trading_Journal_ML.db exists in the project folder
ls -l Trading_Journal_ML.db
```

### Issue 5: Port 5000 Already in Use

**Solution:**
```bash
# Option 1: Kill process using port 5000
# On macOS/Linux:
lsof -ti:5000 | xargs kill -9

# Option 2: Run Flask on different port
# Edit app.py and change:
# app.run(debug=True, port=5001)
```

---

## Package Descriptions

### Core Libraries
- **numpy**: Numerical computing library
- **pandas**: Data manipulation and analysis
- **Flask**: Web framework for the UI

### Machine Learning
- **scikit-learn**: Machine learning algorithms and tools
- **xgboost**: Gradient boosting framework
- **tensorflow/keras**: Deep learning library for LSTM models

### Data Visualization
- **matplotlib**: Basic plotting library
- **seaborn**: Statistical data visualization
- **plotly**: Interactive charts

### Financial Data
- **yfinance**: Download market data from Yahoo Finance

### AI Integration
- **google-generativeai**: Google Gemini AI integration

---

## Verifying Your Installation

Run this test script to verify all components:

```bash
python3 << 'EOF'
import sys
print(f"Python version: {sys.version}")

try:
    import flask
    print("✓ Flask installed")
except ImportError:
    print("✗ Flask NOT installed")

try:
    import pandas
    print("✓ Pandas installed")
except ImportError:
    print("✗ Pandas NOT installed")

try:
    import numpy
    print("✓ NumPy installed")
except ImportError:
    print("✗ NumPy NOT installed")

try:
    import sklearn
    print("✓ scikit-learn installed")
except ImportError:
    print("✗ scikit-learn NOT installed")

try:
    import xgboost
    print("✓ XGBoost installed")
except ImportError:
    print("✗ XGBoost NOT installed")

try:
    import tensorflow
    print("✓ TensorFlow installed")
except ImportError:
    print("✗ TensorFlow NOT installed")

try:
    import google.generativeai
    print("✓ Google Generative AI installed")
except ImportError:
    print("✗ Google Generative AI NOT installed")

print("\n✓ Installation verification complete!")
EOF
```

---

## Development Environment Setup

### Using VS Code (Recommended)

1. **Install VS Code Extensions:**
   - Python (Microsoft)
   - Pylance
   - Flask Snippets

2. **Configure Python Interpreter:**
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Python: Select Interpreter"
   - Choose the virtual environment (`venv`)

3. **Run Flask App:**
   - Press `F5` to start debugging
   - Or run `python3 app.py` in terminal

### Using PyCharm

1. **Open Project:**
   - File → Open → Select project folder

2. **Configure Interpreter:**
   - Settings → Project → Python Interpreter
   - Add Interpreter → Virtual Environment → Existing

3. **Run Configuration:**
   - Run → Edit Configurations
   - Add Python configuration
   - Script path: `app.py`

---

## Updating Dependencies

### Update All Packages
```bash
pip3 install --upgrade -r requirements.txt
```

### Update Specific Package
```bash
pip3 install --upgrade flask
```

### Freeze Current Environment
```bash
# Save currently installed versions
pip3 freeze > requirements-locked.txt
```

---

## Uninstallation

### Remove Virtual Environment
```bash
# Deactivate virtual environment
deactivate

# Remove venv folder
rm -rf venv
```

### Uninstall All Packages
```bash
pip3 uninstall -r requirements.txt -y
```

---

## Production Deployment

### Using Gunicorn (Linux/Mac)
```bash
# Install gunicorn
pip3 install gunicorn

# Run production server
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Waitress (Windows)
```bash
# Install waitress
pip3 install waitress

# Run production server
waitress-serve --host 0.0.0.0 --port 5000 app:app
```

---

## Getting Help

If you encounter issues:

1. **Check Error Messages**: Read the full error output
2. **Google the Error**: Search for the exact error message
3. **Check Package Versions**: Ensure compatibility
4. **Use Virtual Environment**: Isolates dependencies
5. **Update pip**: `pip3 install --upgrade pip`

---

## Additional Resources

- **Flask Documentation**: https://flask.palletsprojects.com/
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **TensorFlow Guide**: https://www.tensorflow.org/guide
- **XGBoost Docs**: https://xgboost.readthedocs.io/
- **Python Virtual Environments**: https://docs.python.org/3/tutorial/venv.html

---

*Last Updated: 2025-11-27*
*For issues or questions, refer to the project README*
