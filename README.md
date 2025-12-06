<h1 align="center">Electron Spin Resonance (ESR) Spectrum Analyzer</h1>

<p>A PyTorch-based neural network for classifying synthetic Electron Spin Resonance (ESR) signals into three distinct spectral patterns.</p>

- **Class 0**: Single Gaussian-like peak (single ESR transition)
- **Class 1**: Double Gaussian peaks (multiple ESR transitions)
- **Class 2**: Irregular oscillating pattern (noisy/complex spectra)

## Features
* Adds configurable noise and signal variability to simulate realistic experimental conditions.
* Implements synthetic signal generation --> train/test split --> feeding into PyTorch DataLoader for efficient batch processing.
* Provides a configurable neural network architecture defined in `config.py`, allowing easy re‑tuning of model capacity and regularization.
* Includes training scripts, performance monitoring (loss/accuracy curves), and simple visualization of results for quick evaluation of model behaviour.
* New signal classes can be added, model architecture can be swapped, and data‑augmentation parameters can be adjusted.

## Installation

Requirements:

* <a href="https://www.python.org/downloads/">Python 3.13+</a>
* <a href="https://numpy.org/install/">NumPy</a>
* <a href="https://matplotlib.org/stable/install/index.html">Matplotlib</a>
* <a href="https://pytorch.org/get-started/locally/">PyTorch</a>

Install Packages:
```bash
# Clone repository
git clone 
cd esr-spectrum-analyzer

# Create virtual environment (macOS / Linux)
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install numpy matplotlib torch scikit-learn
```

## Quick Start
```bash
# 1. Generate synthetic dataset
python src/data/generate_signals.py

# 2. Train model
python src/training/train.py

# 3. Visualize training curves
python visualize.py

# 4. Results will be saved in outputs/

# Tip: If you want to increase complexity in the signals,
# increase noise, increase dataset variability, add validation split, or use dropout.
```

## Configuration
All hyperparameters can be found in `config.py`. Below are the key settings:
```python
# Data
SIGNAL_LENGTH = 1000
NUM_SAMPLES_PER_CLASS = 1000

# Model
HIDDEN_SIZES = (256, 128)
DROPOUT_RATE = 0.2

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 20
```
