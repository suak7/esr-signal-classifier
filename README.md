# Electron Spin Resonance (ESR) Signal Classifier

## Project Overview
<p>A PyTorch-based neural network that demonstrates AI signal classification, dataset generation, model training, and performance visualization. The goal is to classify synthetic electron spin resonance (ESR) signals into three distinct classes based on their characteristic patterns:</p>

* Class 0: Single Gaussian-like peak
* Class 1: Double Gaussian peaks
* Class 2: Irregular / high-frequency noisy pattern

The dataset is designed to mimic realistic ESR measurements, including natural variability and noise.

## Features
* Generates synthetic signals with different patterns: Class 0, Class 1, and Class 2
* Noise addition to simulate realistic experimental conditions
* Splitting of dataset into training and test sets using PyTorch DataLoader
* Visualizes performance curves

## Getting Started

Requirements:

* <a href="https://www.python.org/downloads/">Python 3.13+</a>
* <a href="https://numpy.org/install/">NumPy</a>
* <a href="https://matplotlib.org/stable/install/index.html">Matplotlib</a>
* <a href="https://pytorch.org/get-started/locally/">PyTorch</a>

Install Packages:
```bash
# Create + activate venv (macOS / Linux)
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install numpy matplotlib torch scikit-learn
```

## Generating the Plots
```bash
# Generate dataset
python3 data_generation.py

# Train the model
python3 train.py

# Visualize training curves
python3 visualize.py
```

* Tip: If you want to increase complexity in the signals, increase noise, increase dataset variability, add validation split, or use dropout.
