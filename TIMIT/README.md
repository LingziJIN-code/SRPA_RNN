# SRPA: Sequential Regularized Piecewise Affine Algorithm for Training RNNs on the TIMIT dataset

## Overview

This repository implements and compares optimization methods for training Elman RNNs with ReLU activation on the TIMIT audio denoising task:

- **SRPA (SPLQP)**: Sequential regularized piecewise affine algorithm
- **GD methods**: Gradient Descent, with clipping (GDC), with Nesterov momentum (GDNes)
- **SGD methods**: Stochastic Gradient Descent, Adam

## Project Structure

```
TIMIT/
├── SRPA/                          # Core SRPA implementation
│   ├── main_N.py                  # Entry point for SRPA training on N-sample datasets
│   ├── SPLQP_N.py                 # SRPA (SPLQP) algorithm for N-sample datasets
│   ├── TIMIT_SPLQP_tuning.py     # Tune SRPA parameters on TIMIT data
│   ├── dependency_packages.py     # Package installation and imports
│   └── mypackages/                # Helper functions 
├── Compare_SRPA_GDs_SGDs/         # Comparison experiments
│   ├── TIMIT_GDs_SGDs.py          # GD and SGD training
│   └── TIMIT_auto_tuning_2_repeat.py  # Auto-tuning for GD/SGD
├── mixed_fre_timit_mini.pkl       # Input features (mixed/noisy frequency-domain)
├── ori_fre_timit_mini.pkl         # Target features (original/clean frequency-domain)
├── plotfigures4_TIMIT.py          # Error comparison plots
└── plot_splqp_feasibilitymax_res_2.py  # Feasibility and residual plots
```

## Dataset

The TIMIT dataset consists of two pickle files containing frequency-domain audio features:

- **Input features**: `mixed_fre_timit_mini.pkl` (mixed/noisy frequency-domain features)
- **Target features**: `ori_fre_timit_mini.pkl` (original/clean frequency-domain features)
- **Dimensions**: N=70 sequences, T=126 timesteps, Nx=Ny=129 frequency bins
- **Data splitting**: 70% training (N_train=49), 30% testing (N_test=21)

## Installation

```bash
pip install numpy pandas scipy matplotlib tensorflow keras cvxpy
```

## Quick Start

```python
# Train with SRPA (SPLQP)
from SPLQP_N import SPLQP_N_ReLU_optimization

results = SPLQP_N_ReLU_optimization(
    Nh=2, maxiter=600,
    distribution_type="Gaussian",
    mean=0, std=1e-2,
    beta1=0.04, beta2=0.04,
    lambda1=1/(2*129), lambda2=1/(2*2), lambda3=1/(129*2),
    lambda4=1/2, lambda5=1/129,
    eta1=0.7, eta2=1.1, rho1=0.003,
    print_option=2
)
```

## Commands

Run the following commands in the `TIMIT` directory.

### SRPA Training

```bash
# Tune SRPA parameters (optional)
python SRPA/TIMIT_SPLQP_tuning.py

# Train with SRPA
python SRPA/main_N.py
```

### GD and SGD Training

```bash
# Auto-tune parameters for GD/SGD (optional)
python Compare_SRPA_GDs_SGDs/TIMIT_auto_tuning_2_repeat.py

# Train with GD, GDC, GDNes, SGD, Adam
python Compare_SRPA_GDs_SGDs/TIMIT_GDs_SGDs.py
```

### Plot Results

```bash
# Generate error comparison plots
python plotfigures4_TIMIT.py

# Generate SRPA feasibility and residual plots
python plot_splqp_feasibilitymax_res_2.py
```

## Key Outputs

| File | Description |
|------|-------------|
| `TIMIT_Errors_*.csv` | Training results per optimizer |
| `optimizer_*_comparison.png` and `optimizer_iterations_*.png` | Error comparison plots |
| `splqp_*.png` | SRPA feasibility/residual plots |


## Citation

> **A sequential regularized piecewise affine algorithm for nonconvex nonsmooth multicomposite optimization in RNN training**  
> Lingzi Jin, Xiao Wang, Xiaojun Chen