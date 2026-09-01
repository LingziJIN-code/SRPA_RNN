# SRPA: Sequential Regularized Piecewise Affine Algorithm for Training RNNs on the Volatility of S\&P Index dataset

## Overview

This repository implements and compares optimization methods for training Elman RNNs with ReLU activation:

- **SRPA (SPLQP)**: Sequential regularized piecewise affine algorithm
- **GD methods**: Gradient Descent, with clipping (GDC), with Nesterov momentum (GDNes)
- **SGD methods**: Stochastic Gradient Descent, Adam

## Project Structure

```
SP500/
├── SRPA/                          # Core SRPA implementation
│   ├── main.py                    # Entry point for SRPA training
│   ├── SPLQP.py                   # SRPA (SPLQP) algorithm
│   ├── sp500_SPLQP_tuning.py     # Tune SRPA parameters on S&P 500 data
│   ├── dependency_packages.py     # Package installation and imports
│   └── mypackages/                # Helper functions 
├── Compare_SRPA_GDs_SGDs/         # Comparison experiments
│   ├── sp500_GDs_SGDs_5.11.1.py  # GD and SGD training
│   └── sp500_auto_tuning_2_repeat.py  # Auto-tuning for GD/SGD 
├── plotfigures5_sp500.py          # Error comparison plots
└── plot_splqp_feasibilitymax_res_2.py  # Feasibility and residual plots
```

## Dataset

The `clean_SP500.csv` file contains the Volatility of S\&P Index dataset with 11 input features and 1 output target:

- **Input features**: DP, EP, MKT, HML, SMB, STR, TB, TS, DEF, INF, IP
- **Output target**: RV_SP500 (realized volatility)
- **Total samples**: 437 time steps
- **Data splitting**: 90% training, 10% testing

## Installation

```bash
pip install numpy pandas scipy matplotlib tensorflow keras cvxpy
```

## Quick Start

```python
# Train with SRPA (SPLQP)
from SRPA.SPLQP import SPLQP_ReLU_optimization

results = SPLQP_ReLU_optimization(
    dataset_name="clean_SP500",
    Ny=1, Nh=20, maxiter=1000,
    distribution_type="Glorot",
    mean=0, std=20,
    beta1=0.1, beta2=0.1,
    lambda1=1/(20*1), lambda2=1/(20*20), lambda3=1/(11*20),
    lambda4=1/20, lambda5=1/1,
    eta1=0.7, eta2=1.1, rho1=0.03,
    print_option=2
)
```

## Commands

Run the following commands in the `SP500` directory.

### SRPA Training

```bash
# Tune SRPA parameters (optional)
python SRPA/sp500_SPLQP_tuning.py

# Train with SRPA
python SRPA/main.py
```

### GD and SGD Training

```bash
# Auto-tune parameters for GD/SGD (optional)
python Compare_SRPA_GDs_SGDs/sp500_auto_tuning_2_repeat.py

# Train with GD, GDC, GDNes, SGD, Adam
python Compare_SRPA_GDs_SGDs/sp500_GDs_SGDs_5.11.1.py
```

### Plot Results

```bash
# Generate error comparison plots
python plotfigures5_sp500.py

# Generate SRPA feasibility and residual plots
python plot_splqp_feasibilitymax_res_2.py
```

## Key Outputs

| File | Description |
|------|-------------|
| `sp500_Errors_*.csv` | Training results per optimizer |
| `optimizer_*_comparison.png` and `optimizer_iterations_*.png` | Error comparison plots |
| `splqp_*.png` | SRPA feasibility/residual plots |


## Citation

> **A sequential regularized piecewise affine algorithm for nonconvex nonsmooth multicomposite optimization in RNN training**  
> Lingzi Jin, Xiao Wang, Xiaojun Chen
