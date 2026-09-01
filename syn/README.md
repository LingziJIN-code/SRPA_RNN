# SRPA: Sequential Quadratic Programming for Training RNNs on the synthetic dataset

## Overview

This repository implements and compares optimization methods for training Elman RNNs with ReLU activation:

- **SRPA (SPLQP)**: Sequential regularized piecewise affine algorithm
- **GD methods**: Gradient Descent, with clipping (GDC), with Nesterov momentum (GDNes)
- **SGD methods**: Stochastic Gradient Descent, Adam

## Project Structure

```
syn/
├── SRPA-main/SRPA/          # Core SRPA implementation
│   ├── main.py              # Entry point
│   ├── SPLQP.py             # SRPA(SPLQP) algorithm
│   ├── gen_synthetic_dataset.py  # generating synthetic dataset
│   └── synT10_SPLQP_tuning.py   # tune the parameters of SRPA(SPLQP) on the synthetic dataset
│   ├── dependency_packages.py     # Package installation and imports
│   └── mypackages/                # Helper functions
├── Compare_SRPA_GDs_SGDs/   # Comparison experiments
│   ├── synT10_GDs_SGDs.py
│   └── synT10_auto_tuning_2_repeat.py
├── plotfigures4.py          # Error comparison plots
└── plot_splqp_feasibilitymax_res_2.py  # Feasibility and optimality plots
```

## Installation

```bash
pip install numpy pandas scipy matplotlib tensorflow keras cvxpy
```

## Quick Start

```python
# Generate dataset
from gen_synthetic_dataset import generate_synthetic_dataset
generate_synthetic_dataset(Nh=4, Nx=5, Ny=3, T_total=10, mean_true=0, stddev_true=0.8, seed=123456, standardize=True)

# Train with SRPA(SPLQP)
from SPLQP import SPLQP_ReLU_optimization
results = SPLQP_ReLU_optimization(
    dataset_name="./SynDataset_Nh4_Nx5_Ny3_T10",
    Ny=3, Nh=4, maxiter=100, distribution_type="Gaussian",
    mean=0, std=1e-1, beta1=1, beta2=1,
    lambda1=1.2/(4*3), lambda2=1.2/(4*4), lambda3=1.2/(5*4),
    lambda4=1.2/4, lambda5=1.2/3, eta1=0.9, eta2=1.3, rho1=0.5, print_option=2
)
```

## Commands

Run the following commands in the `syn` directory.

Given the files `SynDataset_Nh4_Nx5_Ny3_T10.csv` and `SynDataset_Nh4_Nx5_Ny3_T10_new.txt`. 

```bash
# Ensure you are in the syn directory

# SRPA
python SRPA-main/SRPA/synT10_SPLQP_tuning.py    # Skip this step if the parameters are already tuned
python SRPA-main/SRPA/main.py

# SGDs
python Compare_SRPA_GDs_SGDs/synT10_auto_tuning_2_repeat.py    # Skip this step if the parameters are already tuned
python Compare_SRPA_GDs_SGDs/synT10_GDs_SGDs.py
 
# Plot and compare results
python plotfigures4.py
python plot_splqp_feasibilitymax_res_2.py
```

## Key Outputs

| File | Description |
|------|-------------|
| `synT10_Errors_*.csv` | Training results per optimizer |
| `optimizer_*_comparison.png` and `optimizer_iterations_*.png` | Error comparison plots | 
| `splqp_*.png` | SRPA feasibility/residual plots |

## Citation

> **A sequential regularized piecewise affine algorithm for nonconvex nonsmooth multicomposite optimization in RNN training**  
> Lingzi Jin, Xiao Wang, Xiaojun Chen  

 
