# SRPA: Sequential Regularized Piecewise Affine Algorithm for Training RNNs

This repository implements and compares optimization methods for training Elman RNNs with ReLU activation on three different datasets.

## Overview

The SRPA (Sequential Regularized Piecewise Affine) algorithm, formerly known as SPLQP, is compared with traditional gradient-based methods:

- **SRPA (SPLQP)**: Sequential regularized piecewise affine algorithm
- **GD methods**: Gradient Descent, with clipping (GDC), with Nesterov momentum (GDNes)
- **SGD methods**: Stochastic Gradient Descent, Adam

## Datasets

This repository includes experiments on three datasets:

1. **Synthetic Dataset** (`syn/`): Generated synthetic data 
2. **Volatility of S&P Index dataset** (`SP500/`): Financial time series prediction
3. **TIMIT Audio Denoising** (`TIMIT/`): Audio denoising task

## Project Structure

```
SRPA_RNN/
├── syn/                          # Synthetic dataset experiments
│   ├── SRPA-main/SRPA/           # Core SRPA implementation
│   ├── Compare_SRPA_GDs_SGDs/    # Comparison experiments
│   ├── plotfigures4.py           # Error comparison plots
│   └── plot_splqp_feasibilitymax_res_2.py
├── SP500/                        # Volatility of S&P Index dataset experiments
│   ├── SRPA/                     # Core SRPA implementation
│   ├── Compare_SRPA_GDs_SGDs/    # Comparison experiments
│   ├── plotfigures5_sp500.py     # Error comparison plots
│   └── plot_splqp_feasibilitymax_res_2.py
├── TIMIT/                        # TIMIT audio denoising experiments
│   ├── SRPA/                     # Core SRPA implementation
│   ├── Compare_SRPA_GDs_SGDs/    # Comparison experiments
│   ├── plotfigures4_TIMIT.py     # Error comparison plots
│   └── plot_splqp_feasibilitymax_res_2.py
└── README.md                     # This file
```

## Installation

```bash
pip install numpy pandas scipy matplotlib tensorflow keras cvxpy
```
 
## Commands

### Synthetic Dataset

```bash
cd syn

# SRPA
python SRPA-main/SRPA/synT10_SPLQP_tuning.py    #optional
python SRPA-main/SRPA/main.py

# GD and SGD
python Compare_SRPA_GDs_SGDs/synT10_auto_tuning_2_repeat.py    #optional
python Compare_SRPA_GDs_SGDs/synT10_GDs_SGDs.py

# Plot results
python plotfigures4.py
python plot_splqp_feasibilitymax_res_2.py
```

### S&P 500 Volatility

```bash
cd SP500

# SRPA Training
python SRPA/sp500_SPLQP_tuning.py    #optional
python SRPA/main.py

# GD and SGD Training
python Compare_SRPA_GDs_SGDs/sp500_auto_tuning_2_repeat.py    #optional
python Compare_SRPA_GDs_SGDs/sp500_GDs_SGDs_5.11.1.py

# Plot Results
python plotfigures5_sp500.py
python plot_splqp_feasibilitymax_res_2.py
```

### TIMIT Audio Denoising

```bash
cd TIMIT

# SRPA Training
python SRPA/TIMIT_SPLQP_tuning.py    #optional
python SRPA/main_N.py

# GD and SGD Training
python Compare_SRPA_GDs_SGDs/TIMIT_auto_tuning_2_repeat.py    #optional
python Compare_SRPA_GDs_SGDs/TIMIT_GDs_SGDs.py

# Plot Results
python plotfigures4_TIMIT.py
python plot_splqp_feasibilitymax_res_2.py
```

## Key Outputs

| Dataset | CSV Files | Plot Files |
|---------|-----------|------------|
| Synthetic | `synT10_Errors_*.csv` | `optimizer_iterations_*.png`, `splqp_*.png` |
| S&P 500 | `sp500_Errors_*.csv` | `optimizer_iterations_*.png`, `splqp_*.png` |
| TIMIT | `TIMIT_Errors_*.csv` | `optimizer_iterations_*.png`, `splqp_*.png` |

## Citation

> **A sequential regularized piecewise affine algorithm for nonconvex nonsmooth multicomposite optimization in RNN training**  
> Lingzi Jin, Xiao Wang, Xiaojun Chen

## Detailed Documentation

For more details, see the README files in the following subdirectories:

- **[syn](syn/README.md)** - synthetic dataset
- **[SP500](SP500/README.md)** - Volatility of S&P Index dataset
- **[TIMIT](TIMIT/README.md)** - TIMIT dataset
