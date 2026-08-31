# An Augmented Lagrangian Method (ALM) with Block Coordinate Descent (BCD) for Training Recurrent Neural Networks (RNNs)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

**ALMBCD_RNN** is a Python package implementing an optimization algorithm that leverages the Augmented Lagrangian Method (ALM) with Block Coordinate Descent (BCD) to train Elman RNNs. The **ALMBCD_RNN** is particularly effective at addressing the gradient vanishing and gradient exploding problems that often arise when using (Stochastic) Gradient Descent methods to train RNNs. By formulating the training process as a constrained optimization problem and solving it using the ALM with BCD, **ALMBCD_RNN** stabilizes the training process, ensuring more reliable convergence and improved model performance.

This package is based on the paper:

> **Title:** An Augmented Lagrangian Method for Training Recurrent Neural Networks  
> **Authors:** Yue Wang, Chao Zhang, Xiaojun Chen  
> **Year:** 2024  
> **arXiv ID:** [2402.13687](https://arxiv.org/abs/2402.13687)  
> **DOI:** [10.48550/arXiv.2402.13687](https://doi.org/10.48550/arXiv.2402.13687)
> 
> **Abstract.** Recurrent Neural Networks (RNNs) are widely used to model sequential data in a wide range of areas, such as natural language processing, speech recognition, machine translation, and time series analysis. In this paper, we model the training process of RNNs with the ReLU activation function as a constrained optimization problem with a smooth nonconvex objective function and piecewise smooth nonconvex constraints. We prove that any feasible point of the optimization problem satisfies the no nonzero abnormal multiplier constraint qualification (NNAMCQ), and any local minimizer is a Karush-Kuhn-Tucker  (KKT) point of the problem. Moreover, we propose an augmented Lagrangian method (ALM) and design an efficient block coordinate descent (BCD) method to solve the subproblems of the ALM.
The update of each block of the BCD method has a closed-form solution. The stop criterion for the inner loop is easy to check and can be stopped in finite steps. Moreover, we show that the BCD method can generate a directional stationary point of the subproblem. Furthermore, we establish the global convergence of the ALM to a KKT point of the constrained optimization problem. Compared with the state-of-the-art algorithms, numerical results demonstrate the efficiency and effectiveness of the ALM for training RNNs.

## Functions

This package includes two primary functions:
 
### `ALM_BCD_ReLU`

- **Description**: This function applies the ALM with BCD method to train an Elman RNN using the ReLU (Rectified Linear Unit) activation function.
- **Input Parameters**:
  - `Nh`: Number of hidden units
  - `Ny`: Number of output units
  - `Nx`: Number of input units
  - `T`: Size of the training set
  - `gamma0`: Initial value of penalty parameters (default: 1)
  - `lambda1` to `lambda6`: Regularization parameters.
  - `eta1`: Update stop criterion (default: 0.99)
  - `eta2`: Update parameter for gamma (default: 5/6)
  - `eta3`: Order for xi norm (default: 0.01)
  - `eta4`: Update ratio for epo (default: 5/6)
  - `epo_0`: Initial value of \epsilon_0 (default: 0.1)
  - `epo_star`: Stop criterion for epo_k (default: 1e-20)
  - `sigma`: Penalty term parameter \mu (default: 1e-5)
  - `ALM_m`: Number of iterations for non-monotone line search (default: 1)
  - `submaxiter`: Maximum iterations for subproblems (default: 100)
- **Returns**:
  - A dictionary containing the training error, test error, feasibility violations, and the time taken for each iteration of the ALM_BCD including corresponding initial values.


### `gen_synthetic`

- **Description**: This function is used to generate and save a synthetic dataset that can be used for training and testing ALMBCD_RNN.
- **Usage**: The `gen_synthetic` function creates a synthetic sequencical dataset based on the specified parameters, which can be used to simulate different scenarios for RNN training.
- **Input Parameters**:
  - `Nh` (int): Number of hidden units.
  - `Nx` (int): Dimensionality of the input data.
  - `Ny` (int): Dimensionality of the output data.
  - `T_total` (int): Total time series length.
  - `mean_true` (float): Mean for generating parameters A, W, V, b, c.
  - `stddev_true` (float): Standard deviation for generating parameters A, W, V, b, c.
  - `e_mean` (float): Mean of the noise, default is 0.
  - `e_stddev` (float): Standard deviation of the noise, default is 1e-3.
  - `seed` (int): Random seed for reproducibility, default is 123456.
  - `standardize` (bool): Whether to standardize the data, default is True.
- **Returns**:
  - None. The function saves the generated dataset directly to a CSV file.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/YueWANG25/ALMRNN.git
cd ALMRNN/ALMRNN
```

## Usage

To see an example of how to use the functions in this package, you can refer to the main.py file. This script demonstrates how to:
- Import a real-world dataset or generate a synthetic dataset.
- Train Recurrent Neural Networks (RNNs) using the ALM with BCD optimization method, with both ReLU and ELU activation functions.

### Running the example
```python
python main.py
```
The script will load the specified dataset, perform the optimization, and output the results.



