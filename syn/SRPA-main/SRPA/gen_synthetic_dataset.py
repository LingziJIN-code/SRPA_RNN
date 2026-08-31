
#%% 
import numpy as np
import pandas as pd
import random

#%% Helper functions
def CalculY(x_dataset, A, W, V, b, c, T, Nx, Ny, Nh):
    # This function is used to calculate y^hat by A, W, V, b, c recurrently nor auxi1liary variables h and u
    for t in range(T):
        x_t = x_dataset[t, :].reshape(Nx, 1)
        if t == 0:
            ut_t = V @ x_t + b
            ut = ut_t
            ht_t = np.maximum(ut_t, 0)
            ht = ht_t
            y_hat_t = A @ ht_t + c
            y_hat = y_hat_t.T
        else:
            ut_t = W @ ht_t + V @ x_t + b
            ut = np.concatenate((ut, ut_t), axis=1)
            ht_t = np.maximum(ut_t, 0)
            ht = np.concatenate((ht, ht_t), axis=1)
            y_hat_t = A @ ht_t + c
            y_hat = np.concatenate((y_hat, y_hat_t.T), axis=0)
    u = ut.T.ravel()
    h = ht.T.ravel()
    u = u.reshape(T * Nh, 1)
    h = h.reshape(T * Nh, 1)
    return u, ut, h, ht, y_hat

def GenVari(Ny, Nh, Nx, T, mean, stddev):  
    # Generate parameters A, W, V, b, c
    A = np.random.normal(loc=mean, scale=stddev, size=(Ny, Nh))
    W = np.random.normal(mean, stddev, size=(Nh, Nh))
    V = np.random.normal(mean, stddev, size=(Nh, Nx))
    b = np.random.normal(mean, stddev, size=(Nh, 1))
    c = np.random.normal(mean, stddev, size=(Ny, 1))
    return A, W, V, b, c

def GenInput(Nx, T):  
    # Generate input data
    x = np.random.uniform(low=-1, high=1, size=Nx*T).reshape(T, Nx)
    return x

def GenOutput(x_dataset, A_true, W_true, V_true, b_true, c_true, T, Nx, Ny, Nh, e_mean, e_stddev):
    _, _, _, _, y_dataset = CalculY(x_dataset, A_true, W_true, V_true, b_true, c_true, T, Nx, Ny, Nh)
    y_dataset = y_dataset + np.random.normal(e_mean, e_stddev, size=Ny*T).reshape(T, Ny)
    return y_dataset

def Standardization_Gen(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


#%% The function to generate synthetic datasets
def generate_synthetic_dataset(Nh, Nx, Ny, T_total, mean_true, stddev_true, e_mean=0, e_stddev=1e-3, seed=123456, standardize=True):
    """
    Function to generate and save a synthetic dataset.

    Parameters:
    Nh (int): Number of hidden units.
    Nx (int): Dimensionality of the input data.
    Ny (int): Dimensionality of the output data.
    T_total (int): Total time series length.
    mean_true (float): Mean for generating parameters A, W, V, b, c.
    stddev_true (float): Standard deviation for generating parameters A, W, V, b, c.
    e_mean (float): Mean of the noise, default is 0.
    e_stddev (float): Standard deviation of the noise, default is 1e-3.
    seed (int): Random seed for reproducibility, default is 123456.
    standardize (bool): Whether to standardize the data, default is True.

    Returns:
    dataset (numpy.ndarray): The generated dataset, including input and output data.
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Generate true weights and biases
    A_true, W_true, V_true, b_true, c_true = GenVari(Ny, Nh, Nx, T_total, mean_true, stddev_true)
    
    # Save the generated true weights and biases
    np.savetxt(f"A_true_mean{mean_true}std{stddev_true}_T{T_total}.txt", A_true)
    np.savetxt(f"W_true_mean{mean_true}std{stddev_true}_T{T_total}.txt", W_true)
    np.savetxt(f"V_true_mean{mean_true}std{stddev_true}_T{T_total}.txt", V_true)
    np.savetxt(f"b_true_mean{mean_true}std{stddev_true}_T{T_total}.txt", b_true)
    np.savetxt(f"c_true_mean{mean_true}std{stddev_true}_T{T_total}.txt", c_true)

    # Generate input dataset
    x_dataset = GenInput(Nx, T_total)
    
    # Generate output dataset
    y_dataset = GenOutput(x_dataset, A_true, W_true, V_true, b_true, c_true, T_total, Nx, Ny, Nh, e_mean, e_stddev)
    
    # Standardize the dataset (optional)
    if standardize:
        y_dataset = Standardization_Gen(y_dataset)
    
    # Combine input and output data into the final dataset
    dataset = np.concatenate((x_dataset, y_dataset), axis=1)
    dataset_df = pd.DataFrame(dataset)
    # Save the final dataset
    filename = f"SynDataset_Nh{Nh}_Nx{Nx}_Ny{Ny}_T{T_total}.csv"
    dataset_df.to_csv(filename, index=False, header=False)
    return