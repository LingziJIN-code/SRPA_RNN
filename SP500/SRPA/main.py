from SPLQP import SPLQP_ReLU_optimization  
from dependency_packages import np, random

#%%   Import SP500 datasets
Nh = 20 #r 
Nx = 11 #n
Ny = 1 #m
T_total = 437  
dataset_name = "clean_SP500"


#%% Use ALM_BCD to train RNNs when setting activation function as ReLU
tau=1
np.random.seed(123456)
random.seed(123456) 
results = SPLQP_ReLU_optimization(dataset_name=dataset_name,
                                    Ny=Ny,
                                    Nh=Nh,
                                    maxiter=100,  
                                    distribution_type="Glorot",
                                    mean=0,
                                    std=20,
                                    beta1=0.1,
                                    beta2=0.1,
                                    lambda1 = tau / (Nh * Ny),   # regularization parameters
                                    lambda2 = tau / (Nh * Nh), 
                                    lambda3 = tau / (Nx * Nh), 
                                    lambda4 = tau / Nh, 
                                    lambda5 = tau / Ny, 
                                    eta1=0.7,
                                    eta2=1.1, 
                                    rho1=0.03, 
                                    print_option=2)

 