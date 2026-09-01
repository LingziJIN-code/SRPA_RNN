from SPLQP_N import SPLQP_N_ReLU_optimization 
from dependency_packages import np, random

#%% dimension of RNN

Nx = 129
Ny = 129 
Nh = 2



#%% Use SPLQP to train RNNs when setting activation function as ReLU
tau=5e-4
np.random.seed(1234)
random.seed(1234) 
results = SPLQP_N_ReLU_optimization(Nh=Nh,
                                    maxiter=600,  
                                    distribution_type="Gaussian",
                                    mean=0,
                                    std=1e-2,
                                    beta1=0.04,
                                    beta2=0.04,
                                    lambda1 = tau / (Nh * Ny),   # regularization parameters
                                    lambda2 = tau / (Nh * Nh), 
                                    lambda3 = tau / (Nx * Nh), 
                                    lambda4 = tau / Nh, 
                                    lambda5 = tau / Ny, 
                                    eta1=0.7,
                                    eta2=1.1, 
                                    rho1=0.003, 
                                    print_option=2)
  
 