from SPLQP import SPLQP_ReLU_optimization
from gen_synthetic_dataset import generate_synthetic_dataset
from dependency_packages import np, random

#%% Import dataset
# Import synthetic datasets
Nh = 4 #r,
Nx = 5 #n
Ny = 3 #m
T_total = 10  
generate_synthetic_dataset(
    Nh, Nx, Ny, T_total,
    mean_true=0, stddev_true=0.8,
    e_mean=0, e_stddev=1e-3,
    seed=123456, standardize=True
)
dataset_name = f"./SynDataset_Nh{Nh}_Nx{Nx}_Ny{Ny}_T{T_total}"

 


#%% Use SRPA(SPLQP) to train RNNs  
tau=1.2
np.random.seed(123456)
random.seed(123456)
results = SPLQP_ReLU_optimization(dataset_name=dataset_name,
                                    Ny=Ny,
                                    Nh=Nh,
                                    maxiter=100,  
                                    distribution_type="Gaussian",
                                    mean=0,
                                    std=1e-1,
                                    beta1=1,
                                    beta2=1,
                                    lambda1 = tau / (Nh * Ny),   # regularization parameters
                                    lambda2 = tau / (Nh * Nh), 
                                    lambda3 = tau / (Nx * Nh), 
                                    lambda4 = tau / Nh, 
                                    lambda5 = tau / Ny, 
                                    eta1=0.9,
                                    eta2=1.3, 
                                    rho1=0.5, 
                                    print_option=2)

 
 