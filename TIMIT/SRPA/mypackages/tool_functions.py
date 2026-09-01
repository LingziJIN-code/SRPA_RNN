#mypackages/tool_functions
import numpy as np


#%% Functions as tools
def get_dim(var):
    dim = 1
    a = var.shape
    for i in range(len(a)):
        dim = dim * a[i]
    return dim


def vectorwise(x):
    dim = get_dim(x[0])
    res = x[0].reshape([dim, 1], order="F") #一般情况下默认order=C,就是按行切再相连。这里F是按列切，连成向量了
    for i in range(len(x) - 1):
        dim = get_dim(x[i + 1])
        res = np.concatenate((res, x[i + 1].reshape([dim, 1], order="F")),
                             axis=0)
    return res


def variableNorm(*variables):
    total_norm = 0
    for item in variables:
        total_norm += np.linalg.norm(item)
    return total_norm


def ReLU(item):
    return np.maximum(item, 0)


def ELU(u):
    # Calculate the function value for ELU
    # Input: vector
    result_ELU = np.where(u >= 0, u, np.exp(u) - 1)
    return result_ELU


def ELU_grad(u):
    # Calculate the gradient for ELU
    # Input: vector
    result_ELU_grad = np.where(u >= 0, 1, np.exp(u))
    return result_ELU_grad


def leaky_ReLU(u):
    result_leaky_ReLU = np.where(u >= 0, u, 0.01 * u)
    return result_leaky_ReLU


def grad_leaky_ReLU(u):
    result_leaky_ReLU = np.where(u >= 0, 1, 0.01)
    return result_leaky_ReLU
