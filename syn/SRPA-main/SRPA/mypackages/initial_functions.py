#mypackages/initial_func
import numpy as np
import mypackages.tool_functions as toolfunc


#%% Funtions for initialization
def iniVariable(Ny, Nh, Nx, T, mean, stddev, method_name):
    # This function is for variables initialization using He initialization
    if method_name == "He":
        A = np.random.normal(0, np.sqrt(2 / Nh), size=(Ny, Nh))
        W = np.random.normal(0, np.sqrt(2 / Nh), size=(Nh, Nh))
        V = np.random.normal(0, np.sqrt(2 / Nx), size=(Nh, Nx))
        b = np.zeros((Nh, 1))
        c = np.zeros((Ny, 1))

    elif method_name == "Gaussian":
        A = np.random.normal(mean, stddev, size=(Ny, Nh))
        W = np.random.normal(mean, stddev, size=(Nh, Nh))
        V = np.random.normal(mean, stddev, size=(Nh, Nx))
        b = np.zeros((Nh, 1))
        c = np.zeros((Ny, 1))

    elif method_name == "Glorot":
        stddev = np.sqrt(2 / (Nh + Ny))
        A = np.random.normal(0, stddev, size=(Ny, Nh))

        stddev = np.sqrt(2 / (Nh + Nh))
        W = np.random.normal(0, stddev, size=(Nh, Nh))

        stddev = np.sqrt(2 / (Nx + Nh))
        V = np.random.normal(0, stddev, size=(Nh, Nx))

        b = np.zeros((Nh, 1))
        c = np.zeros((Ny, 1))

    elif method_name == "LeCun":
        A = np.random.normal(0, np.sqrt(1 / Nh), size=(Ny, Nh))
        W = np.random.normal(0, np.sqrt(1 / Nh), size=(Nh, Nh))
        V = np.random.normal(0, np.sqrt(1 / Nx), size=(Nh, Nx))
        b = np.zeros((Nh, 1))
        c = np.zeros((Ny, 1))

    z_in = toolfunc.vectorwise([W, V, b])
    z_out = toolfunc.vectorwise([A, c])
    return A, W, V, b, c, z_in, z_out
