# mypackages/grad_AL
import numpy as np
import mypackages.functions_calculate_ALfunc as ALfunc
import mypackages.tool_functions as toolfunc


#%% Functions to calculate gradient of AL function respect to variable \bf{s}
def GradALM_h(W, V, b, A, c, z_in, z_out, u, ut, h, ht, x_dataset, y_dataset,
              xi1, xi2, gamma1, gamma2, T, Ny, Nx, Nh):
    # This function is to calculate the gradient for lagrange function L_gamma1(z, h, u) of h
    # Calculate the gradient of g1
    grad_g1 = []
    phi_all = ALfunc.PhiX(ht, z_out, Nh, Ny, T)
    for t in range(T):
        grad_g1_t = (2 / T) * np.transpose(A) @ (y_dataset[t] - phi_all[t, :])
        if t == 0:
            grad_g1 = grad_g1_t
        else:
            grad_g1 = np.concatenate((grad_g1, grad_g1_t), axis=0)
    grad_g1 = grad_g1.reshape(grad_g1.shape[0], 1)

    # Calculate the gradient of g4
    grad_g4 = []
    for t in range(1, T):
        grad_g4_t = np.dot(W.T, xi1[t * Nh:(t + 1) * Nh])
        if t == 1:
            grad_g4 = grad_g4_t
        else:
            grad_g4 = np.concatenate((grad_g4, grad_g4_t), axis=0)
    grad_g4 = np.concatenate((grad_g4, np.zeros((Nh, 1))), axis=0)
    # Calculate the gradient of g6
    grad_g6 = []
    for t in range(1, T):
        grad_g6_t = gamma1 * W.T @ (
            ut[:, [t]] - W @ ht[:, [t - 1]] -
            V @ x_dataset[t].reshape([Nx, 1], order="F") - b)
        if t == 1:
            grad_g6 = grad_g6_t
        else:
            grad_g6 = np.concatenate((grad_g6, grad_g6_t), axis=0)
    grad_g6 = np.concatenate((grad_g6, np.zeros((Nh, 1))), axis=0)
    grad_g8 = xi2
    grad_g9 = gamma2 * (h - toolfunc.ReLU(u))
    grad_lagr_h = -grad_g1 - grad_g4 - grad_g6 + grad_g8 + grad_g9
    return grad_lagr_h


def GradALM_u(W, V, b, u, ut, h, ht, z_in, x_dataset, xi1, xi2, gamma1, gamma2,
              T, Nh, Nx):
    # This function is to calculate the gradient for lagrange function L_gamma1(z, h, u) of u
    # Calculate the gradient of g7
    grad_g7 = gamma1 * (u - ALfunc.PsiX(ht, x_dataset, z_in, Nx, Nh, T)) + xi1
    grad_gu = np.empty((Nh * T, 1))
    for tt in range(T):
        grad_gu[tt * Nh:(tt + 1) * Nh, :] = (np.dot(
            np.diag(ALfunc.ELU_grad(ut[:, tt])),
            (ht[:, tt] - ALfunc.ELU(ut[:, tt])).reshape(Nh, 1) +
            gamma1 * xi1[tt * Nh:(tt + 1) * Nh, :])).reshape(Nh, 1)
    return grad_g7 + gamma1 * grad_gu


#z_out=[A,c]
def GradALM_zout(A, c, z_out, h, ht, y_dataset, lambda1, lambda5, T, Ny, Nx,
                 Nh):
    # This function is to calculate the gradient for lagrange function L_gamma1(z, h, u) of zout
    # Calculate the gradient of g8
    y_vector = y_dataset - ALfunc.PhiX(ht, z_out, Nh, Ny, T)
    grad_g8 = (-2 / T) * ALfunc.PhiTy(ht, y_vector, Nh, Ny, T)
    # Calculate the gradient of g9
    grad_g9 = np.dot(
        np.diag(
            np.concatenate(
                (2 * lambda1 * np.ones(Ny * Nh), 2 * lambda5 * np.ones(Ny)),
                0)), z_out)
    grad_lagr_zout = grad_g8 + grad_g9
    return grad_lagr_zout


# z_in=[W,V,b]
def GradALM_zin(W, V, b, z_in, u, h, ht, x_dataset, xi1, gamma1, lambda2,
                lambda3, lambda4, T, Ny, Nx, Nh):
    # This function is to calculate the gradient for lagrange function L_gamma1(z, h, u) of zin
    # Calculate the gradient of g11
    grad_g11 = 2 * np.dot(
        np.diag(
            np.concatenate(
                (lambda2 * np.ones(Nh * Nh), lambda3 * np.ones(Nx * Nh),
                 lambda4 * np.ones(Nh)), 0)), z_in)
    # Calculate the gradient of g12
    psidouble = ALfunc.PsiDouble(ht, x_dataset, Nh, T)
    grad_g12 = gamma1 * np.dot(psidouble, z_in) - ALfunc.PsiTx(
        ht, x_dataset, xi1, Nx, Nh,
        T) - gamma1 * ALfunc.PsiTx(ht, x_dataset, u, Nx, Nh, T)
    grad_lagr_zin = grad_g11 + grad_g12
    return grad_lagr_zin
