#mypackages/functions_calculate_ALfunc
import numpy as np
import mypackages.tool_functions as toolfunc


#%% Functions to calculate function values
def PhiX(ht, x_vector, Nh, Ny, T):
    # This function is to calculate Phi*vector
    phi_blk = np.concatenate((ht.T, np.ones([T, 1])), axis=1)
    y_hat = np.dot(phi_blk, x_vector.reshape([Nh + 1, Ny]))
    return y_hat


def PhiTy(ht, y_vector, Nh, Ny, T):
    # This function is to calculate Phi^T*vector
    phiT_blk = np.concatenate((ht, np.ones([1, T])))
    out = np.dot(phiT_blk, y_vector).reshape([Ny * Nh + Ny, 1])
    return out


def Psi_blk(ht, x_dataset, Nh, T):
    psi = np.concatenate((np.concatenate((np.zeros(
        (1, Nh)), ht[:, :T - 1].T)), x_dataset, np.ones((T, 1))),
                         axis=1)
    return psi


def PsiX(ht, x_dataset, x_vector, Nx, Nh, T):
    # This function is to calculate Psi*vector
    out = np.empty([T, Nh])
    psi = Psi_blk(ht, x_dataset, Nh, T)
    for row_i in range(Nh):
        index_i = np.concatenate(
            (np.arange(row_i, Nh**2 + row_i, Nh),
             np.arange(Nh**2 + row_i, Nh**2 + Nx * Nh + row_i, Nh),
             np.arange(row_i + Nh**2 + Nx * Nh, row_i + Nh**2 + Nx * Nh + Nh,
                       Nh)))
        out[..., row_i] = np.dot(psi, x_vector[index_i]).T
    psix = out.reshape([T * Nh, 1])
    return psix


def PsiTx(ht, x_dataset, x_vector, Nx, Nh, T):
    # This function is to calculate Psi^T*vector
    out = np.empty([Nh + Nx + 1, Nh])
    for row_i in range(Nh):
        index_i = np.arange(row_i, T * Nh + row_i, Nh)
        out[..., row_i] = np.dot(
            Psi_blk(ht, x_dataset, Nh, T).T, x_vector[index_i]).T
    psix = out.reshape([Nh * (Nh + Nx + 1), 1])
    return psix


def PsiDouble(ht, x_dataset, Nh, T):
    # This function is to calculate Psi^T*Psi
    blk1 = np.kron(np.concatenate((np.zeros((Nh, 1)), ht[:, :T - 1]), axis=1),
                   np.identity(Nh))
    blk2 = np.kron(x_dataset.T, np.identity(Nh))
    blk3 = np.tile(np.identity(Nh), (1, T))
    ele1 = np.concatenate(
        (np.dot(blk1, blk1.T), np.dot(blk1, blk2.T), np.dot(blk1, blk3.T)),
        axis=1)
    ele2 = np.concatenate(
        (np.dot(blk2, blk1.T), np.dot(blk2, blk2.T), np.dot(blk2, blk3.T)),
        axis=1)
    ele3 = np.concatenate(
        (np.dot(blk3, blk1.T), np.dot(blk3, blk2.T), np.dot(blk3, blk3.T)),
        axis=1)
    psix = np.concatenate((ele1, ele2, ele3), axis=0)
    return psix


def CalculY(x_dataset, A, W, V, b, c, T, Nh, Nx, Ny):
    # This function is used to calculate y^hat by A, W, V, b, c recurrently nor auxi1liary variables h and u
    for t in range(T):
        x_t = x_dataset[t, :].reshape(Nx, 1)
        if t == 0:
            ut_t = V @ x_t + b
            ut = ut_t
            ht_t = toolfunc.ReLU(ut_t)
            ht = ht_t
            y_hat_t = A @ ht_t + c
            y_hat = y_hat_t.T
        else:
            ut_t = W @ ht_t + V @ x_t + b
            ut = np.concatenate((ut, ut_t), axis=1)
            ht_t = toolfunc.ReLU(ut_t)
            ht = np.concatenate((ht, ht_t), axis=1)
            y_hat_t = A @ ht_t + c
            y_hat = np.concatenate((y_hat, y_hat_t.T), axis=0)
    u = ut.T.ravel()
    h = ht.T.ravel()
    u = u.reshape(T * Nh, 1)
    h = h.reshape(T * Nh, 1)
    return u, ut, h, ht, y_hat

def Calculauxi(x_dataset, A, W, V, b, c, T, Nh, Nx, Ny):
    # This function is used to calculate y^hat by A, W, V, b, c recurrently nor auxi1liary variables h, u and v
    for t in range(T):
        x_t = x_dataset[t, :].reshape(Nx, 1)
        if t == 0:
            ut_t = V @ x_t + b
            ut = ut_t
            ht_t = toolfunc.ReLU(ut_t)
            ht = ht_t
            y_hat_t = A @ ht_t + c
            y_hat = y_hat_t.T
        else:
            ut_t = W @ ht_t + V @ x_t + b
            ut = np.concatenate((ut, ut_t), axis=1)
            ht_t = toolfunc.ReLU(ut_t)
            ht = np.concatenate((ht, ht_t), axis=1)
            y_hat_t = A @ ht_t + c
            y_hat = np.concatenate((y_hat, y_hat_t.T), axis=0)
    u = ut.T.ravel()
    h = ht.T.ravel()
    u = u.reshape(T * Nh, 1)
    h = h.reshape(T * Nh, 1)
    v = y_hat.ravel().reshape(-1,1)
    vt = y_hat.T
    return u, ut, h, ht, v, vt, y_hat


def CalculY_auxi(z_out, ht, Nh, Ny, T):
    # This function is used to calculate y^hat by V, c, h 
    y_hat = PhiX(ht, z_out, Nh, Ny, T)
    return y_hat


# Below functions 'Valueloss', 'ValuePenalty', 'ValueRegula', 'ValueRegula' are
# used to calculate the value of argumented Lagrangian function
def ValueLoss(y_dataset, y_hat, Ny, T):
    # This function is used to calculate the function value of loss function
    # L(z,h) = 1/T sum_t(||y_t-v_t||^2)
    y_hat_flat = y_hat.ravel().reshape(T * Ny, 1)
    y_dataset_flat = y_dataset.ravel().reshape(T * Ny, 1)
    return (1 / T) * np.sum((y_hat_flat - y_dataset_flat)**2)

def ValueLoss1(y_dataset, v, Ny, T):
    # This function is used to calculate the function value of loss function
    # L(z,h) = 1/T sum_t(||y_t-v_t||^2)
    y_dataset_flat = y_dataset.ravel().reshape(T * Ny, 1)
    return (1 / T) * np.sum((v - y_dataset_flat)**2)


def ValueRegula(A, W, V, b, c, u, lambda1, lambda2, lambda3, lambda4, lambda5,
                lambda6):
    # This function is used to calculate the regulation term in AL function
    reg1 = lambda1 * np.sum(A**2) + lambda2 * np.sum(W**2) + lambda3 * np.sum(
        V**2)
    reg2 = lambda4 * np.sum(b**2) + lambda5 * np.sum(c**2)
    reg3 = lambda6 * np.sum(u**2)
    Regula = reg1 + reg2 + reg3
    return Regula


def ValueALMterm(x_dataset, ht, h, u, z_in, xi1, xi2, gamma1, gamma2, Nx, Nh,
                 T):
    # This function is used to calculate the penalty term for AL function
    feasi_uhz = u - PsiX(ht, x_dataset, z_in, Nx, Nh, T)
    feasi_hu = h - toolfunc.ReLU(u)
    valueALMterm = np.dot(xi1.T, feasi_uhz) + np.dot(
        xi2.T, feasi_hu) + (gamma1 * 0.5) * np.sum(
            (feasi_uhz.reshape(T * Nh, 1))**2) + 0.5 * gamma2 * np.sum(
                (feasi_hu.reshape(T * Nh, 1))**2)  # 修改了计算错误
    return valueALMterm

def Valuel1term(x_dataset, ht, h, u, v, z_in, z_out, beta1, beta2, Nx, Nh, Ny, T):
    # This function is used to calculate the l1-penalty term for Theta
    feasi_uhz = u - PsiX(ht, x_dataset,     
                         z_in, Nx, Nh, T)
    feasi_hu = h - toolfunc.ReLU(u)
    feasi_vhz = v- CalculY_auxi(z_out, ht, Nh, Ny, T).ravel().reshape(-1,1)
    valuel1term = beta1 * ( np.sum(
            np.abs(feasi_uhz)) 
            + np.sum(np.abs(feasi_hu)) ) + beta2 * np.sum(np.abs(feasi_vhz))  
    return valuel1term


def LossALM(x_dataset, y_dataset, A, W, V, b, c, ht, h, u, z_in, z_out, xi1,
            xi2, gamma1, gamma2, lambda1, lambda2, lambda3, lambda4, lambda5,
            lambda6, Nx, Nh, Ny, T):
    # This function calculate the function value of argumented Lagrangian function
    y_auxi = CalculY_auxi(z_out, ht, Nh, Ny, T)
    valueloss = ValueLoss(y_dataset, y_auxi, Ny, T)
    valueRegula = ValueRegula(A, W, V, b, c, u, lambda1, lambda2, lambda3,
                              lambda4, lambda5, lambda6)
    valueALMterm = ValueALMterm(x_dataset, ht, h, u, z_in, xi1, xi2, gamma1,
                                gamma2, Nx, Nh, T)
    loss_ALM = valueloss + valueRegula + valueALMterm
    return loss_ALM
