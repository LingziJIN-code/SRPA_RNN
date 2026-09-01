#mypackages/BCD_closed form
import numpy as np
import mypackages.functions_calculate_ALfunc as ALfunc
import mypackages.tool_functions as toolfunc


#%% Functions to calculate closed-form solutions of BCD
def updateZin_closed_blk(u, ht, x_dataset, xi1, gamma1, lambda2, lambda3,
                         lambda4, Nx, Nh, T, row_i):
    psi = ALfunc.Psi_blk(ht, x_dataset, Nh, T)
    psi_T = psi.T
    index = np.arange(row_i, T * Nh + row_i, Nh)
    P = 2 * np.diag(
        np.concatenate((lambda2 * np.ones(Nh), lambda3 * np.ones(Nx),
                        lambda4 * np.ones(1)))) + gamma1 * np.dot(psi_T, psi)
    g = np.dot(psi_T, xi1[index]) + gamma1 * np.dot(psi_T, u[index])
    try:
        L = np.linalg.cholesky(P)
        y = np.linalg.solve(L, g)
        zin_kj_blk = np.linalg.solve(L.T, y)
        W_kj_blk = zin_kj_blk[:Nh, :].T
        V_kj_blk = zin_kj_blk[Nh:Nh + Nx, :].T
        b_kj_blk = zin_kj_blk[Nh + Nx:2 * Nh + Nx, :].T
        return W_kj_blk, V_kj_blk, b_kj_blk
    except:
        print(
            'Warning: P is not symmetric positive definite. zin cannot be solved by Cholesky'
        )


def updateZin_closed(W, V, b, u, ht, x_dataset, xi1, gamma1, lambda2, lambda3,
                     lambda4, Nx, Nh, T):
    W = np.zeros([Nh, Nh])
    V = np.zeros([Nh, Nx])
    b = np.zeros([Nh, 1])
    for row_i in range(Nh):
        W[row_i, :], V[row_i, :], b[row_i, :] = updateZin_closed_blk(
            u, ht, x_dataset, xi1, gamma1, lambda2, lambda3, lambda4, Nx, Nh,
            T, row_i)
    zin = toolfunc.vectorwise([W, V, b])
    return zin, W, V, b


def updateZout_closed(ht, y_dataset, lambda1, lambda5, Nh, Ny, T):
    Phisum = []
    Phiysum = []
    phi_blk = np.concatenate((ht, np.ones([1, T])))
    Phisum = np.kron(np.dot(phi_blk, phi_blk.T), np.identity(Ny))
    Phiysum = ALfunc.PhiTy(ht, y_dataset, Nh, Ny, T)
    Q = (2 / T) * Phisum + 2 * np.diag(
        np.hstack((lambda1 * np.ones(Nh * Ny), lambda5 * np.ones(Ny))))
    f = (2 / T) * Phiysum
    # Method1.1: If Q is not positive definite
    # %   zout_kj=pinv(Q)*f;
    # %   zout_k=zout_kj;
    # Method1.2: If Q is positive definite
    try:
        L = np.linalg.cholesky(Q)
        y = np.linalg.solve(L, f)
        zout_kj = np.linalg.solve(L.T, y)
        return zout_kj
    except:
        # dbstop
        return 'Warning: Q is not symmetric positive definite. zout cannot be solved by Cholesky'


def updateH_closed(A_kj, W_kj, V_kj, b_kj, c_kj, ut_kj, x_trainset, y_trainset,
                   gamma1, gamma2, xi1, xi2, Nx, Nh, Ny, T):
    D1 = gamma1 * W_kj.T @ W_kj + (
        2 / T) * A_kj.T @ A_kj + gamma2 * np.identity(Nh)
    D2 = (2 / T) * A_kj.T @ A_kj + gamma2 * np.identity(Nh)
    D1_inv = np.linalg.inv(D1)
    D2_inv = np.linalg.inv(D2)
    xit1 = xi1.reshape([Nh, T], order="F")
    xit2 = xi2.reshape([Nh, T], order="F")
    ht_ing = np.zeros((Nh, T))  # to record the calculation results
    for t in range(0, T - 1):
        b1 = gamma2 * toolfunc.ReLU(
            ut_kj[:, [t]]
        ) - xit2[:, [t]] + W_kj.T @ xit1[:, [t + 1]] + gamma1 * W_kj.T @ (
            ut_kj[:, [t + 1]] - V_kj @ x_trainset[t + 1].reshape(
                [Nx, 1], order="F") - b_kj) + (2 / T) * A_kj.T @ (
                    y_trainset[t].reshape([Ny, 1], order="F") - c_kj)
        ht_ing[:, [t]] = D1_inv @ b1
    b2 = gamma2 * toolfunc.ReLU(
        ut_kj[:, [T - 1]]) - xit2[:, [T - 1]] + (2 / T) * A_kj.T @ (
            y_trainset[T - 1].reshape([Ny, 1], order="F") - c_kj)
    ht_ing[:, [T - 1]] = D2_inv @ b2
    ht_kj = ht_ing
    return ht_kj


def ValueLi_u_vec(u, u_last, h, cons_a, gamma1, gamma2, beta1, beta2, sigma,
                  lambda6):
    L = 0.5 * gamma1 * (u - cons_a + beta1)**2 + 0.5 * gamma2 * (
        h - toolfunc.ReLU(u) +
        beta2)**2 + 0.5 * sigma * (u - u_last)**2 + lambda6 * u**2
    return L


def updateU_closed(z_in, z_out, A, W, V, b, h, ht, u_last, xi1, xi2, gamma1,
                   gamma2, lambda1, lambda2, lambda3, lambda4, lambda6, sigma,
                   x_dataset, y_dataset, Nh, Nx, Ny, T):
    # compute constant k
    beta1 = xi1 / gamma1
    beta2 = xi2 / gamma2
    cons_a = ALfunc.PsiX(ht, x_dataset, z_in, Nx, Nh, T)
    # case1: u > 0
    u_posi = (gamma1 * cons_a - xi1 + gamma1 * h + xi2 +
              sigma * u_last) * (1 / (2 * gamma1 + 2 * lambda6 + sigma))
    index_posi = u_posi <= 0
    u_posi[index_posi] = 0
    # case2: u <= 0
    u_neg = (gamma1 * cons_a - xi1 +
             sigma * u_last) * (1 / (gamma1 + 2 * lambda6 + sigma))
    index_neg = u_neg >= 0
    u_neg[index_neg] = 0
    # compare function value of case1 and case2
    L_posi = ValueLi_u_vec(u_posi, u_last, h, cons_a, gamma1, gamma2, beta1,
                           beta2, sigma, lambda6)
    L_neg = ValueLi_u_vec(u_neg, u_last, h, cons_a, gamma1, gamma2, beta1,
                          beta2, sigma, lambda6)
    index_L = L_posi > L_neg
    u_posi[index_L] = u_neg[index_L]
    u_star = u_posi
    return u_star
