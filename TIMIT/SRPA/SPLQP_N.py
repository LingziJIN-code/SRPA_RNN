# Import and install dependency packages
from dependency_packages import np, pd, scipy, copy, time, random, math, gc, cp 
import pickle


# Import mypackages
import mypackages.tool_functions as toolfunc
import mypackages.initial_functions as inifunc
import mypackages.functions_calculate_ALfunc as ALfunc 



 
def load_split_datasetN(sizerate_training=0.9):
    """
    根据给定的比例分割三维数据列表。
    
    参数:
    sizerate_training : float
        训练集比例 (0.0 - 1.0)。
        
    返回:
    x_trainset, y_trainset, x_testset, y_testset : numpy.ndarray
        分割后的训练集和测试集特征与标签。
    N_train, N_test, T, Nx, Ny : int
        训练集样本量、测试集样本量、时刻数、以及input维度和label维度。
    """
    # 0. 读取pkl数据文件 
    with open('mixed_fre_timit_mini.pkl', 'rb') as f:
        mixed = pickle.load(f)
    with open('ori_fre_timit_mini.pkl', 'rb') as f:
        ori = pickle.load(f)

    # 1. 获取维度 
    # mixed为(N,Nx,T), ori为(N,Ny,T) 
    mixed_arr = np.array(mixed)
    ori_arr = np.array(ori)
    N, Nx, T = mixed_arr.shape
    _, Ny, _ = ori_arr.shape 
    
    # 2. 计算分割点
    N_train = int(sizerate_training * N) 
    N_test = N- N_train
     
    
    # 4. 切片 
    # x_trainset: mixed 的前 N_train 个样本, (N_train, T, Nx)
    x_trainset = mixed_arr[:N_train].transpose(0, 2, 1)
    # x_testset: mixed 的剩余样本, (N_test, T, Nx)
    x_testset = mixed_arr[N_train:].transpose(0, 2, 1)
    
    # y_trainset: ori 的前 N_train 个样本, (N_train, T, Ny)
    y_trainset = ori_arr[:N_train].transpose(0, 2, 1)
    # y_testset: ori 的剩余样本, (N_test, T, Ny)
    y_testset = ori_arr[N_train:].transpose(0, 2, 1)

    return x_trainset, y_trainset, x_testset, y_testset, N_train, N_test, T, Nx, Ny
 



def initialize_SPLQP_params(Nh,
                          Ny,
                          Nx,
                          T,
                          beta1,
                          beta2,
                          lambda1,
                          eta1=0.99,
                          eta2=1.2,
                          rho1=None):
    """
    Initialize parameters for the SPLQP algorithm.

    Parameters:
    - Nh: Number of hidden units
    - Ny: Number of output units
    - Nx: Number of input units
    - T: Size of the training set
    - beta1: penalty parameter for hidden layer (default: 1)
    - beta2: penalty parameter for output layer (default: 1)
    - lambda1/2/3/4/5: Regularization parameters
    - eta1: in (0,1), parameter for measuring the sufficient descent (default: 0.99)
    - eta2: >1, update parameter for rho (default: 1.2)
    - rho1: initial value for rho, the regularization parameter

    Returns:
    - A dictionary containing the initialized parameters.
    """

    # Calculate default values if not provided
    if rho1 is None:
        # rho1 = ( max(2*lambda1,2/T) + pow(T*(Nh+Ny),0.5) * pow( max(beta1*beta1*Nh+beta2*beta2*Ny, T ),0.5) ) / (1-eta1)   
        rho1=0.24 
        print("rho_1", rho1)  
     
 

    return {
        "eta1": eta1,
        "eta2": eta2,
        "rho1":rho1
    }   #注意，返回的是个字典


def initialize_variablesN(Ny,
                         Nh,
                         Nx,
                         T,
                         N_train,
                         N_test,
                         distribution_type="Gaussian",
                         mean=None,
                         std=None,
                         x_trainset=None,
                         y_trainset=None,
                         x_testset=None,
                         y_testset=None,
                         beta1=None,
                         beta2=None,
                         lambda1=None,
                         lambda2=None,
                         lambda3=None,
                         lambda4=None,
                         lambda5=None,):
    """
    Initialize variables for the SPLQP algorithm.

    Parameters:
    - Ny: Number of output units
    - Nh: Number of hidden units
    - Nx: Number of input units
    - T: Number of time steps
    - N_train: Size of the training set
    - N_test: Size of the test set
    - distribution_type: Type of distribution for initialization ("Gaussian" or other, default: "Gaussian")
    - mean: Mean for the distribution (default: 0 for Gaussian, 0 for other)
    - std: Standard deviation for the distribution (default: 1e-3 for Gaussian, 20 for other)
    - x_trainset: Training set inputs
    - y_trainset: Training set outputs 
    - x_testset: Test set inputs
    - y_testset: Test set outputs
    - beta1: penalty parameter for hidden layer (default: 1)
    - beta2: penalty parameter for output layer (default: 1)
    - lambda1,2,3,4,5: Regularization parameters

    Returns:
    - A dictionary containing initialized variables.
    """

    # Set defaults based on distribution type
    if distribution_type == "Gaussian":
        if mean is None:
            mean = 0
        if std is None:
            std = 1e-3
    else:
        if mean is None:
            mean = 0
        if std is None:
            std = 20
    # np.random.seed(123456)
    # random.seed(123456)

    # Initialize variables using the specified distribution
    # 随机生成A0, W0, V0, b0, c0后, feedforward来生成h0,u0,v0, 所以生成的z0是可行点
    # y_hat0是feedforward得到的vt0^T,与y_trainset同维度 
    # hthetaout0就是v=hz约束的右端值的T*Ny矩阵，由于此时v0是feedfoward得到的，所以其实是=vt0^T=y_hat0
    A0, W0, V0, b0, c0, thetain_0, thetaout_0 = inifunc.iniVariable(
        Ny, Nh, Nx, T, mean, std, distribution_type)
    # hti0 = np.empty((N_train, Nh, T), dtype=np.float32) #32位浮点数节省内存
    hti0 = np.empty((N_train, Nh, T))
    y_hat0 = np.empty((N_train, T, Ny))
    u0 = np.empty((N_train*T*Nh,1))
    h0 = np.empty((N_train*T*Nh,1))
    v0 = np.empty((N_train*T*Ny,1))
    hthetaout0 = np.empty((N_train,T,Ny))
    hthetain0 = np.empty((N_train,T,Nh))
    for i in range(N_train):
        u0[i*T*Nh:(i+1)*T*Nh], _, h0[i*T*Nh:(i+1)*T*Nh], hti0[i,:,:], v0[i*T*Ny:(i+1)*T*Ny], _, y_hat0[i,:,:] = ALfunc.Calculauxi(x_trainset[i], A0, W0, V0, b0, c0, T, Nh, Nx, Ny) 
        # 
        hthetaout0[i,:,:] = ALfunc.CalculY_auxi(thetaout_0, hti0[i,:,:], Nh, Ny, T) 
        hthetain0[i,:,:] = ALfunc.PsiX(hti0[i,:,:], x_trainset[i], thetain_0, Nx, Nh, T).reshape((T,Nh), order='C') 

    # Ensure all necessary parameters are provided
    if None in (lambda1, lambda2, lambda3, lambda4, lambda5):
        raise ValueError("The lambda parameter must be provided.")

    regvalue0 = ALfunc.ValueRegula(A0, W0, V0, b0, c0, u0, 
                                   lambda1=lambda1, 
                                   lambda2=lambda2,
                                   lambda3=lambda3, 
                                   lambda4=lambda4, 
                                   lambda5=lambda5, 
                                   lambda6=0)
    # print("regvalue0", regvalue0)
    valuel1term = beta1 * ( np.sum(
            np.abs(u0 - hthetain0.ravel().reshape(-1,1))) 
            + np.sum(np.abs(h0 - toolfunc.ReLU(u0))) ) + beta2 * np.sum(np.abs(v0- hthetaout0.ravel().reshape(-1,1))) 
    lossTheta0 = ALfunc.ValueLoss1N(y_trainset, v0, T, N_train) + regvalue0 + valuel1term

    # Feasibility Violation, ell_2 norm
    FeasVi_h0 = (1 /(N_train*T)) * np.sum((h0 - toolfunc.ReLU(u0))**2)**(1/2) 
    FeasVi_u0 = (1 /(N_train*T)) * np.sum(
        (u0 - hthetain0.ravel().reshape(-1,1) )**2)**(1/2)
    FeasVi_v0 = (1 /(N_train*T)) * np.sum((v0- hthetaout0.ravel().reshape(-1,1))**2)**(1/2)

    #Trainerror
    TrainErr0 = (1 /(N_train*T)) * np.sum((y_hat0 - y_trainset)**2)
    if np.isnan(TrainErr0) == True:
        TrainErr0 = 1e+30
    
    #Testerror
    # N>1，有多个序列，testerror的序列与trainerror序列独立时
    y_test_hat0 = np.empty((N_test, T, Ny))
    for i in range(N_test):
        _, _, _, _, y_test_hat0[i,:,:] = ALfunc.CalculY(x_testset[i], A0, W0, V0, b0, c0, T, Nh, Nx, Ny) 
    TestErr0 = (1 / (N_test*T)) * np.sum((y_test_hat0 - y_testset)**2)


    

    return {
        "A0": A0,
        "W0": W0,
        "V0": V0,
        "b0": b0,
        "c0": c0,
        "thetain_0": thetain_0,
        "thetaout_0": thetaout_0,
        "u0": u0, 
        "h0": h0,
        "hti0": hti0,
        "v0":v0,   
        "lossTheta0": lossTheta0,
        "feasvi_h0": FeasVi_h0,
        "feasvi_u0": FeasVi_u0,
        "feasvi_v0": FeasVi_v0,
        "TrainErr0": TrainErr0,
        "TestErr0": TestErr0
    }   #注意，返回的是个字典


def SPLQP_N_ReLU_optimization(Nh=20,
                              maxiter=10, 
                              distribution_type="Gaussian",
                              mean=None,
                              std=None,
                              beta1=1,
                              beta2=1,
                              lambda1=None,
                              lambda2=None,
                              lambda3=None,
                              lambda4=None,
                              lambda5=None,
                              eta1=0.99,
                              eta2=6 / 5, 
                              rho1=0.24, 
                              print_option=2):
    """
    Perform SPLQP to train RNNs.

    Parameters:
    - Nh: Number of hidden units (default: 20)
    - maxiter: Maximum number of iterations for the outer loop (default: 10)
    - distribution_type: Type of distribution for variable initialization ("Gaussian" "He", "Glorot", "LeCun", default: "Gaussian")
    - mean: Mean for the initialization distribution (default: 0)
    - std: Standard deviation for the initialization distribution (default: 1e-3 for Gaussian, 20 for other)
    - beta1: penalty parameter for hidden layer (default: 1)
    - beta2: penalty parameter for output layer (default: 1)
    - lambda1/2/3/4/5: Regularization parameters
    - eta1, eta2: Parameters for updating the algorithm
    - rho1: initial value for rho
    - print: Control the verbosity of the output (default: 0)
        - 0: No output
        - 1: Print final TrainErr, TestErr, FeasVi_h, FeasVi_u, FeasVi_v
        - 2: Print TrainErr, TestErr, FeasVi_h, FeasVi_u, FeasVi_v, Theta_value and time for each outer iteration

    Returns:
    - results: A dictionary containing lists for TrainErr, TestErr, time, FeasVi_h, and FeasVi_u
    """
    

    # Print the optimization problem being solved
    if print_option >= 1:
        print(
            "\nWe use a sequential regularized piecewise affine method to solve the following RNN training problem with ReLU:"
        )
        print("minimize Theta(z):=F(z) + p(z)")
        print("where p(z) is the l1-penalty terms with (beta1,beta2) for")
        print("u - Psi(h,theta_in) = 0")
        print("h - ReLU(u) = 0")
        print("v - PhiX(h,theta_out) = 0\n")

    # Load and split the dataset
    x_trainset, y_trainset, x_testset, y_testset, N_train, N_test, T, Nx, Ny = load_split_datasetN(0.7)    
    # # 测试是否分割正确
    # print(
    #         f"{x_trainset}\n{y_trainset}\n{x_testset}\n{y_testset}\n"
    #     )
    print(x_trainset.shape, x_testset.shape, y_trainset.shape, y_testset.shape)
    Ntheta = Nh * (Ny+Nh+Nx+1) + Ny

    
    #计算罚参阈值
    gammay = np.sum(y_trainset**2) / (N_train*T)
    gamma1 = ( (gammay/lambda2)**(T/2) -1 ) / ( (gammay/lambda2)**(1/2) -1)
    print("gammay=", gammay, " gamma1=", gamma1)
    print("beta1>", 2*gamma1*gammay/((lambda1*N_train*T)**0.5), " beta2>", 2*(gammay/(N_train*T))**0.5) 

    # Initialize variables
    variables = initialize_variablesN(Ny,
                                     Nh,
                                     Nx,
                                     T,
                                     N_train,
                                     N_test,
                                     distribution_type=distribution_type,
                                     mean=mean,
                                     std=std,
                                     x_trainset=x_trainset,
                                     y_trainset=y_trainset,
                                     x_testset=x_testset,
                                     y_testset=y_testset,
                                     beta1=beta1,
                                     beta2=beta2,
                                     lambda1=lambda1,
                                     lambda2=lambda2,
                                     lambda3=lambda3,
                                     lambda4=lambda4,
                                     lambda5=lambda5)

    # Initialize lists to store results
    FeasVi_h, FeasVi_u, FeasVi_v = [variables["feasvi_h0"]], [variables["feasvi_u0"]], [variables["feasvi_v0"]]
    TrainErr, TestErr = [variables["TrainErr0"]], [variables["TestErr0"]]  
    rho_k = rho1
    funcval_k = [variables["lossTheta0"].item()]
    res_value=[]
    res_norm=[]
    time_SPLQPRNN = [0]
    if print_option == 2:
        print(f"\n{'Iteration':<10}{'TrainErr':<15}{'TestErr':<15}{'FeasVi_h':<15}{'FeasVi_u':<15}{'FeasVi_v':<15}{'Theta_v':<15}{'Time (s)':<10}")
        print("-" * 70)
        print(
            f"{0:<10}{TrainErr[-1]:<15.4f}{TestErr[-1]:<15.4f}{FeasVi_h[-1]:<15.4f}{FeasVi_u[-1]:<15.4f}{FeasVi_v[-1]:<15.4f}{funcval_k[-1]:<15.4f}{time_SPLQPRNN[-1]:<10.4f}"
        )
                    
    # Initialize variables for both the problem
    A_k, W_k, V_k, b_k, c_k = variables["A0"], variables[
                "W0"], variables["V0"], variables["b0"], variables["c0"]
    thetain_k, thetaout_k, h_k, u_k, v_k, hti_k = variables[
                "thetain_0"], variables["thetaout_0"], variables["h0"], variables[
                    "u0"], variables["v0"], variables["hti0"] 
    
    

    # Begin outer loop
    for k in range(maxiter):
        start_k = time.perf_counter()
        # print(f'Iteration {k+1}/{maxiter}') 
        

        ####用cvxpy求解子问题
        #准备工作，计算两个列向量
        sgn_u_k = np.sign( u_k + np.random.uniform(-1e-15, 1e-15, size=(N_train*T*Nh, 1)) ) 
        possgn_u_k = np.maximum(sgn_u_k,0)
        # sgn_diag = np.diag(sgn_u_k.ravel())
        # sgn_sparse = sparse.diags(sgn_u_k.ravel())
        #定义变量，cvxpy中可设置变量形状，先统一为一维数组吧
        s = cp.Variable(Ntheta + N_train*T*(2*Nh + Ny))
        # sthetaout = cp.reshape(cp.hstack([s[:Ny*Nh], s[Ntheta-Ny:Ntheta]]), (-1,1), order='F')
        # sthetain = cp.reshape(s[Ny*Nh : Ntheta-Ny] , (-1,1), order='F')
        SA = cp.reshape(s[:Ny*Nh], (Ny, Nh), order='F')
        SW = cp.reshape(s[Ny*Nh : Nh*(Ny+Nh)], (Nh, Nh), order='F')
        SV = cp.reshape(s[Nh*(Ny+Nh) : Nh*(Ny+Nh+Nx)], (Nh, Nx), order='F')
        sb = cp.reshape(s[Nh*(Ny+Nh+Nx) : Ntheta-Ny], (-1,1), order='F')
        sc = cp.reshape(s[Ntheta-Ny : Ntheta], (-1,1), order='F')
        sh = cp.reshape(s[Ntheta : Ntheta+N_train*T*Nh], (-1,1), order='F')
        su = cp.reshape(s[Ntheta+N_train*T*Nh : Ntheta+2*N_train*T*Nh], (-1,1), order='F')
        sv = cp.reshape(s[Ntheta+2*N_train*T*Nh : ], (-1,1), order='F')
        # 
        # 损失函数线性项
        term1 = cp.matmul((2*(v_k - y_trainset.ravel().reshape(-1,1))/(N_train*T) ).T, sv) 
        # 正则项线性项  
        term2 = 2 * lambda1 * cp.sum(cp.multiply(SA, A_k)) + 2 * lambda2 * cp.sum(cp.multiply(SW, W_k)) + 2 * lambda3 * cp.sum(cp.multiply(SV, V_k)) + 2 * lambda4 * cp.sum(cp.multiply(sb, b_k)) + 2 * lambda5 * cp.sum(cp.multiply(sc, c_k))
        # 二次正则项
        term3 = 0.5 * rho_k * cp.sum_squares(s)
        # h-[u]_+惩罚项，h_k, dh, u_k, du 都是列向量
        h_plus_sh = h_k + sh
        u_plus_su = u_k + su
        # u_plus_su_pos = cp.pos(u_plus_su)  # [u]_+ in CVXPY
        term4 = beta1 * cp.norm1(h_plus_sh - cp.multiply(u_plus_su,possgn_u_k))

        # u-hz惩罚项
        # 不用for循环
        # 滞后项部分
            # h_trial_prev = np.hstack([ np.zeros((N_train, Nh)), h_trial.reshape((N_train,-1))[:,:-Nh] ])
            # h_trial_prev = h_trial_prev.reshape((-1,1), order='C').reshape((Nh, -1), order='F')
            # valuel1term_2 = u_trial.reshape((Nh, -1), order='F') - W_trial @ h_trial_prev - V_trial @ x_trainset.reshape((-1,Nx)).T - b_trial
        # h_prev = np.concatenate([np.zeros((N_train,Nh,1)), hti_k[:,:,:-1]], axis=2)
        # h_prev = np.transpose(h_prev,(1,0,2)).reshape((Nh,-1)) 
        h_prev = np.hstack([ np.zeros((N_train, Nh)), h_k.reshape((N_train,-1))[:,:-Nh] ])
        h_prev = h_prev.reshape((-1,1), order='C').reshape((Nh, -1), order='F')
        sh_prev = cp.hstack([ np.zeros((N_train, Nh)), sh.reshape((N_train,-1), order='C')[:,:-Nh] ])
        sh_prev = sh_prev.reshape((-1,1), order='C').reshape((Nh, -1), order='F')
        W_part = (W_k + SW) @ h_prev + W_k @ sh_prev
        # 线性/常数项部分
        # x_trainset 形状为 (N_train, T, Nx) -> 转为 (Nx, T*N_train)
        X_part = (V_k + SV) @ x_trainset.reshape((-1,Nx)).T + b_k + sb  # 广播 b_k + sb 到每一列
        
        uh = (u_k + su).reshape((Nh, -1), order='F') - X_part - W_part 
        term5 = beta1 * cp.norm1(uh)

        #v-hz惩罚项
        # 不用for循环
        h_curr = h_k.reshape((Nh, -1), order='F')
        sh_curr = sh.reshape((Nh, -1), order='F')
        
        vh = (v_k + sv).reshape((Ny, -1), order='F') - \
             ((A_k + SA) @ h_curr + A_k @ sh_curr)  - \
             (c_k + sc) # 这里的 c_k+sc 会自动广播到 (Ny, T)
             
        term6 = beta2 * cp.norm1(vh)
        # add up as objective
        objective = cp.Minimize(term1 + term2 + term3 + term4 + term5 + term6)
        # add constraints
        constraints = [cp.multiply(u_plus_su,sgn_u_k) >= 0]
        # constraints = [sgn_diag @ u_plus_su >= 0]
        # constraints = [sgn_sparse @ u_plus_su >= 0]
        # solve
        prob = cp.Problem(objective,constraints)
        prob.solve(solver=cp.OSQP, verbose=False) 
        
        d = s.value.reshape(-1,1)
        print("CVXPY status:", prob.status)
        print("Optimal value:", prob.value)
        
        # 记录res_norm
        res_norm.append( np.sqrt(np.sum(d**2)) )
        # 记录res_value
        res_value_temp = prob.value + ALfunc.ValueLoss1N(y_trainset, v_k, T, N_train) + lambda1 * np.sum(A_k** 2) + lambda2 * np.sum(W_k** 2) + lambda3 * np.sum(V_k** 2) + lambda4 * np.sum(b_k** 2) + lambda5 * np.sum(c_k** 2)
        res_value.append( funcval_k[-1] - res_value_temp )





        # if result.fun == objective_function(initial_guess):
        #     end_k = time.perf_counter()
        #     time_SPLQPRNN.append(end_k - start_k)
        #     print(
        #     f"\nThe sequential piecewise linear-quadratic programming method teminates at {k+1}th iteration without update.\n")
        #     break

        # # update z_k
        # d = result.x.reshape(-1,1)  # 现在d是列向量
        # value of Theta(z_k+d_k) 
        A_trial = A_k + d[:Ny*Nh].reshape((Ny,Nh), order="F")
        W_trial = W_k + d[Ny*Nh : Nh*(Ny+Nh)].reshape((Nh,Nh), order="F")
        V_trial = V_k + d[Nh*(Ny+Nh) : Nh*(Ny+Nh+Nx)].reshape((Nh,Nx), order="F")
        b_trial = b_k + d[Nh*(Ny+Nh+Nx) : Ntheta-Ny]
        c_trial = c_k + d[Ntheta-Ny : Ntheta]
        h_trial = h_k + d[Ntheta : Ntheta+N_train*T*Nh] 
        u_trial = u_k + d[Ntheta+N_train*T*Nh : Ntheta+2*N_train*T*Nh]
        v_trial = v_k + d[Ntheta+2*N_train*T*Nh : ]
        #  
        regvalue_trial = lambda1 * np.sum(A_trial ** 2) + lambda2 * np.sum(W_trial ** 2) + lambda3 * np.sum(V_trial ** 2) + lambda4 * np.sum(b_trial ** 2) + lambda5 * np.sum(c_trial ** 2)
        #  
        h_trial_prev = np.hstack([ np.zeros((N_train, Nh)), h_trial.reshape((N_train,-1))[:,:-Nh] ])
        h_trial_prev = h_trial_prev.reshape((-1,1), order='C').reshape((Nh, -1), order='F')
        valuel1term_2 = u_trial.reshape((Nh, -1), order='F') - W_trial @ h_trial_prev - V_trial @ x_trainset.reshape((-1,Nx)).T - b_trial
        # x_trainset 形状为 (N_train, T, Nx) -> 转为 (Nx, T*N_train)
        h_trial_curr = h_trial.reshape((Nh, -1), order='F')
        valuel1term_3 = v_trial.reshape((Ny, -1), order='F') - A_trial @ h_trial_curr - c_trial 
        valuel1term = beta1 * np.sum(np.abs(h_trial - toolfunc.ReLU(u_trial))) + beta1 * np.sum(np.abs(valuel1term_2)) + beta2 * np.sum(np.abs(valuel1term_3))  
        # 
        lossTheta_trial = ALfunc.ValueLoss1N(y_trainset, v_trial, T, N_train) + regvalue_trial + valuel1term
        # Determine the amount of decrease
        if funcval_k[-1]-lossTheta_trial >= 0.5 * eta1 * rho_k * np.dot(d.T,d):
            flag = 1 # successful 
            # A_k, W_k, V_k, b_k, c_k += DA, DW, DV, db, dc
            A_k = A_trial
            W_k = W_trial
            V_k = V_trial
            b_k = b_trial
            c_k = c_trial
            h_k = h_trial 
            u_k = u_trial
            v_k = v_trial 
        else:
            flag = 0   # unsuccessful
            rho_k *= eta2
        
        # Record time
        end_k = time.perf_counter()
        time_SPLQPRNN.append(end_k - start_k)

        print(f"flag_{k+1}: ", flag )
        # print(funcval_k[-1]-lossTheta_trial, 0.5 * eta1 * rho_k * np.dot(d.T,d))
        if flag ==1:
        # Record value of Theta(z_k)
            funcval_k.append(lossTheta_trial.item())

            # Record TrainErr
            y_hat_k = np.empty((N_train, T, Ny))
            for i in range(N_train):
                _, _, _, _, _, _, y_hat_k[i,:,:] = ALfunc.Calculauxi(x_trainset[i], A_k, W_k, V_k, b_k, c_k, T, Nh, Nx, Ny) 
            TrainErr.append((1 /(N_train*T)) * np.sum((y_hat_k - y_trainset)**2))
            
            # Record TestErr  
            y_test_hat = np.empty((N_test, T, Ny))
            for i in range(N_test):
                _, _, _, _, y_test_hat[i,:,:] = ALfunc.CalculY(x_testset[i], A_k, W_k, V_k, b_k, c_k, T, Nh, Nx, Ny) 
            TestErr.append((1 / (N_test*T)) * np.sum((y_test_hat - y_testset)**2)) 

            # # Record violation in l1-norm
            # FeasVi_h.append( (1 / T) * np.sum(np.abs(h_k - toolfunc.ReLU(u_k))) ) 
            # FeasVi_u.append( (1 / T) * np.sum(np.abs(u_k - ALfunc.PsiX(ht_k, x_trainset, thetain_k, Nx, Nh, T))) )
            # hthetaout_k = ALfunc.CalculY_auxi(thetaout_k, ht_k, Nh, Ny, T)
            # FeasVi_v.append( (1 / T) * np.sum(np.abs(v_k- hthetaout_k.ravel().reshape(-1,1))) ) 
            # Record violation in l2-norm

            # beta1 * np.sum(np.abs(h_trial - toolfunc.ReLU(u_trial))) + beta1 * np.sum(np.abs(valuel1term_2)) + beta2 * np.sum(np.abs(valuel1term_3)) 

            FeasVi_h.append( (1 / (N_train*T)) * np.sum( (h_k - toolfunc.ReLU(u_k))**2 )**(1/2) ) 
            FeasVi_u.append( (1 / (N_train*T)) * np.sum( (valuel1term_2)**2 )**(1/2) ) 
            FeasVi_v.append( (1 / (N_train*T)) * np.sum( valuel1term_3**2 )**(1/2) )
        else:
            funcval_k.append(funcval_k[-1])
            TrainErr.append(TrainErr[-1])
            TestErr.append(TestErr[-1])
            FeasVi_h.append(FeasVi_h[-1])
            FeasVi_u.append(FeasVi_u[-1])
            FeasVi_v.append(FeasVi_v[-1])

        if print_option == 2:
            print(
                f"{k+1:<10}{TrainErr[-1]:<15.4f}{TestErr[-1]:<15.4f}{FeasVi_h[-1]:<15.4f}{FeasVi_u[-1]:<15.4f}{FeasVi_v[-1]:<15.4f}{funcval_k[-1]:<15.4f}{time_SPLQPRNN[-1]:<10.4f}"
            ) 

        

    # Prepare results to return
    res_value.append(float('nan'))
    res_norm.append(float('nan'))
    results = {
        "TrainErr": TrainErr,
        "TestErr": TestErr,
        "time": time_SPLQPRNN,
        "FeasVi_h": FeasVi_h,
        "FeasVi_u": FeasVi_u,
        "FeasVi_v": FeasVi_v,
        "Theta_v": funcval_k,
        "res_value": res_value,
        "res_norm": res_norm
    }

    if print_option == 1:
        print(f"\nFinal Results:\n{'-' * 40}")
        print(f"TrainErr   : {TrainErr[-1]:.4f}")
        print(f"TestErr    : {TestErr[-1]:.4f}")
        print(f"FeasVi_h   : {FeasVi_h[-1]:.4f}")
        print(f"FeasVi_u   : {FeasVi_u[-1]:.4f}")
        print(f"FeasVi_v   : {FeasVi_v[-1]:.4f}")
        print(f"{'-' * 40}\n")

    """
    将多个数组保存为CSV文件，第一列为iterations
    
    参数:
    funcval_k, TrainErr, ... : 同长度的数组
    filename : 输出文件名 (默认: 'synT10_Errors_SPLQP.csv')
    """
    
    filename= 'TIMIT_Errors_SPLQP.csv'
    # 检查所有数组长度是否一致
    lengths = [len(arr) for arr in [funcval_k, TrainErr, TestErr, FeasVi_h, FeasVi_u, FeasVi_v, time_SPLQPRNN]]
    if len(set(lengths)) != 1:
        raise ValueError(f"数组长度不一致! 长度: {lengths}")
    
    n = lengths[0]
    iterations = np.arange(n)
    
    df = pd.DataFrame({
        'iterations': iterations,
        'funcval_kSPLQP': funcval_k,
        'TrainErrSPLQP': TrainErr,
        'TestErrSPLQP': TestErr,
        'FeasVi_hSPLQP': FeasVi_h,
        'FeasVi_uSPLQP': FeasVi_u,
        'FeasVi_vSPLQP': FeasVi_v,
        'TimeSPLQP': time_SPLQPRNN,
        "res_value": res_value,
        "res_norm": res_norm
    })
    
    df.to_csv(filename, index=False)
    print(f"成功保存 {n} 行数据到 {filename}")

    return results
