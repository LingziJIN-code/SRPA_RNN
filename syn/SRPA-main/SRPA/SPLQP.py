# Import and install dependency packages
from dependency_packages import np, pd, scipy, copy, time, random, math, gc, cp
from scipy.optimize import minimize #subalgorithm


# Import mypackages
import mypackages.tool_functions as toolfunc
import mypackages.initial_functions as inifunc
import mypackages.functions_calculate_ALfunc as ALfunc
import mypackages.BCD_closed_form as BCD_closed
from gen_synthetic_dataset import generate_synthetic_dataset


def load_dataset(dataset_name):
    # dataset_ori = pd.read_csv(dataset_name + '.csv', header=0, index_col=0) #header=0, index_col=0会把csv第0行作为列名，第0列作为行索引，不会当成内容，也就是自动删掉了第0行和第0列
    dataset_ori = pd.read_csv(dataset_name + '.csv', header=None, index_col=None) #数据集本身没有行索引和列索引的时候用这个
    return dataset_ori


def split_dataset(dataset_ori, Ny, sizerate_training=0.9):
    dataset_length, data_dim = dataset_ori.shape
    Nx = data_dim - Ny
    train_length = int(sizerate_training * dataset_length)
    T = train_length
    T_test = dataset_length - T

    x_trainset = dataset_ori.iloc[:train_length, :Nx].to_numpy() 
    y_trainset = dataset_ori.iloc[:train_length, Nx:].to_numpy()  

    x_testset = dataset_ori.iloc[train_length:, :Nx].to_numpy()  
    y_testset = dataset_ori.iloc[train_length:, Nx:].to_numpy()

    return x_trainset, y_trainset, x_testset, y_testset, T, T_test, Nx


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
    - lambda1: Regularization parameter
    - eta1: in (0,1), parameter for measuring the sufficient descent (default: 0.99)
    - eta2: >1, update parameter for rho (default: 1.2)
    - rho1: initial value for rho, the regularization parameter

    Returns:
    - A dictionary containing the initialized parameters.
    """

    # Calculate default values if not provided,默认按251120版的line68设置
    if rho1 is None:
        # rho1 = ( max(2*lambda1,1/T) + pow(T*(Nh+Ny),0.5) * pow( max(beta1*beta1*Nh+beta2*beta2*Ny, T ),0.5) ) / (1-eta1) 
        rho1=0.24 
        print("rho_1", rho1) 
        # rho1 = 1
        # rho1 = 10000
     
 

    return {
        "eta1": eta1,
        "eta2": eta2,
        "rho1":rho1
    }   #注意，返回的是个字典


def initialize_variables(Ny,
                         Nh,
                         Nx,
                         T,
                         T_test,
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
    - T: Size of the training set
    - distribution_type: Type of distribution for initialization ("Gaussian" or other, default: "Gaussian")
    - mean: Mean for the distribution (default: 0 for Gaussian, 0 for other)
    - std: Standard deviation for the distribution (default: 1e-3 for Gaussian, 20 for other)
    - x_trainset: Training set inputs
    - y_trainset: Training set outputs 
    - beta1: penalty parameter for hidden layer (default: 1)
    - beta2: penalty parameter for output layer (default: 1)
    - lambda1: Regularization parameters

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
    # y_hat0是feedforward得到的vt0^T,于y_trainset同维度 
    # hthetaout0就是v=hz约束的右端值的T*Ny矩阵，由于此时v0是feedfoward得到的，所以其实是=vt0^T=y_hat0
    A0, W0, V0, b0, c0, thetain_0, thetaout_0 = inifunc.iniVariable(
        Ny, Nh, Nx, T, mean, std, distribution_type)
    u0, ut0, h0, ht0, v0, vt0, y_hat0 = ALfunc.Calculauxi(x_trainset, A0, W0, V0, 
                                                          b0, c0, T, Nh, Nx, Ny) 
    hthetaout0 = ALfunc.CalculY_auxi(thetaout_0, ht0, Nh, Ny, T) 
    # print("number of ui=0", np.sum(u0==0))

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
    lossTheta0 = ALfunc.ValueLoss1(y_trainset, v0, Ny, T) + regvalue0 + ALfunc.Valuel1term(x_trainset, ht0, h0, u0, v0, thetain_0, thetaout_0, beta1, beta2, Nx, Nh, Ny,
            T)

    #Trainerror
    TrainErr0 = (1 / T) * np.sum((y_hat0 - y_trainset)**2)
    if np.isnan(TrainErr0) == True:
        TrainErr0 = 1e+30
    
    #Testerror 
    _, _, _, _, y_test_hat0 = ALfunc.CalculY(np.concatenate((x_trainset, x_testset), axis=0), A0, W0, V0, b0, c0,
                                             T+T_test, Nh, Nx, Ny)
    y_test_hat0 = y_test_hat0[T:,]
    TestErr0 = (1 / T_test) * np.sum((y_test_hat0 - y_testset)**2)


    # FeasVi_h0 = (1 / T) * np.sum(np.abs(h0 - toolfunc.ReLU(u0))) 
    # FeasVi_u0 = (1 / T) * np.sum(
    #     np.abs(u0 - ALfunc.PsiX(ht0, x_trainset, thetain_0, Nx, Nh, T)))
    # FeasVi_v0 = (1 / T) * np.sum(np.abs(v0- hthetaout0.ravel().reshape(-1,1)))
    FeasVi_h0 = (1 / T) * np.sum((h0 - toolfunc.ReLU(u0))**2)**(1/2) 
    FeasVi_u0 = (1 / T) * np.sum(
        (u0 - ALfunc.PsiX(ht0, x_trainset, thetain_0, Nx, Nh, T))**2)**(1/2)
    FeasVi_v0 = (1 / T) * np.sum((v0- hthetaout0.ravel().reshape(-1,1))**2)**(1/2)

    return {
        "A0": A0,
        "W0": W0,
        "V0": V0,
        "b0": b0,
        "c0": c0,
        "thetain_0": thetain_0,
        "thetaout_0": thetaout_0,
        "u0": u0,
        "ut0": ut0,
        "h0": h0,
        "ht0": ht0,
        "v0":v0, 
        "vt0":vt0,
        "y_hat0": y_hat0,
        "hthetaout0": hthetaout0, 
        "lossTheta0": lossTheta0,
        "feasvi_h0": FeasVi_h0,
        "feasvi_u0": FeasVi_u0,
        "feasvi_v0": FeasVi_v0,
        "TrainErr0": TrainErr0,
        "TestErr0": TestErr0
    }   #注意，返回的是个字典


def SPLQP_ReLU_optimization(dataset_name,
                              Ny=1,
                              Nh=20,
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
                              eta2=5 / 6, 
                              rho1=0.24, 
                              print_option=2):
    """
    Perform SPLQP to train RNNs.

    Parameters:
    - dataset_name: Name of the dataset to load
    - Ny: Number of output units (default: 1)
    - Nh: Number of hidden units (default: 20)
    - maxiter: Maximum number of iterations for the outer loop (default: 10) 
    - distribution_type: Type of distribution for variable initialization ("Gaussian" "He", "Glorot", "LeCun", default: "Gaussian")
    - mean: Mean for the initialization distribution (default: 0 for Gaussian, 0 for other)
    - std: Standard deviation for the initialization distribution (default: 1e-3 for Gaussian, 20 for other)
    - beta1: penalty parameter for hidden layer (default: 1)
    - beta2: penalty parameter for output layer (default: 1)
    - lambda1/2/3/4/5: Regularization parameters
    - eta1, eta2: Parameters for updating the algorithm 
    - print: Control the verbosity of the output (default: 0)
        - 0: No output
        - 1: Print final TrainErr, TestErr, FeasVi_h, FeasVi_u, FeasVi_v
        - 2: Print TrainErr, TestErr, FeasVi_h, FeasVi_u, FeasVi_v, Theta_value and time for each outer iteration

    Returns:
    - results: A dictionary containing lists for TrainErr, TestErr, time, FeasVi_h, FeasVi_u, FeasVi_v
    """
    

    # Print the optimization problem being solved
    if print_option >= 1:
        print(
            "\nWe use a sequential piecewise linear-quadratic programming method to solve the following RNN training problem with ReLU:"
        )
        print("minimize Theta(z):=F(z) + p(z)")
        print("where p(z) is the l1-penalty terms with (beta1,beta2) for")
        print("u - Psi(h,theta_in) = 0")
        print("h - ReLU(u) = 0")
        print("v - PhiX(h,theta_out) = 0\n")

    # Load and split the dataset
    dataset_ori = load_dataset(dataset_name)
    # # 测试分割前是否正确
    # print(
    #         f"{dataset_ori}\n"
    #     )
    # x_trainset, y_trainset, x_testset, y_testset, T, T_test, Nx = split_dataset(
    #     dataset_ori, Ny)
    x_trainset, y_trainset, x_testset, y_testset, T, T_test, Nx = split_dataset(
        dataset_ori, Ny, 0.8)   #T小时分割比为0.8，T大时为0.9
    # # 测试是否分割正确
    # print(
    #         f"{x_trainset}\n{y_trainset}\n{x_testset}\n{y_testset}\n"
    #     )
    Ntheta = Nh * (Ny+Nh+Nx+1) + Ny

     

    # # Initialize SPLQP parameters
    # params = initialize_SPLQP_params(Nh,
    #                                Ny,
    #                                Nx,
    #                                T,
    #                                beta1=beta1,
    #                                beta2=beta2,
    #                                lambda1=lambda1,
    #                                eta1=eta1,
    #                                eta2=eta2,
    #                                rho1=None)

    # Initialize variables
    variables = initialize_variables(Ny,
                                     Nh,
                                     Nx,
                                     T,
                                     T_test,
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
    # rho_k = [params["rho1"]]
    # rho_k = params["rho1"] 
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
    thetain_k, thetaout_k, h_k, u_k, v_k, ht_k, ut_k, vt_k = variables[
                "thetain_0"], variables["thetaout_0"], variables["h0"], variables[
                    "u0"], variables["v0"], variables["ht0"], variables["ut0"], variables["vt0"]
    
    y_trainset_flat = y_trainset.ravel().reshape((-1,1))

    # Begin outer loop
    for k in range(maxiter):
        start_k = time.perf_counter()
        

        ####用cvxpy求解子问题
        #准备工作，计算两个列向量 
        sgn_u_k = np.sign( u_k + np.random.uniform(-1e-15, 1e-15, size=(T*Nh, 1)) )
        possgn_u_k = np.maximum(sgn_u_k,0)
        #定义变量，cvxpy中可设置变量形状，先统一为一维数组
        s = cp.Variable(Ntheta + T*(2*Nh + Ny))
        sthetaout = cp.reshape(cp.hstack([s[:Ny*Nh], s[Ntheta-Ny:Ntheta]]), (-1,1), order='F')
        sthetain = cp.reshape(s[Ny*Nh : Ntheta-Ny] , (-1,1), order='F')
        SA = cp.reshape(s[:Ny*Nh], (Ny, Nh), order='F')
        SW = cp.reshape(s[Ny*Nh : Nh*(Ny+Nh)], (Nh, Nh), order='F')
        SV = cp.reshape(s[Nh*(Ny+Nh) : Nh*(Ny+Nh+Nx)], (Nh, Nx), order='F')
        sb = cp.reshape(s[Nh*(Ny+Nh+Nx) : Ntheta-Ny], (-1,1), order='F')
        sc = cp.reshape(s[Ntheta-Ny : Ntheta], (-1,1), order='F')
        sh = cp.reshape(s[Ntheta : Ntheta+T*Nh], (-1,1), order='F')
        su = cp.reshape(s[Ntheta+T*Nh : Ntheta+2*T*Nh], (-1,1), order='F')
        sv = cp.reshape(s[Ntheta+2*T*Nh : ], (-1,1), order='F')
        # 
        # 损失函数线性项
        term1 = cp.matmul((2*(v_k - y_trainset.ravel().reshape(-1,1))/T).T, sv) 
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
        uh_list = []
        for t in range(T):
            x_t = x_trainset[t, :].reshape(-1, 1)
            if t == 0:
                uh_t = u_k[t*Nh:(t+1)*Nh] - V_k @ x_t - b_k + su[t*Nh:(t+1)*Nh] - SV @ x_t - sb
            else:
                uh_t = u_k[t*Nh:(t+1)*Nh] - W_k @ h_k[(t-1)*Nh:t*Nh] - V_k @ x_t - b_k + su[t*Nh:(t+1)*Nh] - W_k @ sh[(t-1)*Nh:t*Nh] - SW @ h_k[(t-1)*Nh:t*Nh] - SV @ x_t - sb 
            uh_list.append(uh_t)
        uh = cp.hstack(uh_list)
        term5 = beta1 * cp.norm1(uh)
        #v-hz惩罚项
        vh_list = []
        for t in range(T):
            vh_t = (v_k[t*Ny:(t+1)*Ny] - A_k @ h_k[t*Nh:(t+1)*Nh] - c_k +
                    sv[t*Ny:(t+1)*Ny] - A_k @ sh[t*Nh:(t+1)*Nh] - SA @ h_k[t*Nh:(t+1)*Nh] - sc)
            vh_list.append(vh_t)
        vh = cp.hstack(vh_list)
        term6 = beta2 * cp.norm1(vh)
        # add up as objective
        objective = cp.Minimize(term1 + term2 + term3 + term4 + term5 + term6)
        # add constraints
        constraints = [cp.multiply(u_plus_su,sgn_u_k) >= 0]
        # solve
        prob = cp.Problem(objective,constraints)
        # prob.solve(solver=cp.OSQP, verbose=True)
        prob.solve(solver=cp.OSQP, verbose=False)
        d = s.value.reshape(-1,1)
        print("CVXPY status:", prob.status)
        print("Optimal value:", prob.value)

        # 记录res_norm
        res_norm.append( np.sqrt(np.sum(d**2)) )
        # 记录res_value
        res_value_temp = prob.value + (1/T)*np.sum((v_k-y_trainset_flat)**2) + lambda1 * np.sum(A_k** 2) + lambda2 * np.sum(W_k** 2) + lambda3 * np.sum(V_k** 2) + lambda4 * np.sum(b_k** 2) + lambda5 * np.sum(c_k** 2)
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
        dthetaout = np.concatenate( (d[:Ny*Nh], d[Ntheta-Ny:Ntheta]), axis=0 )
        dthetain = d[Ny*Nh : Ntheta-Ny]
        DA = d[:Ny*Nh].reshape((Ny,Nh), order="F")
        DW = d[Ny*Nh : Nh*(Ny+Nh)].reshape((Nh,Nh), order="F")
        DV = d[Nh*(Ny+Nh) : Nh*(Ny+Nh+Nx)].reshape((Nh,Nx), order="F")
        db = d[Nh*(Ny+Nh+Nx) : Ntheta-Ny]
        dc = d[Ntheta-Ny : Ntheta]
        dh = d[Ntheta : Ntheta+T*Nh]
        dht = dh.reshape((Nh,T), order="F")
        du = d[Ntheta+T*Nh : Ntheta+2*T*Nh]
        dv = d[Ntheta+2*T*Nh : ] 
        regvalue_trial = lambda1 * np.sum((A_k+DA) ** 2) + lambda2 * np.sum((W_k+DW) ** 2) + lambda3 * np.sum((V_k+DV) ** 2) + lambda4 * np.sum((b_k+db) ** 2) + lambda5 * np.sum((c_k+dc) ** 2)
        lossTheta_trial = ALfunc.ValueLoss1(y_trainset, v_k+dv, Ny, T) + regvalue_trial + ALfunc.Valuel1term(x_trainset, ht_k+dht, h_k+dh, u_k+du, v_k+dv, thetain_k+dthetain, thetaout_k+dthetaout, beta1, beta2, Nx, Nh, Ny, T)
        # Determine the amount of decrease
        if funcval_k[-1]-lossTheta_trial >= 0.5 * eta1 * rho_k * np.dot(d.T,d):
            flag = 1 # successful
            # A_k, W_k, V_k, b_k, c_k += DA, DW, DV, db, dc
            A_k += DA
            W_k += DW
            V_k += DV
            b_k += db
            c_k += dc
            thetain_k += dthetain
            thetaout_k += dthetaout
            h_k += dh
            ht_k += dht
            u_k += du
            v_k += dv 
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
            _, _, _, _, _, _, y_hat_k = ALfunc.Calculauxi(x_trainset, A_k,
                                                    W_k, V_k, b_k,
                                                    c_k, T, Nh, Nx, Ny)
            TrainErr.append((1 / T) * np.sum((y_hat_k - y_trainset)**2))
            
            # Record TestErr 
            _, _, _, _, _, _, y_test_hat = ALfunc.Calculauxi(
                np.concatenate((x_trainset, x_testset), axis=0), A_k, W_k, V_k, b_k, c_k,
                T+T_test, Nh, Nx, Ny) 
            y_test_hat = y_test_hat[T:,]
            TestErr.append((1 / T_test) * np.sum((y_test_hat - y_testset)**2))

            # # Record violation in l1-norm
            # FeasVi_h.append( (1 / T) * np.sum(np.abs(h_k - toolfunc.ReLU(u_k))) ) 
            # FeasVi_u.append( (1 / T) * np.sum(np.abs(u_k - ALfunc.PsiX(ht_k, x_trainset, thetain_k, Nx, Nh, T))) )
            # hthetaout_k = ALfunc.CalculY_auxi(thetaout_k, ht_k, Nh, Ny, T)
            # FeasVi_v.append( (1 / T) * np.sum(np.abs(v_k- hthetaout_k.ravel().reshape(-1,1))) ) 
            # Record violation in l2-norm
            FeasVi_h.append( (1 / T) * np.sum( (h_k - toolfunc.ReLU(u_k))**2 )**(1/2) ) 
            FeasVi_u.append( (1 / T) * np.sum( (u_k - ALfunc.PsiX(ht_k, x_trainset, thetain_k, Nx, Nh, T))**2 )**(1/2) )
            hthetaout_k = ALfunc.CalculY_auxi(thetaout_k, ht_k, Nh, Ny, T)
            FeasVi_v.append( (1 / T) * np.sum( (v_k- hthetaout_k.ravel().reshape(-1,1))**2 )**(1/2) )
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
    filename= 'synT10_Errors_SPLQP.csv'
    # filename= 'SP500_Errors_SPLQP.csv'
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
