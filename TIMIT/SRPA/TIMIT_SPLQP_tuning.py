from SPLQP_N import SPLQP_N_ReLU_optimization 
from dependency_packages import np, random, os, csv, pd

 

# dimension of RNN
Nx = 129
Ny = 129 
Nh = 2

param_space= {
    'init_methods': ['Gaussian'],
    'gaussian_stddevs': [0.001, 0.01, 0.1], 
    'rho1': [0.001, 0.003, 0.005],
    'eta1': [0.7, 0.8, 0.9],
    'eta2': [1.1, 1.3, 1.5]
}

# 创建文件并写入表头 
filename = 'parameter_tuning_TIMIT_SPLQP_averaged3.csv'
fieldnames = [
    'trial', 'init_method', 'stddev', 'rho1', 'eta1', 'eta2',
    'train_loss', 'test_loss', 'train_loss_std', 'test_loss_std'
]
# 检查文件是否存在，不存在则创建并写入表头
if not os.path.exists(filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    print(f"Created new file: {filename} with headers")
else:
    print(f"Appending to existing file: {filename}")

"""为指定优化器调优参数 - 使用遍历搜索"""
print(f"\n{'='*50}")
print("开始调优 SPLQP 参数")
print(f"{'='*50}")

best_score = float('inf')
best_config = None
trial_results = []
trial_count = 0

with open(filename, 'a', newline='') as csvfile:  # 以追加模式打开文件
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 遍历所有初始化方法
    for init_method in param_space['init_methods']:
        # 处理标准差参数
        if init_method == 'Gaussian':
            stddev_choices = param_space['gaussian_stddevs']
        else:
            stddev_choices = [20]  # 其他方法使用默认值
                
        # 遍历标准差选择
        for stddev in stddev_choices:
            for rho1 in param_space['rho1']: 
                for eta1 in param_space['eta1']:
                    for eta2 in param_space['eta2']: 
                        trial_count += 1
                        print(f"试验 {trial_count}: {init_method}(σ={stddev}), rho1={rho1}, eta1={eta1}, eta2={eta2}")

                        # 存储多次训练的结果
                        train_losses = []
                        test_losses = []

                        # 进行独立训练循环
                        num_repeats = 3
                        tau=5e-4

                        for repeat in range(num_repeats):
                            print(f"  循环 {repeat + 1}/{num_repeats}")

                            # 设置随机种子（每次使用不同的种子）
                            np.random.seed(1234 + repeat * 10)
                            random.seed(1234 + repeat * 10) 

                            results = SPLQP_N_ReLU_optimization(Nh=Nh, maxiter=100, 
                            distribution_type=init_method, mean=0, std=stddev, 
                            beta1=0.04, beta2=0.04, lambda1 = tau / (Nh * Ny), lambda2 = tau / (Nh * Nh), lambda3 = tau / (Nx * Nh), lambda4 = tau / Nh, lambda5 = tau / Ny, 
                            eta1=eta1, eta2=eta2, rho1=rho1, print_option=0) 

                            # 获取本次训练的最终性能
                            final_loss = results["TrainErr"][-1]
                            final_test_loss = results["TestErr"][-1]

                            # 记录结果
                            train_losses.append(final_loss)
                            test_losses.append(final_test_loss)

                        # 计算独立循环的平均结果
                        avg_train_loss = np.mean(train_losses)
                        avg_test_loss = np.mean(test_losses)
                        std_train_loss = np.std(train_losses)
                        std_test_loss = np.std(test_losses)
                        
                        print(f"  平均训练损失: {avg_train_loss:.6f} (±{std_train_loss:.6f})")
                        print(f"  平均验证损失: {avg_test_loss:.6f} (±{std_test_loss:.6f})")

                        trial_result = {
                                'trial': trial_count,
                                'init_method': init_method,
                                'stddev': stddev, 
                                'rho1': rho1,
                                'eta1': eta1,
                                'eta2': eta2,
                                'train_loss': avg_train_loss,
                                'test_loss': avg_test_loss,
                                'train_loss_std': std_train_loss,
                                'test_loss_std': std_test_loss 
                            }
                        trial_results.append(trial_result)
                        # 写入单条记录 
                        writer.writerow(trial_result)
                        print(f"Trial {trial_count} results appended to {filename}")
                            
                        # 更新最佳配置
                        if avg_test_loss < best_score:
                            best_score = avg_test_loss
                            best_config = {
                                'init_method': init_method,
                                'stddev': stddev,
                                'rho1': rho1,
                                'eta1': eta1,
                                'eta2': eta2, 
                                'test_loss': best_score
                            }
        
    print(f"\n总共完成 {trial_count} 次参数组合试验") 
    best_params_df = pd.DataFrame([best_config])
    best_params_df.to_csv('best_SPLQPaveraged3_TIMIT_parameters.csv', index=False) 
                        