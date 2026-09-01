# -*- coding: utf-8 -*-
"""
TIMIT audio dataset
专注于为5种SGDs优化方法找到最优初始化和步长参数
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import time
import random
import gc
import math
import pickle

#%% 基础函数定义

def CalculY(x_dataset, A, W, V, b, c, T, Nh, Nx, Ny):
    # This function is used to calculate y^hat by A, W, V, b, c recurrently nor auxi1liary variables h and u
    for t in range(T):
        x_t = x_dataset[t, :].reshape(Nx, 1)
        if t == 0:
            ut_t = V @ x_t + b
            ut = ut_t
            ht_t = np.maximum(ut_t, 0)
            ht = ht_t
            y_hat_t = A @ ht_t + c
            y_hat = y_hat_t.T
        else:
            ut_t = W @ ht_t + V @ x_t + b
            ut = np.concatenate((ut, ut_t), axis=1)
            ht_t = np.maximum(ut_t, 0)
            ht = np.concatenate((ht, ht_t), axis=1)
            y_hat_t = A @ ht_t + c
            y_hat = np.concatenate((y_hat, y_hat_t.T), axis=0)
    u = ut.T.ravel()
    h = ht.T.ravel()
    u = u.reshape(T * Nh, 1)
    h = h.reshape(T * Nh, 1)
    return u, ut, h, ht, y_hat

#%% 回调类定义
class TimeHistory(keras.callbacks.Callback):
    """记录每个epoch时间的回调"""
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.perf_counter()
        
    def on_train_end(self, logs={}):
        self.totaltime = time.perf_counter() - self.totaltime
        
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.perf_counter()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.perf_counter() - self.epoch_time_start)
        
class TrainErrors(keras.callbacks.Callback):
    # Keras callback which to collect time of each epoch
    def on_train_begin(self, logs={}):
        self.trainerr = []
        # Calculate initial error
        V = self.model.layers[0].get_weights()[0].T
        W = self.model.layers[0].get_weights()[1].T
        b = self.model.layers[0].get_weights()[2].reshape(Nh, 1)
        A = self.model.layers[1].get_weights()[0].T
        c = self.model.layers[1].get_weights()[1].reshape(Ny, 1)
        # _, _, _, _, self.y_hat = CalculY(x_trainset, A, W, V, b, c, T, Nh, Ny)
        # initial_train_err = (1/T) * np.sum((self.y_hat - y_trainset) ** 2)
        # self.trainerr.append(initial_train_err)
        self.y_hat = np.empty((N_train, T, Ny))
        for i in range(N_train):
            _, _, _, _, self.y_hat[i,:,:] = CalculY(x_trainset[i], A, W, V, b, c, T, Nh, Nx, Ny) 
        initial_train_err = (1 /(N_train*T)) * np.sum((self.y_hat - y_trainset)**2)
        self.trainerr.append(initial_train_err)
    
    def on_epoch_end(self, epoch, logs={}):
    ##record weighted matrix
        V = self.model.layers[0].get_weights()[0].T
        W = self.model.layers[0].get_weights()[1].T
        b = self.model.layers[0].get_weights()[2].reshape(Nh, 1)
        A = self.model.layers[1].get_weights()[0].T
        c = self.model.layers[1].get_weights()[1].reshape(Ny, 1)
        ##calculate y_hat
        # _, _, _, _, self.y_hat = CalculY(x_trainset, A, W, V, b, c, T, Nh, Ny)
        # self.trainerr.append((1/T)*np.sum((self.y_hat-y_trainset)**2))
        self.y_hat = np.empty((N_train, T, Ny))
        for i in range(N_train):
            _, _, _, _, self.y_hat[i,:,:] = CalculY(x_trainset[i], A, W, V, b, c, T, Nh, Nx, Ny) 
        self.trainerr.append((1 /(N_train*T)) * np.sum((self.y_hat - y_trainset)**2))


class TestErrors(keras.callbacks.Callback):
    # Keras callback which to collect time of each epoch
    def on_train_begin(self, logs={}):
        self.testerr = []
        # Calculate initial error
        V = self.model.layers[0].get_weights()[0].T
        W = self.model.layers[0].get_weights()[1].T
        b = self.model.layers[0].get_weights()[2].reshape(Nh, 1)
        A = self.model.layers[1].get_weights()[0].T
        c = self.model.layers[1].get_weights()[1].reshape(Ny, 1)
        # _, _, _, _, self.y_hat_test = CalculY(x_testset, A, W, V, b, c, T_test, Nh, Ny)
        # _, _, _, _, self.y_hat_test = CalculY(np.concatenate((x_trainset, x_testset), axis=0), A, W, V, b, c, T+T_test, Nh, Ny)
        # self.y_hat_test = self.y_hat_test[T:,]
        # initial_test_err = (1/T_test) * np.sum((self.y_hat_test - y_testset) ** 2)
        # self.testerr.append(initial_test_err)
        self.y_test_hat = np.empty((N_test, T, Ny))
        for i in range(N_test):
            _, _, _, _, self.y_test_hat[i,:,:] = CalculY(x_testset[i], A, W, V, b, c, T, Nh, Nx, Ny) 
        initial_test_err = (1 / (N_test*T)) * np.sum((self.y_test_hat - y_testset)**2)
        self.testerr.append(initial_test_err)
    
    def on_epoch_end(self, epoch, logs={}):
    ##record weighted matrix
        V = self.model.layers[0].get_weights()[0].T
        W = self.model.layers[0].get_weights()[1].T
        b = self.model.layers[0].get_weights()[2].reshape(Nh, 1)
        A = self.model.layers[1].get_weights()[0].T
        c = self.model.layers[1].get_weights()[1].reshape(Ny, 1)
        ##calculate y_hat
        # _, _, _, _, self.y_hat_test = CalculY(x_testset, A, W, V, b, c, T_test, Nh, Ny)
        # _, _, _, _, self.y_hat_test = CalculY(np.concatenate((x_trainset, x_testset), axis=0), A, W, V, b, c, T+T_test, Nh, Ny)
        # self.y_hat_test = self.y_hat_test[T:,]
        # self.testerr.append((1/T_test)*np.sum((self.y_hat_test-y_testset)**2))
        self.y_test_hat = np.empty((N_test, T, Ny))
        for i in range(N_test):
            _, _, _, _, self.y_test_hat[i,:,:] = CalculY(x_testset[i], A, W, V, b, c, T, Nh, Nx, Ny) 
        self.testerr.append((1 / (N_test*T)) * np.sum((self.y_test_hat - y_testset)**2))

class PrintEpochCallback(tf.keras.callbacks.Callback):
    """每50个epoch打印一次进度"""
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1} complete")

#%% 参数设置
Nh, Nx, Ny= 2, 129, 129
T = 126
N = 70

tau = 5e-4
lambda1 = tau / (Nh * Ny)
lambda2 = tau / (Nh * Nh) 
lambda3 = tau / (Nx * Nh)
lambda4 = tau / Nh
lambda5 = tau / Ny

#%% 数据加载和预处理
def load_and_prepare_data():
    """加载并准备数据"""
    # 加载数据
    with open('mixed_fre_timit_mini.pkl', 'rb') as f:
        mixed = pickle.load(f)
    with open('ori_fre_timit_mini.pkl', 'rb') as f:
        ori = pickle.load(f)
    mixed_arr = np.array(mixed)
    ori_arr = np.array(ori)
    
    # 分割训练测试集 
    sizerate_training = 0.7 
    N_train = int(sizerate_training * N) 
    N_test = N- N_train
    # 切片 
    # x_trainset: mixed 的前 N_train 个样本, (N_train, T, Nx)
    x_trainset = mixed_arr[:N_train].transpose(0, 2, 1)
    # x_testset: mixed 的剩余样本, (N_test, T, Nx)
    x_testset = mixed_arr[N_train:].transpose(0, 2, 1)
    # y_trainset: ori 的前 N_train 个样本, (N_train, T, Ny)
    y_trainset = ori_arr[:N_train].transpose(0, 2, 1)
    # y_testset: ori 的剩余样本, (N_test, T, Ny)
    y_testset = ori_arr[N_train:].transpose(0, 2, 1)
     
    
    # 转换为TensorFlow格式
    x_input = tf.convert_to_tensor(x_trainset, dtype=tf.float32)
    x_test_input = tf.convert_to_tensor(x_testset, dtype=tf.float32)
    y_input = tf.convert_to_tensor(y_trainset, dtype=tf.float32)
    y_test_input = tf.convert_to_tensor(y_testset, dtype=tf.float32)
    
    return (x_trainset, y_trainset, x_testset, y_testset, N_train, N_test, 
            x_input, y_input, x_test_input, y_test_input)

#%% 初始化方法定义
def get_initializer(init_method, stddev, seed_offset):
    """根据初始化方法返回对应的初始化器"""
    if init_method == 'Gaussian':
        return tf.keras.initializers.RandomNormal(mean=0, stddev=stddev, seed=1234+seed_offset)
    elif init_method == 'He':
        return tf.keras.initializers.HeNormal(seed=1234+seed_offset)
    elif init_method == 'Glorot':
        return tf.keras.initializers.GlorotNormal(seed=1234+seed_offset)
    elif init_method == 'LeCun':
        return tf.keras.initializers.LecunNormal(seed=1234+seed_offset)

#%% 参数网格定义
def define_parameter_space():
    """定义参数搜索空间"""
    param_space = {
        'init_methods': ['Gaussian'],
        'gaussian_stddevs': [0.001, 0.01, 0.1],
        'learning_rates': [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        'clipnorm_values': [0.5, 1, 2, 4],
        'batch_sizes': [4, 8, 16]
    }
    return param_space

#%% 模型构建和训练函数
def create_and_train_model(optimizer_name, init_method, stddev, lr, clipnorm, batch_size, 
                          x_input, y_input, x_test_input, y_test_input, N_train,
                          eval_mode=False):
    """创建并训练模型 - 10次循环取平均"""
    # 设置随机种子
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # 存储多次训练的结果
    train_losses = []
    val_losses = []
    
    # 进行3次独立训练循环
    num_iterations = 3
    
    for iteration in range(num_iterations):
        print(f"  循环 {iteration + 1}/{num_iterations}")
        
        # 初始化器设置（每次使用不同的种子）
        ini_mach_1 = get_initializer(init_method, stddev, 1 + iteration * 10)
        ini_mach_2 = get_initializer(init_method, stddev, 2 + iteration * 10) 
        ini_mach_3 = get_initializer(init_method, stddev, 3 + iteration * 10)
        
        # 优化器设置
        if optimizer_name == 'GD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, clipnorm=clipnorm, nesterov=False)
        elif optimizer_name == 'GDC':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, clipnorm=clipnorm, nesterov=False)
        elif optimizer_name == 'GDNes':
            # optimizer = tf.keras.optimizers.SGD(learning_rate=lr,  clipnorm=clipnorm, nesterov=True)
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipnorm=clipnorm, nesterov=True)
        elif optimizer_name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif optimizer_name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # 构建模型
        rnn_cell = tf.keras.layers.SimpleRNNCell(Nh, activation='relu', use_bias=True,
                                                kernel_initializer=ini_mach_1,
                                                recurrent_initializer=ini_mach_2,
                                                kernel_regularizer=keras.regularizers.l2(lambda3),
                                                recurrent_regularizer=keras.regularizers.l2(lambda2),
                                                bias_regularizer=keras.regularizers.l2(lambda4))
        
        model = tf.keras.Sequential([
            tf.keras.layers.RNN(rnn_cell, return_sequences=True),
            tf.keras.layers.Dense(units=Ny, use_bias=True,
                                kernel_initializer=ini_mach_3,
                                kernel_regularizer=keras.regularizers.l2(lambda1),
                                bias_regularizer=keras.regularizers.l2(lambda5))
        ])
        
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                     optimizer=optimizer,
                     metrics=[tf.keras.metrics.MeanSquaredError()])
        
        # 训练参数
        epochs = 100 if eval_mode else 300  # 评估模式使用较少epochs
        
        # 第一次前向传播,触发权重初始化, 以保证后续回调函数正常运行
        y = model(x_input)

        # 初始化回调
        timec = TimeHistory()
        tae = TrainErrors()
        tee = TestErrors()
        
        # 训练模型
        if optimizer_name in ['GD', 'GDC', 'GDNes']:
            # 批量梯度下降
            history = model.fit(x_input, y_input,
                                epochs=epochs,
                                batch_size=N_train,
                                validation_data=(x_test_input, y_test_input),
                                verbose=0, shuffle=False,
                                callbacks=[timec, tae, tee, PrintEpochCallback()])
        else:
            # 随机梯度下降 
            history = model.fit(
                x_input, y_input,
                epochs=epochs,
                batch_size=batch_size,      
                shuffle=True,       # 自动在 N 维度随机打乱
                verbose=0,
                validation_data=(x_test_input, y_test_input),
                callbacks=[timec, tae, tee, PrintEpochCallback()]
            )
        
        # 获取本次训练的最终性能
        final_loss = tae.trainerr[-1]
        final_val_loss = tee.testerr[-1]
        
        # 记录结果
        train_losses.append(final_loss)
        val_losses.append(final_val_loss)
        
        # 清理内存
        del model, rnn_cell, history, timec, tae, tee
        gc.collect()
    
    # 计算平均结果
    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    std_train_loss = np.std(train_losses)
    std_val_loss = np.std(val_losses)
    
    print(f"  平均训练损失: {avg_train_loss:.6f} (±{std_train_loss:.6f})")
    print(f"  平均验证损失: {avg_val_loss:.6f} (±{std_val_loss:.6f})")
    
    return {
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'train_loss_std': std_train_loss,
        'val_loss_std': std_val_loss,
        'success': True
    }
#%% 参数调优主函数
def tune_optimizer_parameters(optimizer_name, x_input, y_input, x_test_input, y_test_input, N_train):
    """为指定优化器调优参数 - 使用遍历搜索"""
    print(f"\n{'='*50}")
    print(f"开始调优 {optimizer_name} 参数")
    print(f"{'='*50}")
    
    param_space = define_parameter_space()
    best_score = float('inf')
    best_config = None
    trial_results = []
    trial_count = 0
    
    # 遍历所有初始化方法
    for init_method in param_space['init_methods']:
        # 处理标准差参数
        if init_method == 'Gaussian':
            stddev_choices = param_space['gaussian_stddevs']
        else:
            stddev_choices = [20]  # 其他方法使用默认值
            
        # 遍历标准差选择
        for stddev in stddev_choices:
            # 遍历学习率
            for lr in param_space['learning_rates']:
                # 根据优化器类型处理batch_size和clipnorm
                if optimizer_name in ['GD', 'GDNes']:
                    # GD和GDNes: batch_size固定为N_train, clipnorm固定为None
                    batch_size_choices = [N_train]
                    clipnorm_choices = [None]
                elif optimizer_name == 'GDC':
                    # GDC: batch_size固定为N_train, clipnorm遍历所有值
                    batch_size_choices = [N_train]
                    clipnorm_choices = param_space['clipnorm_values']
                else:
                    # SGD和Adam: batch_size遍历所有值, clipnorm固定为None
                    batch_size_choices = param_space['batch_sizes']
                    clipnorm_choices = [None]
                
                # 遍历batch_size和clipnorm组合
                for batch_size in batch_size_choices:
                    for clipnorm in clipnorm_choices:
                        trial_count += 1
                        print(f"试验 {trial_count}: {init_method}(σ={stddev}), lr={lr}, "
                              f"clipnorm={clipnorm}, batch={batch_size}")
                        
                        # 评估参数配置
                        result = create_and_train_model(
                            optimizer_name, init_method, stddev, lr, clipnorm, batch_size,
                            x_input, y_input, x_test_input, y_test_input, N_train, 
                            eval_mode=True
                        )
                        
                        # 记录结果
                        trial_result = {
                            'trial': trial_count,
                            'init_method': init_method,
                            'stddev': stddev,
                            'learning_rate': lr,
                            'clipnorm': clipnorm,
                            'batch_size': batch_size,
                            'train_loss': result['train_loss'],
                            'val_loss': result['val_loss'],
                            'train_loss_std': result.get('train_loss_std', 0),
                            'val_loss_std': result.get('val_loss_std', 0),
                            'success': result['success']
                        }
                        trial_results.append(trial_result)
                        
                        # 更新最佳配置
                        if result['success'] and result['val_loss'] < best_score:
                            best_score = result['val_loss']
                            best_config = {
                                'init_method': init_method,
                                'stddev': stddev,
                                'learning_rate': lr,
                                'clipnorm': clipnorm,
                                'batch_size': batch_size,
                                'val_loss': best_score
                            }
    
    print(f"\n总共完成 {trial_count} 次参数组合试验")
    return best_config, trial_results

#%% 主执行函数
def main():
    """主执行函数"""
    print("RNN优化器参数自动调优系统")
    print("="*60)
    
    # 加载数据
    global x_trainset, y_trainset, x_testset, y_testset, N_train, N_test
    (x_trainset, y_trainset, x_testset, y_testset, N_train, N_test, 
     x_input, y_input, x_test_input, y_test_input) = load_and_prepare_data()
     
    
    print(f"数据加载完成:")
    print(f"  训练集大小: {N_train}")
    print(f"  测试集大小: {N_test}")
    print(f"  网络结构: Nh={Nh}, Nx={Nx}, Ny={Ny}")
    
    # 要调优的优化器列表
    optimizers = ["GD", "GDC", "GDNes", "SGD", "Adam"]
    # optimizers = ["GD"]
    
    # 存储调优结果
    tuning_results = {}
    all_trial_data = []
    
    # 逐个调优每个优化器
    for optimizer in optimizers:
        best_config, trial_results = tune_optimizer_parameters(
            optimizer, x_input, y_input, x_test_input, y_test_input, N_train
        )
        
        tuning_results[optimizer] = best_config
        
        # 添加优化器标识到试验数据
        for trial in trial_results:
            trial['optimizer'] = optimizer
        all_trial_data.extend(trial_results)
        
        print(f"\n{optimizer} 最佳配置:")
        if best_config:
            for key, value in best_config.items():
                print(f"  {key}: {value}")
        else:
            print("  未能找到有效的参数配置")
    
    # 保存调优结果
    print(f"\n{'='*60}")
    print("保存调优结果...")
    
    # 保存所有试验数据
    trial_df = pd.DataFrame(all_trial_data)
    trial_df.to_csv('parameter_tuning_detailed_results_TIMIT.csv', index=False)
    
    # 保存最佳参数
    best_params = []
    for optimizer, config in tuning_results.items():
        if config:
            best_params.append({
                'optimizer': optimizer,
                **{k: v for k, v in config.items() if k != 'val_loss'},
                'best_val_loss': config.get('val_loss', float('inf'))
            })
    
    if best_params:
        best_params_df = pd.DataFrame(best_params)
        best_params_df.to_csv('best_optimization_parameters_TIMIT.csv', index=False)
        
        print("\n调优完成！结果文件:")
        print("- parameter_tuning_detailed_results.csv: 详细试验记录")
        print("- best_optimization_parameters.csv: 最佳参数配置")
        
        # 显示最佳结果总结
        print(f"\n{'='*60}")
        print("各优化器最佳性能总结:")
        print(f"{'='*60}")
        for _, row in best_params_df.sort_values('best_val_loss').iterrows():
            print(f"{row['optimizer']:8s}: 验证损失 = {row['best_val_loss']:.6f}")
    else:
        print("未能找到任何有效的参数配置")

if __name__ == "__main__":
    main()