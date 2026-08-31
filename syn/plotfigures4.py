import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import glob
import os
import re

# 设置全局字体和样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5  # 坐标轴线宽
plt.rcParams['lines.linewidth'] = 0.6  # 数据线线宽
plt.rcParams['xtick.major.width'] = 0.5  # x轴刻度线宽
plt.rcParams['ytick.major.width'] = 0.5  # y轴刻度线宽
plt.rcParams['xtick.labelsize'] = 8  # x轴刻度文字
plt.rcParams['ytick.labelsize'] = 8  # y轴刻度文字
plt.rcParams['axes.labelsize'] = 10  # 坐标轴标签
plt.rcParams['legend.fontsize'] = 7  # 图例文字
plt.rcParams['legend.frameon'] = True  # 显示图例边框
plt.rcParams['legend.framealpha'] = 0.9  # 边框透明度
plt.rcParams['legend.facecolor'] = 'white' # 背景白色
plt.rcParams['legend.edgecolor'] = 'gray'  # 边框灰色

def extract_optimizer_name(filename):
    """
    从CSV文件名中提取优化器名称
    
    示例:
    'synT10_Errors_GD.csv' -> 'GD'
    'synT10_Errors_Adam.csv' -> 'Adam'
    """
    # 尝试匹配标准命名模式
    match = re.search(r'Errors_([A-Za-z0-9]+)\.csv$', filename)
    if match:
        return match.group(1)
    
    # 尝试其他可能的命名模式
    match = re.search(r'_(\w+)\.csv$', filename)
    if match:
        return match.group(1)
    
    # 如果无法识别，返回文件名（不带扩展名）
    return os.path.splitext(os.path.basename(filename))[0]

def load_data_from_csvs(directory='.', pattern='synT10_Errors_*.csv'):
    """
    从目录中加载所有匹配模式的CSV文件
    
    参数:
    directory : str
        包含CSV文件的目录, 默认为当前目录
    pattern : str
        文件名匹配模式, 默认为synT10_Errors_*.csv
    
    返回:
    data : dict
        优化器名称 -> 包含cpu_time, train_err, test_err的字典
    """
    # 获取所有匹配的CSV文件
    csv_files = glob.glob(os.path.join(directory, pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"未找到匹配 '{pattern}' 的CSV文件")
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    data = {}
    
    for file in csv_files:
        # 读取CSV文件
        df = pd.read_csv(file)
        
        # 提取优化器名称
        optimizer = extract_optimizer_name(file)
        print(f"  从 {file} 加载数据 -> 优化器: {optimizer}")
        
        # 确保必要的列存在
        TrainErrOpt = 'TrainErr' + str(optimizer)
        TestErrOpt = 'TestErr' + str(optimizer)
        TimeOpt = 'Time' + str(optimizer)
        required_cols = ['iterations', TrainErrOpt, TestErrOpt, TimeOpt]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  警告: {file} 缺少列 {missing_cols}，尝试查找替代列名")
            
            # 尝试替代列名
            if 'iterations' not in df.columns and 'iteration' in df.columns:
                df.rename(columns={'iteration': 'iterations'}, inplace=True)
            if TrainErrOpt not in df.columns and 'train_err' + str(optimizer) in df.columns:
                df.rename(columns={'train_err' + str(optimizer): TrainErrOpt}, inplace=True)
            if TestErrOpt not in df.columns and 'test_err' + str(optimizer) in df.columns:
                df.rename(columns={'test_err' + str(optimizer): TestErrOpt}, inplace=True)
            if TimeOpt not in df.columns and 'time' + str(optimizer) in df.columns:
                df.rename(columns={'time' + str(optimizer): TimeOpt}, inplace=True)
            
            # 再次检查
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  错误: 无法找到必要列 {missing_cols}，跳过此文件")
                continue
        
        # 计算累计CPU时间（默认Time记录的是每个epoch的时间）
        # # 如果已经是累计时间，则直接使用
        # 在函数内部，使用 cpu_time 之前，先初始化
        cpu_time = np.array([])  # 添加默认值
        # if len(df[TimeOpt]) > 1 and df[TimeOpt].iloc[1] > df[TimeOpt].iloc[0]:
        #     cpu_time = df[TimeOpt].values
        # else:
        cpu_time = np.cumsum(df[TimeOpt].values)
        
        if optimizer == 'SPLQP':
            optimizer = 'SRPA'  #算法改名了，从'SPLQP'改为'SRPA$_{1}$'后改成了这个
        
        # 存储数据
        data[optimizer] = {
            'cpu_time': cpu_time,
            'iterations': df['iterations'].values if 'iterations' in df.columns else np.arange(len(cpu_time)),
            'train_err': df[TrainErrOpt].values,
            'test_err': df[TestErrOpt].values
        }
    
    if not data:
        raise ValueError("未成功加载任何有效数据")
    
    return data

def plot_iterations_train_error(data, filename=None, title=None):
    """
    绘制基于迭代次数的训练误差对比图
    
    参数:
    data : dict
        包含各优化器数据的字典
    filename : str, optional
        保存文件名，如果为None则显示图形
    title : str, optional
        图表标题
    """
    # 定义优化器名称和样式
    all_optimizers = ['GD', 'GDC', 'GDNes', 'SGD', 'Adam', 'SRPA']    #算法改名了，从'SPLQP'改为'SRPA$_{1}$'后改成了这个
    colors = [ "#87CEFA", "#008000", "#1E90FF", "#BA55D3", "#7B68EE",  '#FF7F0E']
    markers = ['s', '^', 'o', 'v', 'h', 'p']
    linestyles = ['-', '--', '-', '-.', '-',  '-']
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    # 绘制训练误差（基于iterations）
    for i, opt in enumerate(all_optimizers):
        if opt in data and 'iterations' in data[opt]:
            iterations = data[opt]['iterations']
            train_err = data[opt]['train_err']
            
            # 计算均匀分布的标记点
            n = len(iterations)
            markevery_val = np.linspace(0, n-1, 10, dtype=int) if n >= 10 else slice(None)
    
            ax.plot(iterations, train_err, 
                    color=colors[i], 
                    # marker=markers[i], 
                    linestyle=linestyles[i],
                    markersize=3, markerfacecolor='none', markeredgewidth=1, 
                    # markevery=markevery_val, 
                    label=opt, linewidth=1.2)
    
    # 设置y轴为对数刻度
    ax.set_yscale('log')
    
    # # 优化y轴刻度显示 - 使用科学计数法，减少刻度密度
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-2, 2))  # 设置科学计数法的指数范围
    # ax.yaxis.set_major_formatter(formatter)
    # # ax.yaxis.set_minor_formatter(formatter) # 注释后无次刻度标签
    # ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))  # 减少到5个主要刻度
    # # ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=[1.0]))  # 只保留主要次要刻度
    # ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(2,10)))
    # 优化y轴刻度显示 - 使用真正的科学计数法格式 (10^1, 10^0, 10^-1等)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: f'$10^{{{int(np.log10(x))}}}$'))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda x, p: ''))  # 不显示次要刻度标签
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=8))  # 增加刻度数量
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto'))  # 自动次要刻度
    ax.tick_params(which='minor', length=3, width=0.5)
    ax.tick_params(which='major', length=5, width=0.8)
    
    # 网格设置
    ax.grid(True, which="major", ls="-", alpha=0.3, linewidth=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.2, linewidth=0.3)
    
    # 设置标签（去掉标题）
    ax.set_xlabel('Epochs', fontsize=11)
    ax.set_ylabel('Training Error', fontsize=11)
    
    # if title:
    #     ax.set_title(title, fontsize=12)
    
    # 设置轴范围
    if data:
        max_iter = max([max(data[opt]['iterations']) for opt in data.keys() if 'iterations' in data[opt]])
        min_train = min([min(data[opt]['train_err']) for opt in data.keys()])
        max_train = max([max(data[opt]['train_err']) for opt in data.keys()])
        
        ax.set_xlim(0, max_iter + 1)
        ax.set_ylim(min_train*0.5, min(1e5, max_train*1.2))
    
    # 移除原有的图例
    # ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
    
    # 调整布局为图例外部预留空间
    plt.tight_layout()
    
    # 添加两行图例到图片外部顶部
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', 
              frameon=True, framealpha=0.9, fontsize=7, ncol=3, numpoints=1)
    
    # 保存或显示
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"基于迭代次数的训练误差图表已保存为: {filename}")
    else:
        plt.show()
    
    plt.close()

def plot_iterations_test_error(data, filename=None, title=None):
    """
    绘制基于迭代次数的测试误差对比图
    
    参数:
    data : dict
        包含各优化器数据的字典
    filename : str, optional
        保存文件名，如果为None则显示图形
    title : str, optional
        图表标题
    """
    # 定义优化器名称和样式
    all_optimizers = ['GD', 'GDC', 'GDNes', 'SGD', 'Adam', 'SRPA']    #算法改名了，从'SPLQP'改为'SRPA$_{1}$'后改成了这个
    colors = [ "#87CEFA", "#008000", "#1E90FF", "#BA55D3", "#7B68EE",  '#FF7F0E']
    markers = ['s', '^', 'o', 'v', 'h', 'p']
    linestyles = ['-', '--', '-', '-.', '-',  '-']
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    # 绘制测试误差（基于iterations）
    for i, opt in enumerate(all_optimizers):
        if opt in data and 'iterations' in data[opt]:
            iterations = data[opt]['iterations']
            test_err = data[opt]['test_err']
            
            # 计算均匀分布的标记点
            n = len(iterations)
            markevery_val = np.linspace(0, n-1, 10, dtype=int) if n >= 10 else slice(None)

            ax.plot(iterations, test_err, 
                    color=colors[i], 
                    # marker=markers[i], 
                    linestyle=linestyles[i],
                    markersize=3, markerfacecolor='none', markeredgewidth=1,
                    # markevery=markevery_val,
                    label=opt, linewidth=1.2)
    
    # 设置y轴为对数刻度
    ax.set_yscale('log')
    
    # # 优化y轴刻度显示 - 使用科学计数法，减少刻度密度
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-2, 2))  # 设置科学计数法的指数范围
    # ax.yaxis.set_major_formatter(formatter)
    # # ax.yaxis.set_minor_formatter(formatter)
    # ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))  # 减少到5个主要刻度
    # # ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=[1.0]))  # 只保留主要次要刻度
    # ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(2,10)))
    # 优化y轴刻度显示 - 使用真正的科学计数法格式 (10^1, 10^0, 10^-1等)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: f'$10^{{{int(np.log10(x))}}}$'))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda x, p: ''))  # 不显示次要刻度标签
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=8))  # 增加刻度数量
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto'))  # 自动次要刻度
    ax.tick_params(which='minor', length=3, width=0.5)
    ax.tick_params(which='major', length=5, width=0.8)
    
    # 网格设置
    ax.grid(True, which="major", ls="-", alpha=0.3, linewidth=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.2, linewidth=0.3)
    
    # 设置标签（去掉标题）
    ax.set_xlabel('Epochs', fontsize=11)
    ax.set_ylabel('Test Error', fontsize=11)
    
    # if title:
    #     ax.set_title(title, fontsize=12)
    
    # 设置轴范围
    if data:
        max_iter = max([max(data[opt]['iterations']) for opt in data.keys() if 'iterations' in data[opt]])
        min_test = min([min(data[opt]['test_err']) for opt in data.keys()])
        max_test = max([max(data[opt]['test_err']) for opt in data.keys()])
        
        ax.set_xlim(0, max_iter + 1)
        ax.set_ylim(min_test*0.9, min(1e5, max_test*1.5))
        # print(min_test)
        # ax.set_ylim(min_test*0.9, min(1e5, max_test*1.2))
    
    # 移除原有的图例
    # ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
    
    # 调整布局为图例外部预留空间
    plt.tight_layout()
    
    # 添加两行图例到图片外部顶部
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', 
              frameon=True, framealpha=0.9, fontsize=7, ncol=3, numpoints=1)
    
    # 保存或显示
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"基于迭代次数的测试误差图表已保存为: {filename}")
    else:
        plt.show()
    
    plt.close()

def plot_train_error_comparison(data, filename=None, title=None):
    """
    绘制训练误差对比图
    
    参数:
    data : dict
        包含各优化器数据的字典
    filename : str, optional
        保存文件名，如果为None则显示图形
    title : str, optional
        图表标题
    """
    # 定义优化器名称和样式
    all_optimizers = ['GD', 'GDC', 'GDNes', 'SGD', 'Adam', 'SRPA']    #算法改名了，从'SPLQP'改为'SRPA$_{1}$'后改成了这个
    colors = [ "#87CEFA", "#008000", "#1E90FF", "#BA55D3", "#7B68EE",  '#FF7F0E']
    markers = ['s', '^', 'o', 'v', 'h', 'p']
    linestyles = ['-', '--', '-', '-.', '-',  '-']
    
    # 创建单独的训练误差图
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    # 绘制训练误差
    for i, opt in enumerate(all_optimizers):
        if opt in data:
            #计算10个均匀分布的索引
            n = len(data[opt]['cpu_time'])
            markevery_val = np.linspace(0, n-1, 10, dtype=int) if n >= 10 else slice(None)
    
            ax.plot(data[opt]['cpu_time'], data[opt]['train_err'], 
                    color=colors[i], 
                    # marker=markers[i], 
                    linestyle=linestyles[i],
                    markersize=3, markerfacecolor='none', markeredgewidth=1, 
                    # markevery=markevery_val, 
                    label=opt, linewidth=1.2)
    
    # 设置y轴为对数刻度
    ax.set_yscale('log')
    
    # # 优化y轴刻度显示 - 使用科学计数法，减少刻度密度
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-2, 2))  # 设置科学计数法的指数范围
    # ax.yaxis.set_major_formatter(formatter)
    # # ax.yaxis.set_minor_formatter(formatter)
    # ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))  # 减少到5个主要刻度
    # # ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=[1.0]))  # 只保留主要次要刻度
    # ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(2,10)))
    # 优化y轴刻度显示 - 使用真正的科学计数法格式 (10^1, 10^0, 10^-1等)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: f'$10^{{{int(np.log10(x))}}}$'))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda x, p: ''))  # 不显示次要刻度标签
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=8))  # 增加刻度数量
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto'))  # 自动次要刻度
    ax.tick_params(which='minor', length=3, width=0.5)
    ax.tick_params(which='major', length=5, width=0.8)
    
    # 网格设置
    ax.grid(True, which="major", ls="-", alpha=0.3, linewidth=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.2, linewidth=0.3)
    
    # 设置标签（去掉标题）
    ax.set_xlabel('CPU Time (s)', fontsize=11)
    ax.set_ylabel('Training Error', fontsize=11)
    
    # if title:
    #     ax.set_title(title, fontsize=12)
    
    # 设置轴范围
    max_time = max([max(data[opt]['cpu_time']) for opt in data.keys()])
    min_train = min([min(data[opt]['train_err']) for opt in data.keys()])
    max_train = max([max(data[opt]['train_err']) for opt in data.keys()])
    
    ax.set_xlim(-0.1, max_time + 1)
    ax.set_ylim(min_train*0.5, min(1e5, max_train*1.2))
    
    # 移除原有的图例
    # ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
    
    # 调整布局为图例外部预留空间
    plt.tight_layout()
    
    # 添加两行图例到图片外部顶部
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', 
              frameon=True, framealpha=0.9, fontsize=7, ncol=3, numpoints=1)
    
    # 保存或显示
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"训练误差图表已保存为: {filename}")
    else:
        plt.show()
    
    plt.close()

def plot_test_error_comparison(data, filename=None, title=None):
    """
    绘制测试误差对比图
    
    参数:
    data : dict
        包含各优化器数据的字典
    filename : str, optional
        保存文件名，如果为None则显示图形
    title : str, optional
        图表标题
    """
    # 定义优化器名称和样式
    all_optimizers = ['GD', 'GDC', 'GDNes', 'SGD', 'Adam', 'SRPA']    #算法改名了，从'SPLQP'改为'SRPA$_{1}$'后改成了这个
    colors = [ "#87CEFA", "#008000", "#1E90FF", "#BA55D3", "#7B68EE",  '#FF7F0E']
    markers = ['s', '^', 'o', 'v', 'h', 'p']
    linestyles = ['-', '--', '-', '-.', '-',  '-']
    
    # 创建单独的测试误差图
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    # 绘制测试误差
    for i, opt in enumerate(all_optimizers):
        if opt in data:
            #计算10个均匀分布的索引
            n = len(data[opt]['cpu_time'])
            markevery_val = np.linspace(0, n-1, 10, dtype=int) if n >= 10 else slice(None)

            ax.plot(data[opt]['cpu_time'], data[opt]['test_err'], 
                    color=colors[i], 
                    # marker=markers[i], 
                    linestyle=linestyles[i],
                    markersize=3, markerfacecolor='none', markeredgewidth=1,
                    # markevery=markevery_val,
                    label=opt, linewidth=1.2)
    
    # 设置y轴为对数刻度
    ax.set_yscale('log')
    
    # # 优化y轴刻度显示 - 使用科学计数法，减少刻度密度
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-2, 2))  # 设置科学计数法的指数范围
    # ax.yaxis.set_major_formatter(formatter)
    # # ax.yaxis.set_minor_formatter(formatter)
    # ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))  # 减少到5个主要刻度
    # # ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=[1.0]))  # 只保留主要次要刻度
    # ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(2,10)))
    # 优化y轴刻度显示 - 使用真正的科学计数法格式 (10^1, 10^0, 10^-1等)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: f'$10^{{{int(np.log10(x))}}}$'))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda x, p: ''))  # 不显示次要刻度标签
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=8))  # 增加刻度数量
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto'))  # 自动次要刻度
    ax.tick_params(which='minor', length=3, width=0.5)
    ax.tick_params(which='major', length=5, width=0.8)
    
    # 网格设置
    ax.grid(True, which="major", ls="-", alpha=0.3, linewidth=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.2, linewidth=0.3)
    
    # 设置标签（去掉标题）
    ax.set_xlabel('CPU Time (s)', fontsize=11)
    ax.set_ylabel('Test Error', fontsize=11)
    
    # if title:
    #     ax.set_title(title, fontsize=12)
    
    # 设置轴范围
    max_time = max([max(data[opt]['cpu_time']) for opt in data.keys()])
    min_test = min([min(data[opt]['test_err']) for opt in data.keys()])
    max_test = max([max(data[opt]['test_err']) for opt in data.keys()])
    
    ax.set_xlim(-0.1, max_time + 1)
    ax.set_ylim(min_test*0.9, min(1e5, max_test*1.5))
    
    # 移除原有的图例
    # ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
    
    # 调整布局为图例外部预留空间
    plt.tight_layout()
    
    # 添加两行图例到图片外部顶部
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', 
              frameon=True, framealpha=0.9, fontsize=7, ncol=3, numpoints=1)
    
    # 保存或显示
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"测试误差图表已保存为: {filename}")
    else:
        plt.show()
    
    plt.close()

def plot_optimizer_comparison(data, filename=None, title=None):
    """
    绘制优化器性能对比图（保持原有接口兼容性）
    
    参数:
    data : dict
        包含各优化器数据的字典
    filename : str, optional
        保存文件名，如果为None则显示图形
    title : str, optional
        图表标题
    """
    # 为了向后兼容，仍然提供原来的函数接口
    # 但实际调用新的分离函数
    if filename:
        base_name = filename.replace('.png', '')
        train_filename = f"{base_name}_train.png"
        test_filename = f"{base_name}_test.png"
    else:
        train_filename = None
        test_filename = None
    
    # 生成两个独立的图表
    plot_train_error_comparison(data, train_filename, title)
    plot_test_error_comparison(data, test_filename, title)
    
    print("已生成两个独立的图表文件")



# 使用示例
if __name__ == "__main__":
    try:
        # 从当前目录加载所有匹配模式的CSV文件
        data = load_data_from_csvs(
            directory='.', 
            pattern='synT10_Errors_*.csv'
        )
        
        # 生成基于CPU时间的图表
        plot_train_error_comparison(
            data, 
            filename='optimizer_train_comparison.png',
            title='Training Error Comparison (CPU Time)'
        )
        
        plot_test_error_comparison(
            data, 
            filename='optimizer_test_comparison.png',
            title='Test Error Comparison (CPU Time)'
        )
        
        # 生成基于迭代次数的图表
        plot_iterations_train_error(
            data, 
            filename='optimizer_iterations_train.png',
            title='Training Error vs Iterations'
        )
        
        plot_iterations_test_error(
            data, 
            filename='optimizer_iterations_test.png',
            title='Test Error vs Iterations'
        )
        
        print("\n图表生成成功!")
        print("已生成四个图表文件:")
        print("1. optimizer_train_comparison.png - 基于CPU时间的训练误差对比图")
        print("2. optimizer_test_comparison.png - 基于CPU时间的测试误差对比图")
        print("3. optimizer_iterations_train.png - 基于迭代次数的训练误差对比图")
        print("4. optimizer_iterations_test.png - 基于迭代次数的测试误差对比图")
        print("注意: 如果某些优化器的数据缺失，它们将不会显示在图表中")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

 