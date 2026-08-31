import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import os
import re

# 设置全局字体和样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
# plt.rcParams['legend.fontsize'] = 7
# plt.rcParams['legend.frameon'] = True
# plt.rcParams['legend.framealpha'] = 0.9
# plt.rcParams['legend.facecolor'] = 'white'
# plt.rcParams['legend.edgecolor'] = 'gray'


def find_splqp_files(directory='.'):
    """
    查找包含SPLQP数据的CSV文件
    """
    patterns = [
        '*SPLQP*.csv',
        '*splqp*.csv',
        '*Feas*.csv',
        '*feas*.csv'
    ]

    files = []
    for pattern in patterns:
        matches = glob.glob(os.path.join(directory, pattern))
        files.extend(matches)

    files = sorted(list(set(files)))
    return files



def load_splqp_feasibility_data(directory='.'):
    """
    加载SPLQP算法的可行性指标数据，并计算 FeasVi=max{FeasVi_h, FeasVi_v, FeasVi_u}
    """
    splqp_files = find_splqp_files(directory)

    if not splqp_files:
        raise FileNotFoundError("未找到包含SPLQP数据的CSV文件")

    print(f"找到 {len(splqp_files)} 个可能包含SPLQP数据的文件:")
    for f in splqp_files:
        print(f"  - {os.path.basename(f)}")

    data = {}

    for file in splqp_files:
        try:
            df = pd.read_csv(file)

            print(f"\n正在处理文件: {os.path.basename(file)}")
            print(f"列名: {list(df.columns)}")

            target_columns = ['FeasVi_hSPLQP', 'FeasVi_vSPLQP', 'FeasVi_uSPLQP']
            actual_columns = {}

            # 查找 FeasVi_h, FeasVi_v, FeasVi_u 对应列
            for target_col in target_columns:
                for col in df.columns:
                    if target_col.lower() == col.lower():
                        actual_columns[target_col] = col
                        break

            # 检查是否三个列都找到
            missing_cols = [col for col in target_columns if col not in actual_columns]

            if missing_cols:
                print(f"  警告: 文件中缺少列: {missing_cols}")
                continue

            print(f"  找到可行性指标列: {list(actual_columns.values())}")

            # 提取迭代次数
            if 'iterations' in df.columns:
                iterations = df['iterations'].values
            elif 'iteration' in df.columns:
                iterations = df['iteration'].values
            else:
                iterations = np.arange(len(df))
                print(f"  使用默认迭代次数: 0-{len(df)-1}")

            # 计算 FeasVi = max{FeasVi_h, FeasVi_v, FeasVi_u}
            feasvi_h = df[actual_columns['FeasVi_hSPLQP']].values
            feasvi_v = df[actual_columns['FeasVi_vSPLQP']].values
            feasvi_u = df[actual_columns['FeasVi_uSPLQP']].values

            feasvi = np.maximum.reduce([feasvi_h, feasvi_v, feasvi_u])

            # 存储数据
            data[os.path.basename(file)] = {
                'iterations': iterations,
                'FeasVi': feasvi,
                'data': df
            }

            print("  已计算 FeasVi = max{FeasVi_h, FeasVi_v, FeasVi_u}")

        except Exception as e:
            print(f"  错误处理文件 {file}: {str(e)}")
            continue

    if not data:
        raise ValueError("未成功加载任何SPLQP可行性数据")

    return data


def plot_splqp_feasibility_comparison(data, filename=None, title=None, show_xlabel=True):
    """
    绘制 FeasVi 随 iteration 变化的曲线
    """
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    plotted_curves = 0

    # 使用第一个成功加载的文件绘图
    for file_name, file_data in data.items():
        iterations = file_data['iterations']
        feasvi = file_data['FeasVi']

        n = len(iterations)
        markevery_val = np.linspace(0, n - 1, 8, dtype=int) if n >= 8 else slice(None)

        ax.plot(
            iterations,
            feasvi,
            color='#1F77B4',
            # marker='o',
            linestyle='-',
            markersize=4,
            markerfacecolor='none',
            markeredgewidth=1.2,
            # markevery=markevery_val,
            linewidth=1.5
        )

        plotted_curves += 1
        print(f"  成功绘制文件 {file_name} 的 FeasVi 曲线")

        # 只绘制一条曲线，因此绘制第一个文件后退出
        break

    print(f"总共绘制了 {plotted_curves} 条曲线")

    # 设置y轴为对数刻度
    ax.set_yscale('log')

    # 科学计数法格式
    def sci_notation_formatter(x, pos):
        if x <= 0:
            return '0'
        exponent = int(np.log10(x))
        return f'$10^{{{exponent}}}$'

    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(sci_notation_formatter))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda x, p: ''))
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto'))

    ax.tick_params(which='minor', length=3, width=0.5)
    ax.tick_params(which='major', length=5, width=0.8)

    # 网格设置
    ax.grid(True, which="major", ls="-", alpha=0.3, linewidth=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.2, linewidth=0.3)

    # 设置标签
    ax.set_xlabel('Iterations', fontsize=11)
    if not show_xlabel:
        ax.xaxis.get_label().set_color('white') 
        
    ax.set_ylabel('FeasVi', fontsize=11)

    # 设置标题，可选
    if title:
        ax.set_title(title, fontsize=11)

    # 设置x轴范围
    if data:
        max_iter = max([max(file_data['iterations']) for file_data in data.values()])
        ax.set_xlim(0, max_iter + 1)

    # 紧凑布局
    plt.tight_layout()

    # 不添加图例

    # 保存或显示
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"FeasVi曲线图已保存为: {filename}")
    else:
        plt.show()

    plt.close()


def plot_splqp_residual_metrics(data, metric_col, ylabel, filename=None, title=None, show_xlabel=True):
    """
    绘制res_value或res_norm指标图

    参数:
    data : dict
        load_splqp_feasibility_data返回的数据字典
    metric_col : str
        需要绘制的列名，例如 'res_value' 或 'res_norm'
    ylabel : str
        y轴标签
    filename : str, optional
        保存文件名
    title : str, optional
        图标题
    """

    # fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    color = '#D62728'
    marker = 'o'

    plotted = False

    for file_name, file_data in data.items():
        df = file_data['data']

        if metric_col not in df.columns:
            print(f"  文件 {file_name} 中未找到列 {metric_col}")
            continue

        iterations = file_data['iterations']
        y_values = df[metric_col].values

        # 去除NaN，避免绘图异常
        mask = pd.notna(y_values)
        iterations = iterations[mask]
        y_values = y_values[mask]

        if len(y_values) == 0:
            print(f"  文件 {file_name} 中 {metric_col} 全为空，跳过")
            continue

        n = len(iterations)
        markevery_val = np.linspace(0, n - 1, 8, dtype=int) if n >= 8 else slice(None)

        ax.plot(
            iterations,
            y_values,
            color=color,
            # marker=marker,
            linestyle='-',
            markersize=4,
            markerfacecolor='none',
            markeredgewidth=1.2,
            # markevery=markevery_val,
            linewidth=1.5 #,
            # label=metric_col
        )

        plotted = True
        print(f"  成功绘制 {metric_col}，来源文件: {file_name}")

        # 与原代码一致：如果多个CSV都有该列，只使用第一个成功读取的文件
        break

    if not plotted:
        print(f"未找到可绘制的 {metric_col} 数据")
        plt.close()
        return

    # 如果数据均为正数，则使用对数坐标
    if np.all(y_values > 0):
        y_min = np.min(y_values)
        y_max = np.max(y_values)
        
        # 2. 计算动态范围 (Dynamic Range)
        # 避免除以零，增加一个极小值 epsilon
        dynamic_range = y_max / (y_min + 1e-9)
        
        # 3. 设定阈值：如果范围小于 10 倍，用线性；否则用对数
        # 你可以根据需求调整这个阈值，比如 5, 10, 或 100
        if dynamic_range > 10:
            ax.set_yscale('log')

            def sci_notation_formatter(x, pos):
                if x <= 0:
                    return '0'
                exponent = int(np.log10(x))
                return f'$10^{{{exponent}}}$'

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(sci_notation_formatter))
            ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda x, p: ''))
            ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=8))
            ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto'))

    ax.tick_params(which='minor', length=3, width=0.5)
    ax.tick_params(which='major', length=5, width=0.8)

    ax.grid(True, which="major", ls="-", alpha=0.3, linewidth=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.2, linewidth=0.3)

    # x-label 与原来的保持一致
    ax.set_xlabel('Iterations', fontsize=11)
    if not show_xlabel:
        ax.xaxis.get_label().set_color('white')

    # y-label 使用用户指定的数学表达式
    ax.set_ylabel(ylabel, fontsize=11)

    if title:
        ax.set_title(title, fontsize=11)

    ax.set_xlim(0, max(iterations) + 1)

    # ax.legend(
    #     bbox_to_anchor=(0.5, 1.02),
    #     loc='lower center',
    #     frameon=True,
    #     framealpha=0.9,
    #     fontsize=8,
    #     ncol=1,
    #     numpoints=1
    # )

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"{metric_col} 图已保存为: {filename}")
    else:
        plt.show()

    plt.close()


# 使用示例
if __name__ == "__main__":
    try:
        # 加载SPLQP数据
        data = load_splqp_feasibility_data('.')

        # 生成可行性指标综合对比图
        plot_splqp_feasibility_comparison(
            data,
            filename='splqp_feasibility_comparison.png',
            # title='SPLQP Algorithm Feasibility Violations', 
            show_xlabel=False
        )

        # 新增：绘制 res_value 图
        plot_splqp_residual_metrics(
            data,
            metric_col='res_value',
            ylabel=r'$\Phi_{k}( \mathbf{0} ) - \Phi_{k}(s^{k})$',
            filename='splqp_res_value.png',
            title=None, 
            show_xlabel=False
        )

        # 新增：绘制 res_norm 图
        plot_splqp_residual_metrics(
            data,
            metric_col='res_norm',
            ylabel=r'$\|s^{k}\|$',
            filename='splqp_res_norm.png',
            title=None, 
            show_xlabel=False
        )

        # 新增：绘制 funcval_kSPLQP 图
        plot_splqp_residual_metrics(
            data,
            metric_col='funcval_kSPLQP',
            ylabel=r'$\Theta(z^{k})$',
            filename='splqp_funcval.png',
            title=None, 
            show_xlabel=True
        )

        print("\nSPLQP图表生成完成!")
        print("已生成图表:")
        print("  1. splqp_feasibility_comparison.png")
        print("  2. splqp_res_value.png")
        print("  3. splqp_res_norm.png")
        print("  4. splqp_funcval.png")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
