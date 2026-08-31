# mypackage/install packages what we need
import subprocess
####
import sys
import importlib
####

# package_dict = {
#     'numpy': 'np',
#     'pandas': 'pd',
#     'scipy': 'scipy',
#     'copy': 'copy',
#     'time': 'time',
#     'random': 'random',
#     'math': 'math',
#     'gc': 'gc'
# }
package_dict = {
    # 模块名 : PyPI包名（必须完全匹配！）
    'numpy': 'numpy',
    'pandas': 'pandas',  # ✅ 修正：必须是 'pandas' 而不是 'pd'
    'scipy': 'scipy',
    'cvxpy': 'cvxpy', 
    'matplotlib': 'matplotlib',
    'csv': 'csv',
    'os': 'os', 
    
    # 标准库模块（会被自动过滤，可保留）
    'copy': 'copy',
    'time': 'time',
    'random': 'random',
    'math': 'math',
    'gc': 'gc'
}

# def check_and_install_packages(package_dict):
#     """
#     Check if the given packages are installed in the current environment.
#     If a package is not installed, it will be installed automatically.
    
#     Args:
#         packages (list): List of package names.
#     """
#     for package in package_dict:
#         try:
#             __import__(package)
#             print(f"{package} is already installed.")
#         except ImportError:
#             print(f"{package} is not installed. Installing...")
#             subprocess.call(['pip', 'install', package, '-y'])
#             print(f"{package} has been installed.")
def check_and_install_packages(package_dict, use_mirror=True):
    """
    智能检查并安装Python包(支持环境隔离+错误重试+国内镜像)
    
    Args:
        package_dict: {导入名: PyPI包名} 字典 (e.g. {"numpy": "numpy"})
        use_mirror: 是否使用清华镜像源 (国内推荐True)
    """
    # 1. 先过滤掉标准库模块
    stdlib_modules = {'sys', 'os', 'time', 'random', 'math', 'copy', 'gc', 'collections', 'itertools'}
    filtered_dict = {imp: pkg for imp, pkg in package_dict.items() 
                    if imp not in stdlib_modules}
    
    # 2. 使用当前环境的pip
    pip_cmd = [sys.executable, '-m', 'pip', 'install']
    if use_mirror:
        pip_cmd += ['-i', 'https://pypi.tuna.tsinghua.edu.cn/simple']
    
    for import_name, package_name in filtered_dict.items():
        try:
            importlib.import_module(import_name)
            print(f"✅ {import_name} 已安装 (v{importlib.import_module(import_name).__version__})")
        except (ImportError, AttributeError):
            print(f"❗ {import_name} 未安装，正在安装 {package_name}...")
            
            # 3. 带重试的安装（解决网络波动）
            max_retries = 3
            for i in range(max_retries):
                try:
                    # 4. 捕获安装输出和错误
                    result = subprocess.run(
                        pip_cmd + [package_name],
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding='utf-8'
                    )
                    print(f"   ✅ 安装成功: {result.stdout.strip()}")
                    break  # 成功则跳出重试循环
                except subprocess.CalledProcessError as e:
                    print(f"   ❌ 安装失败 (尝试 {i+1}/{max_retries}): {e.stderr}")
                    if i == max_retries - 1:
                        raise RuntimeError(f"安装 {package_name} 失败，请手动执行: pip install {package_name}") from None
    
    # 5. 强制刷新模块缓存（关键！）
    importlib.invalidate_caches()

# Ensure all required packages are installed
check_and_install_packages(package_dict)

# Directly import the packages with their aliases
import numpy as np
import pandas as pd
import scipy as scipy
import copy as copy
import time as time
import random as random
import math as math
import gc as gc
import cvxpy as cp
import matplotlib as mpl
import csv as csv
import os as os

# Make these imports available to other modules
__all__ = ['np', 'pd', 'scipy', 'mpl', 'copy', 'time', 'random', 'math', 'gc', 'cp', 'csv', 'os']
