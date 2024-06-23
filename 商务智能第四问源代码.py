# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:08:16 2024

@author: SUN
"""

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from collections import defaultdict
import math

# 加载数据
file_path = r'C:\Users\SUN\Desktop\train50k_processed.csv'
data = pd.read_csv(file_path)

# 根据 vehicleType 分成四份数据源
data_source_0 = data[data['vehicleType'] == 0]
data_source_1 = data[data['vehicleType'] == 1]
data_source_2 = data[data['vehicleType'] == 2]
data_source_3 = data[data['vehicleType'] == 3]

# 打印每个数据源的行数以确保它们不是空的
print(f"Data source 0 shape: {data_source_0.shape}")
print(f"Data source 1 shape: {data_source_1.shape}")
print(f"Data source 2 shape: {data_source_2.shape}")
print(f"Data source 3 shape: {data_source_3.shape}")

# 定义目标变量列名
target_column = 'price'

def train_and_evaluate(data, target_column):
    """
    训练模型并评估其性能
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, pred)
    return mape, y_test, pred


# 使用完整数据集训练模型并计算其性能
full_data = pd.concat([data_source_0, data_source_1, data_source_2, data_source_3])
full_data_mape, y_test_full, pred_full = train_and_evaluate(full_data, target_column)
print(f"Full data MAPE: {full_data_mape}")

# 使用单个数据源训练模型并计算其性能
data_sources = {
    'data_source_0': data_source_0,
    'data_source_1': data_source_1,
    'data_source_2': data_source_2,
    'data_source_3': data_source_3
}

# 计算所有组合的MAPE
all_combinations = []
for r in range(1, len(data_sources) + 1):
    all_combinations.extend(combinations(data_sources.keys(), r))

combination_mapes = {}
for comb in all_combinations:
    combined_data = pd.concat([data_sources[name] for name in comb])
    mape, _, _ = train_and_evaluate(combined_data, target_column)
    combination_mapes[comb] = mape
    print(f"Combination {comb} MAPE: {mape}")

# 计算Shapley值
shapley_values = defaultdict(float)
n_sources = len(data_sources)

# 遍历每个数据源组合的贡献
for i in range(1, n_sources + 1):
    for subset in combinations(data_sources.keys(), i):
        combined_data = pd.concat([data_sources[name] for name in subset])
        subset_mape, _, _ = train_and_evaluate(combined_data, target_column)
        
        for source in subset:
            marginal_contribution = (full_data_mape - subset_mape) / math.comb(n_sources - 1, i - 1)
            shapley_values[source] += marginal_contribution

# 归一化Shapley值
shapley_values = {k: v / n_sources for k, v in shapley_values.items()}

# 输出Shapley值
print("Shapley Values:")
for source, value in shapley_values.items():
    print(f"{source}: {value}")


# 可视化Shapley值
plt.figure(figsize=(10, 6))
plt.barh(list(shapley_values.keys()), list(shapley_values.values()), color='skyblue')
plt.xlabel('Shapley Value (MAPE improvement)')
plt.title('Shapley Value Contribution of Each Data Source to Model Performance')
plt.show()

# 可视化每个子集组合的MAPE
plt.figure(figsize=(14, 8))
combinations_labels = [str(comb) for comb in combination_mapes.keys()]
combinations_mapes = list(combination_mapes.values())
plt.barh(combinations_labels, combinations_mapes, color='lightgreen')
plt.xlabel('MAPE')
plt.ylabel('Combinations')
plt.title('MAPE for Each Combination of Data Sources')
plt.show()

# 可视化每个数据源的边际贡献
marginal_contributions = defaultdict(list)

for i in range(1, n_sources + 1):
    for subset in combinations(data_sources.keys(), i):
        combined_data = pd.concat([data_sources[name] for name in subset])
        subset_mape, _, _ = train_and_evaluate(combined_data, target_column)
        
        for source in subset:
            marginal_contribution = (full_data_mape - subset_mape) / math.comb(n_sources - 1, i - 1)
            marginal_contributions[source].append(marginal_contribution)

# 可视化边际贡献
plt.figure(figsize=(10, 6))
for source, contributions in marginal_contributions.items():
    plt.plot(contributions, label=source)

plt.xlabel('Combination Index')
plt.ylabel('Marginal Contribution')
plt.title('Marginal Contributions of Each Data Source')
plt.legend()
plt.show()

# 可视化归一化后的Shapley值
plt.figure(figsize=(10, 6))
plt.barh(list(shapley_values.keys()), list(shapley_values.values()), color='skyblue')
plt.xlabel('Normalized Shapley Value (MAPE improvement)')
plt.title('Normalized Shapley Value Contribution of Each Data Source')
plt.show()

