import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats import gaussian_kde # 导入用于计算连续数据众数的库

# 数据 (与你提供的数据一致)
data = {
    'Subject': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4', 'S4', 'S5', 'S5', 'S6', 'S6', 'S7', 'S7'],
    'Trial': ['T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2'],
    'Index_H_OGP': [0.101, 0.099, 0.105, 0.118, 0.078, 0.070, 0.078, 0.091, 0.077, 0.114, 0.056, 0.071, 0.064, 0.069],
    'Index_L_OGP': [0.061, 0.141, 0.048, 0.104, 0.052, 0.047, 0.056, 0.073, 0.057, 0.086, 0.061, 0.048, 0.017, 0.058],
    'Index_H_MGP': [0.085, 0.065, 0.1, 0.110, 0.074, 0.071, 0.072, 0.086, 0.073, 0.110, 0.049, 0.066, 0.052, 0.068],
    'Index_L_MGP': [0.056, 0.057, 0.048, 0.117, 0.051, 0.044, 0.058, 0.042, 0.055, 0.083, 0.052, 0.048, 0.008, 0.052],
    'middle_H_OGP': [0.063, 0.157, 0.110, 0.119, 0.117, 0.107, 0.138, 0.166, 0.071, 0.106, 0.121, 0.135, 0.117, 0.115],
    'middle_L_OGP': [0.066, 0.129, 0.059, 0.020, 0.213, 0.087, 0.031, 0.159, 0.121, 0.037, 0.168, 0.088, 0.117, 0.312],
    'middle_H_MGP': [0.058, 0.145, 0.105, 0.116, 0.116, 0.103, 0.136, 0.156, 0.058, 0.1, 0.093, 0.125, 0.117, 0.111],
    'middle_L_MGP': [0.064, 0.077, 0.053, 0.019, 0.223, 0.048, 0.035, 0.108, 0.120, 0.031, 0.047, 0.085, 0.117, 0.180],
    'ring_H_OGP': [0.110, 0.095, 0.069, 0.077, 0.109, 0.060, 0.100, 0.132, 0.174, 0.181, 0.101, 0.114, 0.090, 0.106],
    'ring_L_OGP': [0.261, 0.142, 0.125, 0.057, 0.107, 0.087, 0.204, 0.257, 0.184, 0.412, 0.167, 0.157, 0.090, 0.111],
    'ring_H_MGP': [0.107, 0.085, 0.067, 0.073, 0.092, 0.058, 0.094, 0.120, 0.171, 0.174, 0.096, 0.113, 0.082, 0.098],
    'ring_L_MGP': [0.226, 0.102, 0.119, 0.048, 0.116, 0.094, 0.104, 0.224, 0.183, 0.4, 0.1, 0.131, 0.082, 0.098]
}
df = pd.DataFrame(data)

# 计算各组的平均 RMSE
df['Avg_RMSE_H_OGP'] = df[['Index_H_OGP', 'middle_H_OGP', 'ring_H_OGP']].mean(axis=1)
df['Avg_RMSE_L_OGP'] = df[['Index_L_OGP', 'middle_L_OGP', 'ring_L_OGP']].mean(axis=1)
df['Avg_RMSE_H_MGP'] = df[['Index_H_MGP', 'middle_H_MGP', 'ring_H_MGP']].mean(axis=1)
df['Avg_RMSE_L_MGP'] = df[['Index_L_MGP', 'middle_L_MGP', 'ring_L_MGP']].mean(axis=1)

# 定义需要比较的列
cols_to_compare = [
    'Avg_RMSE_L_OGP', 'Avg_RMSE_L_MGP',
    'Avg_RMSE_H_OGP', 'Avg_RMSE_H_MGP'
]
group_labels = ['L-OGP', 'L-MGP', 'H-OGP', 'H-MGP']

# --- 计算统计值和执行检验 ---
# 使用一个字典来存储所有统计数据
stats_dict = {}
for col_name, group_label in zip(cols_to_compare, group_labels):
    # 计算均值、中位数和标准差
    mean_val = df[col_name].mean()
    median_val = df[col_name].median()

    # 正确地计算连续数据的众数 (Mode)
    # 1. 拟合高斯核密度估计（KDE）
    kde = gaussian_kde(df[col_name])
    # 2. 生成一个用于评估密度的值范围
    x_vals = np.linspace(df[col_name].min(), df[col_name].max(), 1000)
    # 3. 找到密度最大的值（即众数）
    mode_val = x_vals[np.argmax(kde(x_vals))]
    
    # 将计算结果存储到字典中
    stats_dict[group_label] = [mean_val, median_val, mode_val]

# 创建一个DataFrame来存储结果
results_df = pd.DataFrame(stats_dict, index=['Mean', 'Median', 'Mode', 'Paired t-test'])

# 执行配对 t 检验和 Wilcoxon 检验
t_stat_H, t_p_H = ttest_rel(df['Avg_RMSE_H_OGP'], df['Avg_RMSE_H_MGP'])
t_stat_L, t_p_L = ttest_rel(df['Avg_RMSE_L_OGP'], df['Avg_RMSE_L_MGP'])

# --- 绘制表格 ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off') # 隐藏坐标轴

# 将结果放入表格数据，四舍五入到4位小数
cell_text = results_df.round(4).values.tolist()

# 添加统计检验结果
cell_text.append(['', '', '', '']) # 空白行
cell_text.append(['', f'p = {t_p_H:.4f}', '', f'p = {t_p_L:.4f}'])

# 定义列和行标签
columns = ('L-OGP', 'L-MGP',  'H-OGP',  'H-MGP')
rows = results_df.index.tolist() + ['', 'H vs M (H)']

# 创建表格
table = ax.table(cellText=cell_text,
                 rowLabels=rows,
                 colLabels=columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

ax.set_title('Statistical Comparison of Avg RMSE', fontsize=16, pad=20)

plt.tight_layout()
plt.savefig('comparison_table.pdf', bbox_inches='tight')
plt.show()
