import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from scipy.stats import wilcoxon

plt.rcParams['svg.fonttype'] = 'none' # 确保字体是可编辑的文本，而非路径
plt.rcParams['svg.hashsalt'] = 'mysalt' # 避免 Inkscape 默认将所有对象合并

# 数据 (与你提供的数据一致)
data = {
    'Subject': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4', 'S4', 'S5', 'S5', 'S6', 'S6', 'S7', 'S7'],
    'Trial': ['T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2'],
    'Index_OGP': [0.1, 0.105, 0.102, 0.117, 0.070, 0.077, 0.077, 0.096, 0.077, 0.114, 0.056, 0.070, 0.064, 0.069],
    'Index_H_OGP': [0.101, 0.099, 0.105, 0.118, 0.078, 0.070, 0.078, 0.091, 0.077, 0.114, 0.056, 0.071, 0.064, 0.069],
    'Index_L_OGP': [0.061, 0.141, 0.048, 0.104, 0.052, 0.047, 0.056, 0.073, 0.057, 0.086, 0.061, 0.048, 0.017, 0.058],
    'Index_MGP': [0.083, 0.064, 0.097, 0.11, 0.086, 0.072, 0.072, 0.084, 0.073, 0.109, 0.049, 0.066, 0.052, 0.068],
    'Index_H_MGP': [0.085, 0.065, 0.1, 0.110, 0.074, 0.071, 0.072, 0.086, 0.073, 0.110, 0.049, 0.066, 0.052, 0.068],
    'Index_L_MGP': [0.056, 0.057, 0.048, 0.117, 0.051, 0.044, 0.058, 0.042, 0.055, 0.083, 0.052, 0.048, 0.008, 0.052],
    'Index_R2_OGP': [0.543, 0.594, 0.713, 0.735, 0.812, 0.866, 0.771, 0.680, 0.842, 0.598, 0.870, 0.862, 0.897, 0.869],
    'Index_R2_MGP': [0.685, 0.840, 0.765, 0.740, 0.832, 0.863, 0.802, 0.736, 0.857, 0.628, 0.899, 0.877, 0.933, 0.874],
    'middle_OGP': [0.064, 0.155, 0.108, 0.119, 0.119, 0.117, 0.138, 0.165, 0.073, 0.105, 0.125, 0.134, 0.117, 0.116],
    'middle_H_OGP': [0.063, 0.157, 0.110, 0.119, 0.117, 0.107, 0.138, 0.166, 0.071, 0.106, 0.121, 0.135, 0.117, 0.115],
    'middle_L_OGP': [0.066, 0.129, 0.059, 0.020, 0.213, 0.087, 0.031, 0.159, 0.121, 0.037, 0.168, 0.088, 0.117, 0.312],
    'middle_MGP': [0.059, 0.142, 0.104, 0.114, 0.115, 0.102, 0.133, 0.149, 0.060, 0.100, 0.092, 0.126, 0.112, 0.112],
    'middle_H_MGP': [0.058, 0.145, 0.105, 0.116, 0.116, 0.103, 0.136, 0.156, 0.058, 0.1, 0.093, 0.125, 0.117, 0.111],
    'middle_L_MGP': [0.064, 0.077, 0.053, 0.019, 0.223, 0.048, 0.035, 0.108, 0.120, 0.031, 0.047, 0.085, 0.117, 0.180],
    'middle_R2_OGP': [0.737, 0.438, 0.643, 0.666, 0.480, 0.537, 0.613, 0.378, 0.747, 0.591, 0.299, 0.487, 0.665, 0.686],
    'middle_R2_MGP': [0.773, 0.526, 0.665, 0.682, 0.521, 0.577, 0.642, 0.493, 0.825, 0.629, 0.624, 0.552, 0.693, 0.716],
    'ring_OGP': [0.129, 0.099, 0.075, 0.072, 0.109, 0.083, 0.100, 0.139, 0.174, 0.189, 0.144, 0.116, 0.106, 0.106],
    'ring_H_OGP': [0.110, 0.095, 0.069, 0.077, 0.109, 0.060, 0.100, 0.132, 0.174, 0.181, 0.101, 0.114, 0.090, 0.106],
    'ring_L_OGP': [0.261, 0.142, 0.125, 0.057, 0.107, 0.087, 0.204, 0.257, 0.184, 0.412, 0.167, 0.157, 0.090, 0.111],
    'ring_MGP': [0.121, 0.086, 0.072, 0.072, 0.084, 0.059, 0.094, 0.092, 0.171, 0.184, 0.096, 0.096, 0.082, 0.106],
    'ring_H_MGP': [0.107, 0.085, 0.067, 0.073, 0.092, 0.058, 0.094, 0.120, 0.171, 0.174, 0.096, 0.113, 0.082, 0.098],
    'ring_L_MGP': [0.226, 0.102, 0.119, 0.048, 0.116, 0.094, 0.104, 0.224, 0.183, 0.4, 0.1, 0.131, 0.082, 0.098],
    'ring_R2_OGP': [0.516, 0.648, 0.895, 0.911, 0.801, 0.885, 0.688, 0.693, 0.182, 0.189, 0.582, 0.770, 0.885, 0.799],
    'ring_R2_MGP': [0.574, 0.735, 0.904, 0.921, 0.844, 0.925, 0.727, 0.750, 0.210, 0.231, 0.811, 0.755, 0.903, 0.801]
}
df = pd.DataFrame(data)

# Calculate the average for each sub-category of OGP and MGP
df['Avg_RMSE_OGP'] = df[['Index_OGP', 'middle_OGP', 'ring_OGP']].mean(axis=1)
df['Avg_RMSE_MGP'] = df[['Index_MGP', 'middle_MGP', 'ring_MGP']].mean(axis=1)
df['Avg_R2_OGP'] = df[['Index_R2_OGP', 'middle_R2_OGP', 'ring_R2_OGP']].mean(axis=1)
df['Avg_R2_MGP'] = df[['Index_R2_MGP', 'middle_R2_MGP', 'ring_R2_MGP']].mean(axis=1)

# 结构 (保持不变)
rmse_sets = {
    "Index": ["Index_OGP", "Index_MGP"],
    "Middle": ["middle_OGP", "middle_MGP"],
    "Ring": ["ring_OGP", "ring_MGP"],
    "Avg": ["Avg_RMSE_OGP", "Avg_RMSE_MGP"]
}

r2_sets = {
    "Index": ["Index_R2_OGP", "Index_R2_MGP"],
    "Middle": ["middle_R2_OGP", "middle_R2_MGP"],
    "Ring": ["ring_R2_OGP", "ring_R2_MGP"],
    "Avg": ["Avg_R2_OGP", "Avg_R2_MGP"]
}

# colors = ['#5ca8d9', '#e16b6b', '#6bbf6b'] # 颜色列表
color = '#2ca02c'

def add_stat_annotation(ax, p_value, x1, x2, y_max, color='black'):
    """
    在图上添加统计显著性标注。
    :param ax: Matplotlib axes 对象
    :param p_value: 检验的 p 值
    :param x1: 第一个箱体的 x 坐标
    :param x2: 第二个箱体的 x 坐标
    :param y_max: 箱体上方 y 坐标的基准值
    :param h: 标注线的高度
    :param color: 标注线的颜色
    """
    star_label = ""
    if p_value < 0.001:
        star_label = "***"
    elif p_value < 0.01:
        star_label = "**"
    elif p_value < 0.05:
        star_label = "*"
    elif p_value < 0.1:
        star_label = "†" # 表示趋势显著
    else:
        star_label = "ns" # 不显著

    # 绘制连接线
    h = y_max * 0.02
    ax.plot([x1, x1, x2, x2], [y_max, y_max + h, y_max + h, y_max],
            lw=1.5, c=color)

    # 绘制星号
    ax.text((x1 + x2) * 0.5, y_max + h, star_label, ha='center', va='bottom',
            color=color, fontsize=16, fontweight='bold')

fig, axes = plt.subplots(2, 4, figsize=(8, 10))
fig.patch.set_visible(False)
for ax in axes.flatten():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelleft=False)  # 隐藏左侧 Y 轴标签

# --- RMSE 部分 ---
for ax_idx, (ax, (title, cols)) in enumerate(zip(axes[0], rmse_sets.items())):
    ogp_rmse_positions = [1]
    mgp_rmse_positions = [2]

    # 绘制 OGP (3个颜色分别用)
    y = df[cols[0]]
    bp = ax.boxplot([y], positions=ogp_rmse_positions, widths=0.6, patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=to_rgba(color, alpha=0.2) if ax_idx == 3 else 'none', edgecolor=to_rgba(color, alpha=1.0), linewidth=1.5, linestyle='-'),
                        whiskerprops=dict(color=color, linewidth=1.5),
                        capprops=dict(color=color, linewidth=1.5),
                        medianprops=dict(color=color, linewidth=2,  linestyle='-'))
    # 遍历中位线对象并添加数值标签
    for i, median_line in enumerate(bp['medians']):
        # 获取中位线的 y 坐标（即中位数）
        median_value = median_line.get_ydata()[0]
        
        # 获取中位线的 x 坐标
        x_pos = median_line.get_xdata()[0]
        
        # 在中位线旁边添加文本标签
        ax.text(x_pos, median_value, f'{median_value:.3f}',
                ha='center', va='bottom', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.scatter(np.random.normal(ogp_rmse_positions, 0.05, size=len(y)), y,
                color=color, alpha=0.8, s=20, zorder=3)
    for box in bp['boxes']:
        box.set_linestyle('-')

    # 绘制 MGP (3个颜色分别用 + 虚线 )
    y = df[cols[1]]
    bp = ax.boxplot([y], positions=mgp_rmse_positions, widths=0.6, patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=to_rgba(color, alpha=0.2) if ax_idx == 3 else 'none', edgecolor=to_rgba(color, alpha=1.0), linewidth=1.5, linestyle='--'),
                        whiskerprops=dict(color=color, linewidth=1.5, linestyle='--'),
                        capprops=dict(color=color, linewidth=1.5, linestyle='--'),
                        medianprops=dict(color=color, linewidth=2, linestyle='--'))
    # 遍历中位线对象并添加数值标签
    for i, median_line in enumerate(bp['medians']):
        # 获取中位线的 y 坐标（即中位数）
        median_value = median_line.get_ydata()[0]
        
        # 获取中位线的 x 坐标
        x_pos = median_line.get_xdata()[0]
        
        # 在中位线旁边添加文本标签
        ax.text(x_pos, median_value, f'{median_value:.3f}',
                ha='center', va='bottom', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.scatter(np.random.normal(mgp_rmse_positions, 0.05, size=len(y)), y,
                color=color, alpha=0.8, s=20, zorder=3)
    for box in bp['boxes']:
        box.set_linestyle('--')
    
    # --- 相关性检验和标注 ---
    # OGP vs MGP (Index, Middle, Ring)
    y_max = ax.get_ylim()[1]
    y_max = y_max * 1.05 # 调整标注高度
    
    # 使用 Wilcoxon 秩和检验
    stat, p_value = wilcoxon(df[cols[0]], df[cols[1]])
    add_stat_annotation(ax, p_value, ogp_rmse_positions[0], mgp_rmse_positions[0], y_max, color)

    ax.set_ylim(0.04, 0.2)

# --- R² 部分 ---
for ax_idx, (ax, (title, cols)) in enumerate(zip(axes[1], r2_sets.items())):
    ogp_r2_positions = [1]
    mgp_r2_positions = [2]
    
    # 绘制 OGP R² 箱线图和散点图
    y = df[cols[0]]
    bp = ax.boxplot([y], positions=ogp_r2_positions, widths=0.6, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=to_rgba(color, alpha=0.2) if ax_idx == 3 else 'none', edgecolor=to_rgba(color, alpha=1.0), linewidth=1.5, linestyle='-'),
               whiskerprops=dict(color=color, linewidth=1.5),
               capprops=dict(color=color, linewidth=1.5),
               medianprops=dict(color=color, linewidth=2, linestyle='-'))
     # 遍历中位线对象并添加数值标签
    for i, median_line in enumerate(bp['medians']):
        # 获取中位线的 y 坐标（即中位数）
        median_value = median_line.get_ydata()[0]
        
        # 获取中位线的 x 坐标
        x_pos = median_line.get_xdata()[0]
        
        # 在中位线旁边添加文本标签
        ax.text(x_pos, median_value, f'{median_value:.3f}',
                ha='center', va='bottom', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.scatter(np.random.normal(ogp_r2_positions[0], 0.05, size=len(y)), y,
               color=color, alpha=0.8, s=20, zorder=3)
    for box in bp['boxes']:
            box.set_linestyle('-')

    # 绘制 MGP R² 箱线图和散点图
    y = df[cols[1]]
    bp = ax.boxplot([y], positions=mgp_r2_positions, widths=0.6, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=to_rgba(color, alpha=0.2) if ax_idx == 3 else 'none', edgecolor=to_rgba(color, alpha=1.0), linewidth=1.5, linestyle='--'),
               whiskerprops=dict(color=color, linewidth=1.5, linestyle='--'),
               capprops=dict(color=color, linewidth=1.5, linestyle='--'),
               medianprops=dict(color=color, linewidth=2, linestyle='--'))
    # 遍历中位线对象并添加数值标签
    for i, median_line in enumerate(bp['medians']):
        # 获取中位线的 y 坐标（即中位数）
        median_value = median_line.get_ydata()[0]
        
        # 获取中位线的 x 坐标
        x_pos = median_line.get_xdata()[0]
        
        # 在中位线旁边添加文本标签
        ax.text(x_pos, median_value, f'{median_value:.3f}',
                ha='center', va='bottom', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.scatter(np.random.normal(mgp_r2_positions[0], 0.05, size=len(y)), y,
               color=color, alpha=0.8, s=20, zorder=3) 
    for box in bp['boxes']:
            box.set_linestyle('--')
    
    # --- R² 相关性检验和标注 ---
    y_max = ax.get_ylim()[1]
    y_max = y_max * 0.98
    
    # 使用 Wilcoxon 秩和检验
    stat, p_value = wilcoxon(df[cols[0]], df[cols[1]])
    add_stat_annotation(ax, p_value, ogp_r2_positions[0], mgp_r2_positions[0], y_max, color)

    ax.set_ylim(0.1, 1)

plt.tight_layout()
plt.savefig('rmse_r2_little.pdf', bbox_inches='tight')

plt.show()