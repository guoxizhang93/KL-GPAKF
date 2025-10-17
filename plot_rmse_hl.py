import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from scipy.stats import wilcoxon

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

# 结构 (保持不变)
rmse_sets = {
    "Index": ["Index_H_OGP", "Index_L_OGP", "Index_H_MGP", "Index_L_MGP"],
    "Middle": ["middle_H_OGP", "middle_L_OGP", "middle_H_MGP", "middle_L_MGP"],
    "Ring": ["ring_H_OGP", "ring_L_OGP", "ring_H_MGP", "ring_L_MGP"]
}

colors = ['#5ca8d9', '#e16b6b'] # 颜色列表

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
    h = y_max * 0.015
    ax.plot([x1, x1, x2, x2], [y_max, y_max + h, y_max + h, y_max],
            lw=1.5, c=color)

    # 绘制星号
    ax.text((x1 + x2) * 0.5, y_max + h, star_label, ha='center', va='bottom',
            color=color, fontsize=16, fontweight='bold')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_visible(False)
for ax in axes.flatten():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- RMSE 部分 ---
for ax_idx, (ax, (title, cols)) in enumerate(zip(axes, rmse_sets.items())):
    ogp_positions = [1, 4, 7, 10]
    mgp_positions = [2, 5, 8, 11]

    # 绘制 OGP (3个颜色分别用)
    for i, col in enumerate(cols[:2]):
        y = df[col]
        bp = ax.boxplot([y], positions=[ogp_positions[i]], widths=0.6, patch_artist=True, showfliers=False,
                         boxprops=dict(facecolor=to_rgba(colors[i], alpha=0.2) if ax_idx == 3 else 'none', edgecolor=to_rgba(colors[i], alpha=1.0),  linewidth=1.5, linestyle='-'),
                         whiskerprops=dict(color=colors[i], linewidth=1.5, linestyle='-'),
                         capprops=dict(color=colors[i], linewidth=1.5, linestyle='-'),
                         medianprops=dict(color=colors[i], linewidth=2, linestyle='-'))
        ax.scatter(np.random.normal(ogp_positions[i], 0.05, size=len(y)), y,
                   color=colors[i], alpha=0.8, s=20, zorder=3)
        for box in bp['boxes']:
            box.set_linestyle('-')

    # 绘制 MGP (3个颜色分别用 + 虚线 )
    for i, col in enumerate(cols[2:]):
        y = df[col]
        bp = ax.boxplot([y], positions=[mgp_positions[i]], widths=0.6, patch_artist=True, showfliers=False,
                         boxprops=dict(facecolor=to_rgba(colors[i], alpha=0.2) if ax_idx == 3 else 'none', edgecolor=to_rgba(colors[i], alpha=1.0), linewidth=1.5, linestyle='--'),
                         whiskerprops=dict(color=colors[i], linewidth=1.5, linestyle='--'),
                         capprops=dict(color=colors[i], linewidth=1.5, linestyle='--'),
                         medianprops=dict(color=colors[i], linewidth=2, linestyle='--'))
        ax.scatter(np.random.normal(mgp_positions[i], 0.05, size=len(y)), y,
                    color=colors[i], alpha=0.8, s=20, zorder=3)
        for box in bp['boxes']:
            box.set_linestyle('--')
    
    # --- 相关性检验和标注 ---
    y_max = ax.get_ylim()[1]
    y_max = y_max * 0.95 # 调整标注高度
    
    for i in range(2):
        stat, p_value = wilcoxon(df[cols[i]], df[cols[i+2]])
        print(f"{p_value}\n")
        add_stat_annotation(ax, p_value, ogp_positions[i], mgp_positions[i], y_max, colors[i])

plt.tight_layout()
# plt.savefig('rmse_hl.pdf', bbox_inches='tight')
plt.show()
